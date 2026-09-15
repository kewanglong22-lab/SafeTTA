#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-8B — final statistical closure

Frozen scope
------------
1) NeoPolyp three-action LOAO:
   paired physical-image clustered bootstrap for
     SHARED_TRANSITION_Q66_DSEM64 - SOURCE_Q66_ONLY
   using the exact frozen R32A1 OOF panel bound by P08A3.

2) Event support:
   - PROMISE12: HARM-positive unique patients per family x action cell.
   - PolypGen unseen MEMO: HARM-positive unique physical images per state
     and pooled.

3) Statistical unit:
   carry forward the P08A-fix2 conclusion that image/frame is the highest
   recoverable PolypGen physical unit unless a higher-level frozen mapping
   had already been found.

No fitting/refitting.
No inference.
No score flip.
No calibration.
No selector change.
No action weighting.
No split-seed selection.
No post-hoc model selection.

The P08A3 semantic binding is authoritative. Historical point estimates are
used only as a parity diagnostic AFTER binding; a mismatch stops execution
before bootstrap and never changes the selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


VERSION = "2026-09-15-B6-P08B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P08_FINAL_STATISTICAL_CLOSURE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "fc231ec059188f44db277940a321f341b2f322b80902664c10889bfb5fe2a147"
)

P08A3_DIR = ROOT / "B6_P08A3_neopolyp_loao_exact_binding_amendment_v1_fix1"
P08A3_LOCK = P08A3_DIR / "B6_P08A3_NEOPOLYP_LOAO_EXACT_BINDING_AMENDMENT.json"
EXPECTED_P08A3_GATE = "PASS_B6_P08A3_NEOPOLYP_LOAO_EXACT_BINDING_AMENDMENT"
EXPECTED_P08A3_STATUS = (
    "FROZEN_LINEAGE_BINDING_BEFORE_P08B_STATISTICAL_CLOSURE_METRICS"
)

P08A_FIX2_DECISION = (
    ROOT / "B6_P08A_final_statistical_closure_asset_binding_audit_v1_fix2"
    / "B6_P08A_FINAL_STATISTICAL_CLOSURE_BINDING.json"
)

R32A1_OOF = (
    ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
    / "R32A1_FIX1_OOF_PREDICTIONS.csv"
)
EXPECTED_R32A1_SHA256 = (
    "9f0aec052ede9a0bb50d886fd9e88d68fcec10acc66d8cac44eeb4fa8611fb02"
)

MRI_PANEL = (
    ROOT / "B6_P01A_mri_matched_representation_attribution_v1"
    / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"
)

POLYPGEN_OUTCOME = (
    ROOT / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MODELCASE_OUTCOMES.csv"
)

OUT_DIR = ROOT / "B6_P08B_final_statistical_closure_v1"

SOURCE_MODEL = "SOURCE_Q66_ONLY"
SHARED_MODEL = "SHARED_TRANSITION_Q66_DSEM64"
EVAL_TYPE = "LOAO"

DIRECTION_TO_ACTION = {
    "PL-CONF90+MEMO-SEG4-1STEP->TENT1": "TENT1",
    "TENT1+MEMO-SEG4-1STEP->PL-CONF90": "PL-CONF90",
    "TENT1+PL-CONF90->MEMO-SEG4-1STEP": "MEMO-SEG4-1STEP",
}
ACTIONS = ["MEMO-SEG4-1STEP", "PL-CONF90", "TENT1"]
SPLIT_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
FOLDS = [0, 1, 2, 3, 4]

EXPECTED_IMAGES = 800
EXPECTED_STATES = 9
EXPECTED_ROWS_PER_SEED_ACTION = 7200
EXPECTED_ROWS_PER_MODEL = 108000
EXPECTED_PAIRED_ROWS = 108000

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260923

# Frozen P08 protocol anchors. These are parity diagnostics only.
ANCHOR_SOURCE_MACRO_AUROC = 0.728190
ANCHOR_SHARED_MACRO_AUROC = 0.741600
ANCHOR_DELTA_MACRO_AUROC = 0.013410
ANCHOR_TOL = 5e-4

PASS_GATE = "PASS_B6_P08B_FINAL_STATISTICAL_CLOSURE_COMPLETE"

REQUIRED_R32_COLUMNS = [
    "split_seed", "fold", "model", "evaluation_type", "direction",
    "sample_id", "model_state_id", "model_family", "action",
    "y_harm", "score",
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


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lineage() -> Dict:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    protocol_sha = sha256_file(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P08 protocol SHA mismatch: expected={EXPECTED_PROTOCOL_SHA256} "
            f"observed={protocol_sha}"
        )

    if not P08A3_LOCK.is_file():
        raise FileNotFoundError(P08A3_LOCK)
    lock = load_json(P08A3_LOCK)
    if lock.get("gate") != EXPECTED_P08A3_GATE:
        raise RuntimeError(f"P08A3 gate changed: {lock.get('gate')!r}")
    if lock.get("status") != EXPECTED_P08A3_STATUS:
        raise RuntimeError(f"P08A3 status changed: {lock.get('status')!r}")

    src = lock.get("frozen_source_state_binding", {})
    shared = lock.get("frozen_shared_transition_binding", {})
    if src.get("model") != SOURCE_MODEL:
        raise RuntimeError("P08A3 SOURCE model binding changed.")
    if shared.get("model") != SHARED_MODEL:
        raise RuntimeError("P08A3 SHARED model binding changed.")
    if src.get("evaluation_type") != EVAL_TYPE or shared.get("evaluation_type") != EVAL_TYPE:
        raise RuntimeError("P08A3 evaluation_type changed.")
    if src.get("split_seeds") != SPLIT_SEEDS or shared.get("split_seeds") != SPLIT_SEEDS:
        raise RuntimeError("P08A3 split seeds changed.")
    if src.get("folds") != FOLDS or shared.get("folds") != FOLDS:
        raise RuntimeError("P08A3 folds changed.")

    # JSON serializes dict order but scientific equality is set/dict equality.
    if src.get("directions") != DIRECTION_TO_ACTION:
        raise RuntimeError("P08A3 SOURCE direction mapping changed.")
    if shared.get("directions") != DIRECTION_TO_ACTION:
        raise RuntimeError("P08A3 SHARED direction mapping changed.")

    if not R32A1_OOF.is_file():
        raise FileNotFoundError(R32A1_OOF)
    r32_sha = sha256_file(R32A1_OOF)
    if r32_sha != EXPECTED_R32A1_SHA256:
        raise RuntimeError(
            f"R32A1 OOF SHA mismatch expected={EXPECTED_R32A1_SHA256} observed={r32_sha}"
        )

    grouping_route = None
    if P08A_FIX2_DECISION.is_file():
        d = load_json(P08A_FIX2_DECISION)
        grouping_route = (
            d.get("PolypGen_grouping", {}).get("route")
        )

    return {
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "p08a3_lock_path": str(P08A3_LOCK),
        "p08a3_lock_sha256": sha256_file(P08A3_LOCK),
        "r32a1_oof_path": str(R32A1_OOF),
        "r32a1_oof_sha256": r32_sha,
        "p08a_fix2_decision_path": (
            str(P08A_FIX2_DECISION) if P08A_FIX2_DECISION.is_file() else None
        ),
        "p08a_fix2_decision_sha256": (
            sha256_file(P08A_FIX2_DECISION) if P08A_FIX2_DECISION.is_file() else None
        ),
        "polypgen_grouping_route": grouping_route,
    }


def read_exact_pair() -> pd.DataFrame:
    header = list(pd.read_csv(R32A1_OOF, nrows=0, low_memory=False).columns)
    missing = [c for c in REQUIRED_R32_COLUMNS if c not in header]
    if missing:
        raise RuntimeError(f"R32A1 missing columns: {missing}")

    d = pd.read_csv(R32A1_OOF, usecols=REQUIRED_R32_COLUMNS, low_memory=False)

    d["split_seed"] = pd.to_numeric(d["split_seed"], errors="raise").astype(int)
    d["fold"] = pd.to_numeric(d["fold"], errors="raise").astype(int)
    d["y_harm"] = pd.to_numeric(d["y_harm"], errors="raise").astype(np.int8)
    d["score"] = pd.to_numeric(d["score"], errors="raise").astype(float)

    keep = (
        (d["evaluation_type"].astype(str) == EVAL_TYPE)
        & (d["model"].astype(str).isin([SOURCE_MODEL, SHARED_MODEL]))
        & (d["direction"].astype(str).isin(DIRECTION_TO_ACTION))
    )
    d = d.loc[keep].copy()

    if len(d) != 2 * EXPECTED_ROWS_PER_MODEL:
        raise RuntimeError(
            f"Expected {2 * EXPECTED_ROWS_PER_MODEL} SOURCE+SHARED rows, got {len(d)}"
        )

    pair_key = [
        "split_seed", "fold", "direction",
        "sample_id", "model_state_id", "model_family", "action",
    ]

    src = (
        d[d["model"].astype(str) == SOURCE_MODEL]
        [pair_key + ["y_harm", "score"]]
        .rename(columns={"y_harm": "y_src", "score": "score_source"})
    )
    shared = (
        d[d["model"].astype(str) == SHARED_MODEL]
        [pair_key + ["y_harm", "score"]]
        .rename(columns={"y_harm": "y_shared", "score": "score_shared"})
    )

    if src.duplicated(pair_key, keep=False).any():
        raise RuntimeError("SOURCE duplicate exact pair keys.")
    if shared.duplicated(pair_key, keep=False).any():
        raise RuntimeError("SHARED duplicate exact pair keys.")

    p = src.merge(
        shared,
        on=pair_key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(p) != EXPECTED_PAIRED_ROWS or not (p["_merge"] == "both").all():
        raise RuntimeError(
            f"Exact pairing failed rows={len(p)} merge={p['_merge'].value_counts().to_dict()}"
        )
    p = p.drop(columns=["_merge"])

    if not np.array_equal(
        p["y_src"].to_numpy(dtype=np.int8),
        p["y_shared"].to_numpy(dtype=np.int8),
    ):
        raise RuntimeError("SOURCE/SHARED HARM labels differ.")
    p = p.rename(columns={"y_src": "y_harm"}).drop(columns=["y_shared"])

    # Direction->held-out action must remain exact.
    for direction, action in DIRECTION_TO_ACTION.items():
        z = p[p["direction"].astype(str) == direction]
        acts = sorted(z["action"].astype(str).unique().tolist())
        if acts != [action]:
            raise RuntimeError(
                f"Direction/action mismatch {direction}: observed={acts}, expected={action}"
            )

    if int(p["sample_id"].nunique()) != EXPECTED_IMAGES:
        raise RuntimeError("NeoPolyp physical-image count drift.")
    if int(p["model_state_id"].nunique()) != EXPECTED_STATES:
        raise RuntimeError("NeoPolyp model-state count drift.")

    return p.reset_index(drop=True)


def binary_metrics(
    y: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)

    if sample_weight is None:
        if np.unique(y).size != 2:
            return np.nan, np.nan
        return (
            float(roc_auc_score(y, score)),
            float(average_precision_score(y, score)),
        )

    w = np.asarray(sample_weight, dtype=float)
    present = w > 0
    if not np.any(present):
        return np.nan, np.nan
    yy = y[present]
    ss = score[present]
    ww = w[present]
    if np.unique(yy).size != 2:
        return np.nan, np.nan
    return (
        float(roc_auc_score(yy, ss, sample_weight=ww)),
        float(average_precision_score(yy, ss, sample_weight=ww)),
    )


def point_metrics(pair: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cell_rows = []

    for seed in SPLIT_SEEDS:
        for action in ACTIONS:
            g = pair[
                (pair["split_seed"] == seed)
                & (pair["action"].astype(str) == action)
            ].copy()

            if len(g) != EXPECTED_ROWS_PER_SEED_ACTION:
                raise RuntimeError(
                    f"seed={seed} action={action}: rows={len(g)} "
                    f"expected={EXPECTED_ROWS_PER_SEED_ACTION}"
                )
            if int(g["sample_id"].nunique()) != EXPECTED_IMAGES:
                raise RuntimeError(f"seed={seed} action={action}: image count drift.")
            if int(g["model_state_id"].nunique()) != EXPECTED_STATES:
                raise RuntimeError(f"seed={seed} action={action}: state count drift.")

            y = g["y_harm"].to_numpy(np.int8)
            a_src, p_src = binary_metrics(y, g["score_source"].to_numpy(float))
            a_sh, p_sh = binary_metrics(y, g["score_shared"].to_numpy(float))

            if not all(np.isfinite([a_src, p_src, a_sh, p_sh])):
                raise RuntimeError(
                    f"Point metric single class/invalid seed={seed} action={action}"
                )

            cell_rows.append({
                "split_seed": seed,
                "heldout_action": action,
                "rows": int(len(g)),
                "physical_images": int(g["sample_id"].nunique()),
                "model_states": int(g["model_state_id"].nunique()),
                "harm_n": int(y.sum()),
                "harm_prevalence": float(y.mean()),
                "source_auroc": a_src,
                "shared_auroc": a_sh,
                "delta_auroc_shared_minus_source": a_sh - a_src,
                "source_auprc": p_src,
                "shared_auprc": p_sh,
                "delta_auprc_shared_minus_source": p_sh - p_src,
            })

    cells = pd.DataFrame(cell_rows)

    per_action = (
        cells.groupby("heldout_action", sort=True)
        .agg(
            source_auroc=("source_auroc", "mean"),
            shared_auroc=("shared_auroc", "mean"),
            delta_auroc_shared_minus_source=(
                "delta_auroc_shared_minus_source", "mean"
            ),
            source_auprc=("source_auprc", "mean"),
            shared_auprc=("shared_auprc", "mean"),
            delta_auprc_shared_minus_source=(
                "delta_auprc_shared_minus_source", "mean"
            ),
        )
        .reset_index()
    )

    summary = {
        "source_macro_action_auroc": float(per_action["source_auroc"].mean()),
        "shared_macro_action_auroc": float(per_action["shared_auroc"].mean()),
        "delta_macro_action_auroc": float(
            per_action["delta_auroc_shared_minus_source"].mean()
        ),
        "source_macro_action_auprc": float(per_action["source_auprc"].mean()),
        "shared_macro_action_auprc": float(per_action["shared_auprc"].mean()),
        "delta_macro_action_auprc": float(
            per_action["delta_auprc_shared_minus_source"].mean()
        ),
    }

    # Historical anchor parity is a diagnostic only.
    anchor = {
        "source_anchor": ANCHOR_SOURCE_MACRO_AUROC,
        "shared_anchor": ANCHOR_SHARED_MACRO_AUROC,
        "delta_anchor": ANCHOR_DELTA_MACRO_AUROC,
        "source_abs_diff": abs(
            summary["source_macro_action_auroc"] - ANCHOR_SOURCE_MACRO_AUROC
        ),
        "shared_abs_diff": abs(
            summary["shared_macro_action_auroc"] - ANCHOR_SHARED_MACRO_AUROC
        ),
        "delta_abs_diff": abs(
            summary["delta_macro_action_auroc"] - ANCHOR_DELTA_MACRO_AUROC
        ),
        "tolerance": ANCHOR_TOL,
    }
    anchor["pass"] = bool(
        anchor["source_abs_diff"] <= ANCHOR_TOL
        and anchor["shared_abs_diff"] <= ANCHOR_TOL
        and anchor["delta_abs_diff"] <= ANCHOR_TOL
    )
    summary["anchor_parity"] = anchor

    return cells, per_action, summary


def prepare_boot_cells(pair: pd.DataFrame) -> Tuple[List[str], Dict[Tuple[int, str], Dict]]:
    image_ids = sorted(pair["sample_id"].astype(str).unique().tolist())
    if len(image_ids) != EXPECTED_IMAGES:
        raise RuntimeError("Bootstrap image universe changed.")
    image_to_i = {x: i for i, x in enumerate(image_ids)}

    cells = {}
    for seed in SPLIT_SEEDS:
        for action in ACTIONS:
            g = pair[
                (pair["split_seed"] == seed)
                & (pair["action"].astype(str) == action)
            ].copy()

            g["_image_i"] = g["sample_id"].astype(str).map(image_to_i).astype(int)
            counts = g.groupby("_image_i").size()
            if len(counts) != EXPECTED_IMAGES or not (counts.to_numpy() == EXPECTED_STATES).all():
                raise RuntimeError(
                    f"seed={seed} action={action}: expected exactly "
                    f"{EXPECTED_STATES} state rows per image."
                )

            cells[(seed, action)] = {
                "y": g["y_harm"].to_numpy(np.int8),
                "source": g["score_source"].to_numpy(float),
                "shared": g["score_shared"].to_numpy(float),
                "image_i": g["_image_i"].to_numpy(int),
            }

    return image_ids, cells


def bootstrap(pair: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_ids, cells = prepare_boot_cells(pair)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    reps = []

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P08B NeoPolyp paired image bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        draw = rng.integers(0, len(image_ids), size=len(image_ids))
        multiplicity = np.bincount(draw, minlength=len(image_ids)).astype(float)

        action_seed = {
            action: {
                "source_auroc": [],
                "shared_auroc": [],
                "delta_auroc": [],
                "source_auprc": [],
                "shared_auprc": [],
                "delta_auprc": [],
            }
            for action in ACTIONS
        }

        for seed in SPLIT_SEEDS:
            for action in ACTIONS:
                c = cells[(seed, action)]
                w = multiplicity[c["image_i"]]

                a_src, p_src = binary_metrics(c["y"], c["source"], sample_weight=w)
                a_sh, p_sh = binary_metrics(c["y"], c["shared"], sample_weight=w)

                if np.isfinite(a_src) and np.isfinite(a_sh):
                    action_seed[action]["source_auroc"].append(a_src)
                    action_seed[action]["shared_auroc"].append(a_sh)
                    action_seed[action]["delta_auroc"].append(a_sh - a_src)

                if np.isfinite(p_src) and np.isfinite(p_sh):
                    action_seed[action]["source_auprc"].append(p_src)
                    action_seed[action]["shared_auprc"].append(p_sh)
                    action_seed[action]["delta_auprc"].append(p_sh - p_src)

        row = {"rep": rep}

        valid_macro_auc = True
        valid_macro_ap = True
        action_auc_deltas = []
        action_ap_deltas = []

        for action in ACTIONS:
            safe = action.replace("-", "_")

            # A valid action summary requires all five frozen split seeds.
            if len(action_seed[action]["delta_auroc"]) == len(SPLIT_SEEDS):
                da = float(np.mean(action_seed[action]["delta_auroc"]))
                sa = float(np.mean(action_seed[action]["source_auroc"]))
                ha = float(np.mean(action_seed[action]["shared_auroc"]))
                action_auc_deltas.append(da)
                row[f"{safe}__source_auroc"] = sa
                row[f"{safe}__shared_auroc"] = ha
                row[f"{safe}__delta_auroc"] = da
            else:
                valid_macro_auc = False
                row[f"{safe}__source_auroc"] = np.nan
                row[f"{safe}__shared_auroc"] = np.nan
                row[f"{safe}__delta_auroc"] = np.nan

            if len(action_seed[action]["delta_auprc"]) == len(SPLIT_SEEDS):
                dp = float(np.mean(action_seed[action]["delta_auprc"]))
                sp = float(np.mean(action_seed[action]["source_auprc"]))
                hp = float(np.mean(action_seed[action]["shared_auprc"]))
                action_ap_deltas.append(dp)
                row[f"{safe}__source_auprc"] = sp
                row[f"{safe}__shared_auprc"] = hp
                row[f"{safe}__delta_auprc"] = dp
            else:
                valid_macro_ap = False
                row[f"{safe}__source_auprc"] = np.nan
                row[f"{safe}__shared_auprc"] = np.nan
                row[f"{safe}__delta_auprc"] = np.nan

        row["macro_action_delta_auroc"] = (
            float(np.mean(action_auc_deltas))
            if valid_macro_auc and len(action_auc_deltas) == len(ACTIONS)
            else np.nan
        )
        row["macro_action_delta_auprc"] = (
            float(np.mean(action_ap_deltas))
            if valid_macro_ap and len(action_ap_deltas) == len(ACTIONS)
            else np.nan
        )

        reps.append(row)

    reps_df = pd.DataFrame(reps)

    summary_rows = []

    def summarize(metric_name: str, values: np.ndarray, scope: str, action: str | None):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < int(0.90 * BOOTSTRAP_REPS):
            raise RuntimeError(
                f"Too few valid bootstrap reps for {metric_name}: "
                f"{len(values)}/{BOOTSTRAP_REPS}"
            )
        summary_rows.append({
            "scope": scope,
            "heldout_action": action,
            "metric": metric_name,
            "requested_reps": BOOTSTRAP_REPS,
            "valid_reps": int(len(values)),
            "bootstrap_mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "ci_excludes_zero": bool(
                np.quantile(values, 0.025) > 0
                or np.quantile(values, 0.975) < 0
            ),
        })

    summarize(
        "delta_auroc_shared_minus_source",
        reps_df["macro_action_delta_auroc"].to_numpy(float),
        "MACRO_3_ACTION",
        None,
    )
    summarize(
        "delta_auprc_shared_minus_source",
        reps_df["macro_action_delta_auprc"].to_numpy(float),
        "MACRO_3_ACTION",
        None,
    )

    for action in ACTIONS:
        safe = action.replace("-", "_")
        summarize(
            "delta_auroc_shared_minus_source",
            reps_df[f"{safe}__delta_auroc"].to_numpy(float),
            "PER_ACTION",
            action,
        )
        summarize(
            "delta_auprc_shared_minus_source",
            reps_df[f"{safe}__delta_auprc"].to_numpy(float),
            "PER_ACTION",
            action,
        )

    return reps_df, pd.DataFrame(summary_rows)


def mri_event_support() -> Tuple[pd.DataFrame, Dict]:
    if not MRI_PANEL.is_file():
        raise FileNotFoundError(MRI_PANEL)

    d = pd.read_csv(MRI_PANEL, low_memory=False)
    required = ["family", "action", "representation", "harm"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"MRI event-support panel missing {missing}")

    patient_col = "case_key" if "case_key" in d.columns else (
        "case_id" if "case_id" in d.columns else None
    )
    if patient_col is None:
        raise RuntimeError("MRI panel has no case_key/case_id patient identifier.")

    # Use one representation only to avoid counting representation duplicates.
    d = d[d["representation"].astype(str) == "FULL_TRANSITION_SAFETTA"].copy()
    if len(d) != 8262:
        raise RuntimeError(f"MRI full-transition rows={len(d)} expected=8262")

    d["harm"] = pd.to_numeric(d["harm"], errors="raise").astype(int)
    d["_patient"] = d[patient_col].astype(str)

    rows = []
    for (family, action), g in d.groupby(["family", "action"], sort=True):
        harm = g[g["harm"] == 1]
        rows.append({
            "family": str(family),
            "action": str(action),
            "rows": int(len(g)),
            "harm_rows": int(len(harm)),
            "harm_prevalence": float(g["harm"].mean()),
            "unique_patients": int(g["_patient"].nunique()),
            "harm_positive_unique_patients": int(harm["_patient"].nunique()),
        })

    out = pd.DataFrame(rows)
    if len(out) != 6:
        raise RuntimeError(f"MRI event-support cells={len(out)} expected=6")

    return out, {
        "path": str(MRI_PANEL),
        "sha256": sha256_file(MRI_PANEL),
        "representation": "FULL_TRANSITION_SAFETTA",
        "patient_column": patient_col,
        "rows": int(len(d)),
        "patients": int(d["_patient"].nunique()),
        "harm_rows": int((d["harm"] == 1).sum()),
        "harm_prevalence": float(d["harm"].mean()),
        "harm_positive_patients_pooled": int(
            d.loc[d["harm"] == 1, "_patient"].nunique()
        ),
    }


def polypgen_event_support() -> Tuple[pd.DataFrame, Dict]:
    if not POLYPGEN_OUTCOME.is_file():
        raise FileNotFoundError(POLYPGEN_OUTCOME)

    d = pd.read_csv(POLYPGEN_OUTCOME, low_memory=False)
    required = ["sample_id", "model_state_id", "memo_harm_label", "memo_delta_dice"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"PolypGen event-support panel missing {missing}")

    if len(d) != 4596:
        raise RuntimeError(f"PolypGen rows={len(d)} expected=4596")
    if int(d["sample_id"].astype(str).nunique()) != 1532:
        raise RuntimeError("PolypGen physical-image count changed.")

    d["memo_harm_label"] = pd.to_numeric(
        d["memo_harm_label"], errors="raise"
    ).astype(int)
    d["memo_delta_dice"] = pd.to_numeric(
        d["memo_delta_dice"], errors="raise"
    ).astype(float)

    y_from_delta = (d["memo_delta_dice"].to_numpy(float) <= -0.02).astype(int)
    if not np.array_equal(y_from_delta, d["memo_harm_label"].to_numpy(int)):
        raise RuntimeError("PolypGen primary -0.02 HARM parity failed.")

    d["_image"] = d["sample_id"].astype(str)

    rows = []
    for state, g in d.groupby("model_state_id", sort=True):
        harm = g[g["memo_harm_label"] == 1]
        rows.append({
            "model_state_id": str(state),
            "rows": int(len(g)),
            "harm_rows": int(len(harm)),
            "harm_prevalence": float(g["memo_harm_label"].mean()),
            "unique_physical_images": int(g["_image"].nunique()),
            "harm_positive_unique_physical_images": int(harm["_image"].nunique()),
        })

    harm = d[d["memo_harm_label"] == 1]
    rows.append({
        "model_state_id": "POOLED_3_STATES",
        "rows": int(len(d)),
        "harm_rows": int(len(harm)),
        "harm_prevalence": float(d["memo_harm_label"].mean()),
        "unique_physical_images": int(d["_image"].nunique()),
        "harm_positive_unique_physical_images": int(harm["_image"].nunique()),
    })

    return pd.DataFrame(rows), {
        "path": str(POLYPGEN_OUTCOME),
        "sha256": sha256_file(POLYPGEN_OUTCOME),
        "rows": int(len(d)),
        "physical_images": int(d["_image"].nunique()),
        "states": int(d["model_state_id"].astype(str).nunique()),
        "harm_rows": int((d["memo_harm_label"] == 1).sum()),
        "harm_prevalence": float(d["memo_harm_label"].mean()),
        "harm_positive_unique_physical_images_pooled": int(
            harm["_image"].nunique()
        ),
        "harm_label_parity_at_minus_0p02": True,
    }


def self_test() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    s0 = np.array([0.1, 0.2, 0.8, 0.9])
    s1 = np.array([0.05, 0.1, 0.9, 0.95])
    a0, p0 = binary_metrics(y, s0)
    a1, p1 = binary_metrics(y, s1)
    assert abs(a0 - 1.0) < 1e-12
    assert abs(a1 - 1.0) < 1e-12
    assert abs(p0 - 1.0) < 1e-12
    assert abs(p1 - 1.0) < 1e-12

    # Integer bootstrap multiplicities must be valid sample weights.
    w = np.array([2, 0, 1, 1], dtype=float)
    aw, pw = binary_metrics(y, s0, sample_weight=w)
    assert np.isfinite(aw) and np.isfinite(pw)

    assert BOOTSTRAP_REPS == 2000
    assert BOOTSTRAP_SEED == 20260923
    assert SOURCE_MODEL == "SOURCE_Q66_ONLY"
    assert SHARED_MODEL == "SHARED_TRANSITION_Q66_DSEM64"

    print("SELF_TEST_METRICS=PASS")
    print("SELF_TEST_WEIGHTED_CLUSTER_BOOTSTRAP=PASS")
    print("SELF_TEST_FROZEN_BINDING=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA P08B final statistical closure: paired NeoPolyp LOAO "
            "bootstrap + MRI/PolypGen event-support audit."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify lineage, exact row pairing, point metrics, and historical "
            "anchor parity. Do not run bootstrap or write the formal stage."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 176)
    print("SafeTTA B6-P0-8B — final statistical closure")
    print("Version              :", VERSION)
    print("NeoPolyp paired reps :", BOOTSTRAP_REPS)
    print("Bootstrap cluster    : physical image / sample_id")
    print("Model fitting        : NO")
    print("Inference            : NO")
    print("Calibration/score flip: NO")
    print("Selector change      : NO")
    print("=" * 176)

    print("\n[1/6] Verify frozen lineage")
    lineage = verify_lineage()
    for k, v in lineage.items():
        print(f"{k} = {v}")
    print("LINEAGE=PASS")

    print("\n[2/6] Bind exact SOURCE/SHARED NeoPolyp OOF rows")
    pair = read_exact_pair()
    print("paired_rows =", len(pair))
    print("physical_images =", pair["sample_id"].astype(str).nunique())
    print("model_states =", pair["model_state_id"].astype(str).nunique())
    print("actions =", sorted(pair["action"].astype(str).unique().tolist()))
    print("PAIR_BINDING=PASS")

    print("\n[3/6] Point metrics + historical anchor parity")
    point_cells, point_actions, point_summary = point_metrics(pair)
    print("\nPER SEED x HELD-OUT ACTION")
    print(point_cells.to_string(index=False))
    print("\nPER ACTION, MEAN OVER 5 FROZEN SPLIT SEEDS")
    print(point_actions.to_string(index=False))
    print("\nMACRO SUMMARY")
    print(json.dumps(point_summary, indent=2))

    if not point_summary["anchor_parity"]["pass"]:
        raise RuntimeError(
            "Historical point-anchor parity failed AFTER exact semantic binding. "
            "STOP before bootstrap. Do not change selectors."
        )
    print("HISTORICAL_POINT_ANCHOR_PARITY=PASS")

    print("\n[4/6] Event-support preflight")
    mri_events, mri_meta = mri_event_support()
    poly_events, poly_meta = polypgen_event_support()
    print("\nMRI EVENT SUPPORT")
    print(mri_events.to_string(index=False))
    print("\nPOLYPGEN EVENT SUPPORT")
    print(poly_events.to_string(index=False))
    print("EVENT_SUPPORT_BINDING=PASS")

    if args.preflight_only:
        print("\nP08B_PREFLIGHT=PASS")
        print("BOOTSTRAP=NOT_RUN")
        print("FORMAL_STAGE=NOT_WRITTEN")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P08B output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[5/6] 2000-replicate paired physical-image clustered bootstrap")
    reps, boot_summary = bootstrap(pair)
    print("\nPAIRED BOOTSTRAP SUMMARY")
    print(boot_summary.to_string(index=False))

    print("\n[6/6] Write final statistical-closure artifacts")
    p_cells = OUT_DIR / "B6_P08B_NEO_LOAO_POINT_BY_SEED_ACTION.csv"
    p_actions = OUT_DIR / "B6_P08B_NEO_LOAO_POINT_BY_ACTION.csv"
    p_reps = OUT_DIR / "B6_P08B_NEO_LOAO_PAIRED_BOOTSTRAP_REPLICATES.csv"
    p_boot = OUT_DIR / "B6_P08B_NEO_LOAO_PAIRED_BOOTSTRAP_SUMMARY.csv"
    p_mri = OUT_DIR / "B6_P08B_MRI_HARM_EVENT_SUPPORT.csv"
    p_poly = OUT_DIR / "B6_P08B_POLYPGEN_HARM_EVENT_SUPPORT.csv"

    point_cells.to_csv(p_cells, index=False)
    point_actions.to_csv(p_actions, index=False)
    reps.to_csv(p_reps, index=False)
    boot_summary.to_csv(p_boot, index=False)
    mri_events.to_csv(p_mri, index=False)
    poly_events.to_csv(p_poly, index=False)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "lineage": lineage,
        "neoPolyp_LOAO": {
            "source_model": SOURCE_MODEL,
            "shared_model": SHARED_MODEL,
            "evaluation_type": EVAL_TYPE,
            "direction_to_heldout_action": DIRECTION_TO_ACTION,
            "split_seeds": SPLIT_SEEDS,
            "folds": FOLDS,
            "physical_images": EXPECTED_IMAGES,
            "model_states": EXPECTED_STATES,
            "paired_oof_rows": EXPECTED_PAIRED_ROWS,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "point_summary": point_summary,
            "bootstrap_summary": boot_summary.to_dict(orient="records"),
        },
        "MRI_event_support": {
            "meta": mri_meta,
            "rows": mri_events.to_dict(orient="records"),
        },
        "PolypGen_event_support": {
            "meta": poly_meta,
            "rows": poly_events.to_dict(orient="records"),
            "highest_recoverable_statistical_unit": (
                "physical image/frame"
                if lineage.get("polypgen_grouping_route")
                == "IMAGE_FRAME_IS_HIGHEST_RECOVERABLE_UNIT_IN_CURRENT_ASSETS"
                else lineage.get("polypgen_grouping_route")
            ),
        },
        "scientific_constraints": {
            "refit": False,
            "inference": False,
            "target_calibration": False,
            "score_flip": False,
            "selector_change": False,
            "action_weighting": False,
            "split_seed_selection": False,
            "performance_based_binding": False,
        },
        "artifacts": {},
    }

    for p in [p_cells, p_actions, p_reps, p_boot, p_mri, p_poly]:
        audit["artifacts"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    p_audit = OUT_DIR / "B6_P08B_FINAL_STATISTICAL_CLOSURE_AUDIT.json"
    p_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "=" * 176,
        "SafeTTA B6-P08B FINAL STATISTICAL CLOSURE COMPLETE",
        "",
        "NEOPOLYP LOAO POINT SUMMARY",
        json.dumps(point_summary, indent=2),
        "",
        "NEOPOLYP PAIRED BOOTSTRAP SUMMARY",
        boot_summary.to_string(index=False),
        "",
        "MRI HARM EVENT SUPPORT",
        mri_events.to_string(index=False),
        "",
        "POLYPGEN HARM EVENT SUPPORT",
        poly_events.to_string(index=False),
        "",
        "PolypGen highest recoverable statistical unit:",
        str(audit["PolypGen_event_support"]["highest_recoverable_statistical_unit"]),
        "",
        "No refit/inference/calibration/score flip/selector change: YES",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 176,
        "",
    ]
    p_report = OUT_DIR / "B6_P08B_FINAL_STATISTICAL_CLOSURE_REPORT.txt"
    p_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
