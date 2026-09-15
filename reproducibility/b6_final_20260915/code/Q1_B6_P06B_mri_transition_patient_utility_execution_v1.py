#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-6B — PROMISE12 MRI candidate-conditioned patient-level utility.

This stage is bound to the already-frozen P06 protocol and the P06A asset audit.

It:
- re-counts SOURCE/candidate TP/FP/FN from frozen FinalB1 packed masks
  against the already revealed/frozen CM6 PROMISE12 packed GT;
- verifies exact slice-Dice / DeltaDice / HARM parity against FinalB3;
- binds SOURCE-state, simple-geometry, and full-transition scores from the
  frozen long-format B6-P01A matched representation table;
- evaluates exact-50 matched-budget deployment utility;
- computes a descriptive 10%-90% coverage curve without selecting an optimum;
- runs 10,000 full-target matched-random policies at exact50;
- runs 2,000 paired PROMISE12 patient-cluster bootstrap replicates,
  with 10 matched-random policies averaged inside each bootstrap replicate.

NO model fitting.
NO inference/TTA rerun.
NO score flip.
NO target calibration.
NO coverage optimization.
NO HARM-threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P06B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

PROTOCOL = CODE / "B6_P06_MRI_TRANSITION_PATIENT_UTILITY_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "d6e02e54cc4c4e39c6321428b5c9bfacdf33716acfca3424b98ec90adfc23b15"
)

P06A_DIR = ROOT / "B6_P06A_mri_transition_patient_utility_asset_binding_audit_v1"
P06A_DECISION = P06A_DIR / "B6_P06A_BINDING_DECISION.json"
EXPECTED_P06A_GATE = "PASS_B6_P06A_MRI_TRANSITION_PATIENT_UTILITY_ASSET_BINDING"

FINALB1 = ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3"
FINALB3 = ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1"
P01A = ROOT / "B6_P01A_mri_matched_representation_attribution_v1"

CM6_DIR = (
    OUT / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
)

GT_PACKED = CM6_DIR / "PROMISE12_GT_MASKS_PACKBITS.npy"
CM6_OUTCOMES = CM6_DIR / "CM6_PROMISE12_OUTCOMES.csv"

FINALB3_OUTCOMES = FINALB3 / "FINALB3_PROMISE12_ACTION_OUTCOMES.csv"
P01A_SCORES = P01A / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"

OUT_DIR = ROOT / "B6_P06B_mri_transition_patient_utility_execution_v1"

FAMILIES = ["DeepLabV3_R50", "SegFormer_B0"]
ACTIONS = ["TENT1", "PL-CONF90", "MEMO-SEG4-1STEP"]

ACTION_TO_MASK = {
    "TENT1": "tent1_current_masks_packbits.npy",
    "PL-CONF90": "pl_conf90_masks_packbits.npy",
    "MEMO-SEG4-1STEP": "memo_seg4_1step_masks_packbits.npy",
}

N_PATIENTS = 50
N_SLICES = 1377
N_CELLS = 6
N_ROWS = N_SLICES * N_CELLS
SEG_SIZE = 352
PACKED_BYTES = (SEG_SIZE * SEG_SIZE) // 8

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

PRIMARY_COVERAGE = 0.50
COVERAGES = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]

FULL_RANDOM_REPS = 10000
FULL_RANDOM_SEED = 20260919

BOOTSTRAP_REPS = 2000
BOOTSTRAP_INNER_RANDOM_DRAWS = 10
BOOTSTRAP_SEED = 20260920

PASS_GATE = "PASS_B6_P06B_MRI_TRANSITION_PATIENT_UTILITY_COMPLETE"

POPCOUNT = np.asarray([bin(i).count("1") for i in range(256)], dtype=np.uint8)


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


def verify_protocol_and_p06a() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P06 protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} got={got}"
        )

    p = load_json(PROTOCOL)
    if p.get("status") != (
        "FROZEN_POST_REVEAL_ADDITIVE_PROTOCOL_BEFORE_NEW_PATIENT_UTILITY_METRICS"
    ):
        raise RuntimeError("P06 protocol status changed.")

    if not P06A_DECISION.is_file():
        raise FileNotFoundError(P06A_DECISION)
    a = load_json(P06A_DECISION)
    if a.get("gate") != EXPECTED_P06A_GATE:
        raise RuntimeError(f"P06A gate changed: {a.get('gate')}")
    if a.get("route") != "RECOUNT_FROM_FROZEN_MASKS":
        raise RuntimeError(f"P06A route changed: {a.get('route')}")

    return {
        "protocol_sha256": got,
        "p06a_decision_path": str(P06A_DECISION),
        "p06a_decision_sha256": sha256_file(P06A_DECISION),
        "p06a_route": a.get("route"),
    }


def canonical_action(x: str) -> str:
    s = str(x).strip().upper().replace("_", "-")
    if "MEMO" in s:
        return "MEMO-SEG4-1STEP"
    if "PL" in s and ("90" in s or "CONF" in s):
        return "PL-CONF90"
    if "TENT" in s:
        return "TENT1"
    raise RuntimeError(f"Unknown action label: {x!r}")


def load_identity_map() -> pd.DataFrame:
    if not CM6_OUTCOMES.is_file():
        raise FileNotFoundError(CM6_OUTCOMES)
    d = pd.read_csv(CM6_OUTCOMES, low_memory=False)

    required = {"family", "global_index", "case_key", "slice_index"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"CM6 outcomes missing identity columns={missing}")

    d = d[d["family"].astype(str).isin(FAMILIES)].copy()
    d = d[["family","global_index","case_key","slice_index"]].drop_duplicates()

    if len(d) != len(FAMILIES) * N_SLICES:
        raise RuntimeError(f"CM6 identity rows={len(d)} expected={2*N_SLICES}")

    for fam in FAMILIES:
        g = d[d["family"].astype(str) == fam].sort_values("global_index")
        gi = g["global_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(gi, np.arange(N_SLICES, dtype=np.int64)):
            raise RuntimeError(f"{fam}: CM6 global_index ordering drift.")

    if d["case_key"].astype(str).nunique() != N_PATIENTS:
        raise RuntimeError("PROMISE12 patient count drift in CM6 identity.")

    return d.reset_index(drop=True)


def detect_representation_column(d: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    candidates = []
    for c in d.columns:
        if c in {
            "family","model_family","action","global_index","case_key","slice_index",
            "risk_score","source_dice","action_dice","delta_dice","harm","benefit"
        }:
            continue
        if d[c].dtype == object or str(d[c].dtype).startswith("string"):
            vals = sorted(d[c].astype(str).dropna().unique().tolist())
            if 3 <= len(vals) <= 12:
                candidates.append((c, vals))

    def classify(label: str) -> str | None:
        s = label.upper().replace("-", "_").replace(" ", "_")
        if "SIMPLE" in s or "GEOM" in s or "MASK_CHANGE" in s:
            return "SIMPLE_GEOMETRY"
        if "FULL" in s or "SAFETTA" in s or "SAFE_TTA" in s:
            return "FULL_TRANSITION"
        if "SEMANTIC" in s and "TRANSITION" in s:
            return "SEMANTIC_TRANSITION"
        if "SOURCE" in s and ("STATE" in s or "Q66" in s):
            return "SOURCE_STATE"
        return None

    valid = []
    for c, vals in candidates:
        mapping = {}
        for v in vals:
            k = classify(v)
            if k is not None:
                mapping[k] = v
        if {"SOURCE_STATE","SIMPLE_GEOMETRY","FULL_TRANSITION"}.issubset(mapping):
            valid.append((c, mapping, vals))

    if len(valid) != 1:
        detail = {c: vals for c, vals in candidates}
        raise RuntimeError(
            "Unable to uniquely identify P01A representation column. "
            f"Candidates={detail}"
        )
    c, mapping, vals = valid[0]
    return c, mapping


def load_scores(identity: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not P01A_SCORES.is_file():
        raise FileNotFoundError(P01A_SCORES)
    d = pd.read_csv(P01A_SCORES, low_memory=False)

    required = {"family","action","case_key","slice_index","risk_score"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"P01A matched score table missing={missing}")

    d = d[d["family"].astype(str).isin(FAMILIES)].copy()
    d["action"] = d["action"].map(canonical_action)

    rep_col, mapping = detect_representation_column(d)

    # Ensure global_index exists; bind it from the immutable CM6 identity if needed.
    if "global_index" not in d.columns:
        idmap = identity[["family","case_key","slice_index","global_index"]].copy()
        d = d.merge(
            idmap,
            on=["family","case_key","slice_index"],
            how="left",
            validate="many_to_one",
        )
        if d["global_index"].isna().any():
            raise RuntimeError("Failed to bind P01A rows to CM6 global_index.")

    keys = ["family","action","global_index","case_key","slice_index"]
    out = None

    role_to_column = {
        "SOURCE_STATE": "risk_source_state",
        "SIMPLE_GEOMETRY": "risk_simple_geometry",
        "FULL_TRANSITION": "risk_full_transition",
    }

    used_labels = {}
    for role, out_col in role_to_column.items():
        label = mapping[role]
        used_labels[role] = label
        g = d[d[rep_col].astype(str) == str(label)].copy()

        if len(g) != N_ROWS:
            raise RuntimeError(
                f"{role} matched score rows={len(g)} expected={N_ROWS}. "
                f"representation label={label!r}"
            )
        if g.duplicated(["family","action","global_index"]).any():
            raise RuntimeError(f"{role}: duplicate family/action/global_index.")

        g = g[keys + ["risk_score"]].rename(columns={"risk_score": out_col})
        out = g if out is None else out.merge(
            g,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

    assert out is not None
    if len(out) != N_ROWS:
        raise RuntimeError(f"Bound three-score rows={len(out)} expected={N_ROWS}")

    for c in role_to_column.values():
        out[c] = pd.to_numeric(out[c], errors="raise")
        if not np.isfinite(out[c].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite score column={c}")

    return out, {
        "path": str(P01A_SCORES),
        "sha256": sha256_file(P01A_SCORES),
        "representation_column": rep_col,
        "representation_labels": used_labels,
    }


def load_packed(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    if a.shape != (N_SLICES, PACKED_BYTES):
        raise RuntimeError(f"{path}: shape={a.shape}")
    if a.dtype != np.uint8:
        raise RuntimeError(f"{path}: dtype={a.dtype}")
    return a


def packed_counts(pred: np.ndarray, gt: np.ndarray, batch: int = 64):
    tp = np.empty(N_SLICES, dtype=np.int64)
    fp = np.empty(N_SLICES, dtype=np.int64)
    fn = np.empty(N_SLICES, dtype=np.int64)

    for start in range(0, N_SLICES, batch):
        stop = min(start + batch, N_SLICES)
        p = np.asarray(pred[start:stop], dtype=np.uint8)
        g = np.asarray(gt[start:stop], dtype=np.uint8)

        tp[start:stop] = POPCOUNT[np.bitwise_and(p, g)].sum(axis=1, dtype=np.int64)
        fp[start:stop] = POPCOUNT[
            np.bitwise_and(p, np.bitwise_not(g))
        ].sum(axis=1, dtype=np.int64)
        fn[start:stop] = POPCOUNT[
            np.bitwise_and(np.bitwise_not(p), g)
        ].sum(axis=1, dtype=np.int64)

    return tp, fp, fn


def dice_from_counts(tp, fp, fn) -> np.ndarray:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    den = 2.0 * tp + fp + fn
    out = np.ones_like(den, dtype=np.float64)
    m = den > 0
    out[m] = 2.0 * tp[m] / den[m]
    return out


def recount_all(identity: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not GT_PACKED.is_file():
        raise FileNotFoundError(GT_PACKED)
    gt = load_packed(GT_PACKED)

    rows = []
    asset_rows = []

    for family in FAMILIES:
        base = FINALB1 / "target_promise12" / family
        src_path = base / "source_current_masks_packbits.npy"
        src = load_packed(src_path)
        stp, sfp, sfn = packed_counts(src, gt)
        sdice = dice_from_counts(stp, sfp, sfn)

        fam_id = (
            identity[identity["family"].astype(str) == family]
            .sort_values("global_index")
            .reset_index(drop=True)
        )
        if len(fam_id) != N_SLICES:
            raise RuntimeError(f"{family}: identity count drift.")

        asset_rows.append({
            "family": family,
            "action": "SOURCE",
            "path": str(src_path),
            "sha256": sha256_file(src_path),
        })

        for action in ACTIONS:
            cand_path = base / ACTION_TO_MASK[action]
            cand = load_packed(cand_path)
            ctp, cfp, cfn = packed_counts(cand, gt)
            cdice = dice_from_counts(ctp, cfp, cfn)
            delta = cdice - sdice
            harm = (delta <= HARM_THRESHOLD).astype(np.int8)
            benefit = (delta >= BENEFIT_THRESHOLD).astype(np.int8)

            z = fam_id.copy()
            z["action"] = action
            z["source_tp"] = stp
            z["source_fp"] = sfp
            z["source_fn"] = sfn
            z["candidate_tp"] = ctp
            z["candidate_fp"] = cfp
            z["candidate_fn"] = cfn
            z["source_dice"] = sdice
            z["action_dice"] = cdice
            z["delta_dice"] = delta
            z["harm"] = harm
            z["benefit"] = benefit
            rows.append(z)

            asset_rows.append({
                "family": family,
                "action": action,
                "path": str(cand_path),
                "sha256": sha256_file(cand_path),
            })

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["family","action","global_index"], kind="mergesort"
    ).reset_index(drop=True)

    if len(out) != N_ROWS:
        raise RuntimeError(f"Recount rows={len(out)} expected={N_ROWS}")

    meta = {
        "gt_path": str(GT_PACKED),
        "gt_sha256": sha256_file(GT_PACKED),
        "mask_assets": asset_rows,
    }
    return out, meta


def verify_finalb3_parity(recount: pd.DataFrame) -> Dict[str, Any]:
    if not FINALB3_OUTCOMES.is_file():
        raise FileNotFoundError(FINALB3_OUTCOMES)
    d = pd.read_csv(FINALB3_OUTCOMES, low_memory=False)
    d = d[d["family"].astype(str).isin(FAMILIES)].copy()
    d["action"] = d["action"].map(canonical_action)

    keys = ["family","action","case_key","slice_index"]
    for c in ["source_dice","action_dice","delta_dice"]:
        if c not in d.columns:
            raise RuntimeError(f"FinalB3 outcomes missing {c}")

    x = recount.merge(
        d[keys + ["source_dice","action_dice","delta_dice","harm","benefit"]].rename(
            columns={
                "source_dice": "ref_source_dice",
                "action_dice": "ref_action_dice",
                "delta_dice": "ref_delta_dice",
                "harm": "ref_harm",
                "benefit": "ref_benefit",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    if len(x) != N_ROWS:
        raise RuntimeError("FinalB3 parity join row count drift.")

    diffs = {}
    for a,b in [
        ("source_dice","ref_source_dice"),
        ("action_dice","ref_action_dice"),
        ("delta_dice","ref_delta_dice"),
    ]:
        mx = float(np.max(np.abs(x[a].to_numpy(float) - x[b].to_numpy(float))))
        diffs[a] = mx
        if mx > 1e-12:
            raise RuntimeError(f"FinalB3 numeric parity failed {a}: max_abs={mx}")

    if not np.array_equal(
        x["harm"].to_numpy(int),
        x["ref_harm"].to_numpy(int),
    ):
        raise RuntimeError("FinalB3 HARM parity failed.")
    if not np.array_equal(
        x["benefit"].to_numpy(int),
        x["ref_benefit"].to_numpy(int),
    ):
        raise RuntimeError("FinalB3 BENEFIT parity failed.")

    return {
        "path": str(FINALB3_OUTCOMES),
        "sha256": sha256_file(FINALB3_OUTCOMES),
        "max_abs_diffs": diffs,
        "harm_exact": True,
        "benefit_exact": True,
    }


def build_bound_rows() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    identity = load_identity_map()
    scores, score_meta = load_scores(identity)
    recount, count_meta = recount_all(identity)
    parity = verify_finalb3_parity(recount)

    keys = ["family","action","global_index","case_key","slice_index"]
    d = recount.merge(scores, on=keys, validate="one_to_one")
    if len(d) != N_ROWS:
        raise RuntimeError("Recount/score bound row count drift.")

    # Fixed deterministic patient and cell indices.
    patients = sorted(d["case_key"].astype(str).unique().tolist())
    if len(patients) != N_PATIENTS:
        raise RuntimeError("PROMISE patient cardinality changed.")
    patient_to_i = {p:i for i,p in enumerate(patients)}

    cell_keys = [(f,a) for f in FAMILIES for a in ACTIONS]
    cell_to_i = {k:i for i,k in enumerate(cell_keys)}

    d["patient_i"] = d["case_key"].astype(str).map(patient_to_i).astype(int)
    d["cell_i"] = [
        cell_to_i[(str(f), str(a))]
        for f,a in zip(d["family"], d["action"])
    ]
    d["patient_cell_i"] = d["cell_i"] * N_PATIENTS + d["patient_i"]

    return d, {
        "scores": score_meta,
        "counts": count_meta,
        "finalb3_parity": parity,
        "patients": patients,
        "cell_keys": [{"family":f,"action":a,"cell_i":i} for (f,a),i in cell_to_i.items()],
    }


def exact_low_risk_selection(d: pd.DataFrame, score_col: str, coverage: float) -> np.ndarray:
    sel = np.zeros(len(d), dtype=bool)
    for cell in range(N_CELLS):
        idx = np.flatnonzero(d["cell_i"].to_numpy(int) == cell)
        k = int(math.floor(len(idx) * float(coverage)))
        g = d.iloc[idx][["case_key","slice_index",score_col]].copy()
        g["_row"] = idx
        g = g.sort_values(
            [score_col,"case_key","slice_index"],
            ascending=[True,True,True],
            kind="mergesort",
        )
        chosen = g["_row"].to_numpy(dtype=np.int64)[:k]
        sel[chosen] = True
    return sel


def policy_metrics(
    d: pd.DataFrame,
    adapt: np.ndarray,
    row_indices: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    n_groups: int | None = None,
) -> Dict[str, float]:
    if row_indices is None:
        idx = np.arange(len(d), dtype=np.int64)
        if group_ids is None:
            group_ids = d["patient_cell_i"].to_numpy(dtype=np.int64)
            n_groups = N_PATIENTS * N_CELLS
    else:
        idx = np.asarray(row_indices, dtype=np.int64)
        if group_ids is None or n_groups is None:
            raise RuntimeError("Bootstrap metrics require explicit group_ids/n_groups.")

    a = np.asarray(adapt, dtype=bool)[idx]

    stp = d["source_tp"].to_numpy(dtype=np.float64)[idx]
    sfp = d["source_fp"].to_numpy(dtype=np.float64)[idx]
    sfn = d["source_fn"].to_numpy(dtype=np.float64)[idx]
    ctp = d["candidate_tp"].to_numpy(dtype=np.float64)[idx]
    cfp = d["candidate_fp"].to_numpy(dtype=np.float64)[idx]
    cfn = d["candidate_fn"].to_numpy(dtype=np.float64)[idx]

    tp = np.where(a, ctp, stp)
    fp = np.where(a, cfp, sfp)
    fn = np.where(a, cfn, sfn)

    tp_g = np.bincount(group_ids, weights=tp, minlength=n_groups)
    fp_g = np.bincount(group_ids, weights=fp, minlength=n_groups)
    fn_g = np.bincount(group_ids, weights=fn, minlength=n_groups)
    den = 2*tp_g + fp_g + fn_g
    dice_g = np.ones_like(den, dtype=np.float64)
    m = den > 0
    dice_g[m] = 2*tp_g[m] / den[m]

    # Every bootstrap patient occurrence has all 6 cells, so all groups are valid.
    patient3d = float(dice_g.mean())

    sd = d["source_dice"].to_numpy(dtype=np.float64)[idx]
    cd = d["action_dice"].to_numpy(dtype=np.float64)[idx]
    slice_deployed = np.where(a, cd, sd)

    harm = d["harm"].to_numpy(dtype=np.int8)[idx].astype(bool)
    benefit = d["benefit"].to_numpy(dtype=np.int8)[idx].astype(bool)

    harm_n = int(harm.sum())
    benefit_n = int(benefit.sum())
    adapted_n = int(a.sum())

    return {
        "coverage": float(a.mean()),
        "adapted_rows": adapted_n,
        "mean_patient_3d_deployed_dice": patient3d,
        "mean_slice_deployed_dice": float(slice_deployed.mean()),
        "prevented_harm_fraction": (
            float(np.sum(harm & (~a)) / harm_n) if harm_n > 0 else np.nan
        ),
        "committed_harm_rate": (
            float(np.sum(harm & a) / adapted_n) if adapted_n > 0 else np.nan
        ),
        "benefit_capture_fraction": (
            float(np.sum(benefit & a) / benefit_n) if benefit_n > 0 else np.nan
        ),
        "harm_n": harm_n,
        "benefit_n": benefit_n,
    }


def deterministic_point_table(d: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,np.ndarray]]:
    policies: Dict[str,np.ndarray] = {
        "SOURCE_ONLY": np.zeros(len(d), dtype=bool),
        "ADAPT_ALL": np.ones(len(d), dtype=bool),
        "SOURCE_STATE_GATING_EXACT50": exact_low_risk_selection(
            d, "risk_source_state", PRIMARY_COVERAGE
        ),
        "SIMPLE_GEOMETRY_GATING_EXACT50": exact_low_risk_selection(
            d, "risk_simple_geometry", PRIMARY_COVERAGE
        ),
        "FULL_TRANSITION_GATING_EXACT50": exact_low_risk_selection(
            d, "risk_full_transition", PRIMARY_COVERAGE
        ),
    }

    rows = []
    for name, sel in policies.items():
        rows.append({"policy": name, **policy_metrics(d, sel)})
    return pd.DataFrame(rows), policies


def coverage_curve(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("SOURCE_STATE", "risk_source_state"),
        ("SIMPLE_GEOMETRY", "risk_simple_geometry"),
        ("FULL_TRANSITION", "risk_full_transition"),
    ]
    for cov in COVERAGES:
        for label, col in specs:
            sel = exact_low_risk_selection(d, col, cov)
            rows.append({
                "method": label,
                "nominal_coverage": cov,
                **policy_metrics(d, sel),
            })
    return pd.DataFrame(rows)


def random_exact_cell_selection(
    d: pd.DataFrame,
    k_by_cell: Dict[int,int],
    rng: np.random.Generator,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    if row_indices is None:
        universe = np.arange(len(d), dtype=np.int64)
    else:
        universe = np.asarray(row_indices, dtype=np.int64)

    sel = np.zeros(len(d), dtype=bool)
    cell_all = d["cell_i"].to_numpy(dtype=np.int64)

    for cell in range(N_CELLS):
        idx = universe[cell_all[universe] == cell]
        k = int(k_by_cell[cell])
        if k < 0 or k > len(idx):
            raise RuntimeError(f"Random k invalid cell={cell} k={k} n={len(idx)}")
        if k > 0:
            chosen = rng.choice(idx, size=k, replace=False)
            sel[chosen] = True
    return sel


def full_target_randomization(
    d: pd.DataFrame,
    full_sel: np.ndarray,
    full_metrics: Dict[str,float],
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    rng = np.random.default_rng(FULL_RANDOM_SEED)
    cell = d["cell_i"].to_numpy(dtype=np.int64)
    k_by_cell = {
        c: int(np.sum(full_sel & (cell == c)))
        for c in range(N_CELLS)
    }

    rows = []
    for rep in tqdm(
        range(FULL_RANDOM_REPS),
        desc="P06B exact50 matched-random",
        unit="rep",
        dynamic_ncols=True,
    ):
        sel = random_exact_cell_selection(d, k_by_cell, rng)
        m = policy_metrics(d, sel)
        rows.append({"rep": rep, **m})

    rnd = pd.DataFrame(rows)
    summary_rows = []
    for metric in [
        "mean_patient_3d_deployed_dice",
        "mean_slice_deployed_dice",
        "prevented_harm_fraction",
        "committed_harm_rate",
        "benefit_capture_fraction",
    ]:
        v = rnd[metric].to_numpy(dtype=np.float64)
        safe = float(full_metrics[metric])
        delta = safe - v
        # For committed-HARM, lower is better; still store raw SafeTTA-random delta.
        summary_rows.append({
            "metric": metric,
            "full_transition": safe,
            "random_mean": float(v.mean()),
            "random_ci95_low": float(np.quantile(v, 0.025)),
            "random_ci95_high": float(np.quantile(v, 0.975)),
            "full_minus_random_mean": float(delta.mean()),
            "delta_ci95_low": float(np.quantile(delta, 0.025)),
            "delta_ci95_high": float(np.quantile(delta, 0.975)),
            "p_random_ge_full": float((1 + np.sum(v >= safe)) / (1 + len(v))),
            "p_random_le_full": float((1 + np.sum(v <= safe)) / (1 + len(v))),
        })
    return rnd, pd.DataFrame(summary_rows)


def bootstrap_dataset_indices(d: pd.DataFrame, sampled_patients: np.ndarray):
    patient = d["patient_i"].to_numpy(dtype=np.int64)
    cell = d["cell_i"].to_numpy(dtype=np.int64)

    parts = []
    groups = []
    for occurrence, p in enumerate(sampled_patients):
        idx = np.flatnonzero(patient == int(p))
        parts.append(idx)
        groups.append(cell[idx] * N_PATIENTS + occurrence)

    return np.concatenate(parts), np.concatenate(groups), N_PATIENTS * N_CELLS


def paired_patient_bootstrap(
    d: pd.DataFrame,
    policies: Dict[str,np.ndarray],
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    pair_specs = [
        ("FULL_TRANSITION_GATING_EXACT50", "SOURCE_STATE_GATING_EXACT50"),
        ("FULL_TRANSITION_GATING_EXACT50", "SIMPLE_GEOMETRY_GATING_EXACT50"),
        ("FULL_TRANSITION_GATING_EXACT50", "SOURCE_ONLY"),
        ("FULL_TRANSITION_GATING_EXACT50", "ADAPT_ALL"),
    ]
    metrics = [
        "mean_patient_3d_deployed_dice",
        "mean_slice_deployed_dice",
        "prevented_harm_fraction",
        "committed_harm_rate",
        "benefit_capture_fraction",
    ]

    rep_rows = []

    full_sel = policies["FULL_TRANSITION_GATING_EXACT50"]
    cell_all = d["cell_i"].to_numpy(dtype=np.int64)

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P06B patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.integers(0, N_PATIENTS, size=N_PATIENTS)
        idx, groups, ngroups = bootstrap_dataset_indices(d, sampled)

        vals = {}
        for name, sel in policies.items():
            vals[name] = policy_metrics(
                d, sel,
                row_indices=idx,
                group_ids=groups,
                n_groups=ngroups,
            )

        # Matched random to FULL selection in this bootstrap replicate.
        k_by_cell = {
            c: int(np.sum(full_sel[idx] & (cell_all[idx] == c)))
            for c in range(N_CELLS)
        }
        random_draw_metrics = []
        for _ in range(BOOTSTRAP_INNER_RANDOM_DRAWS):
            rsel = random_exact_cell_selection(
                d, k_by_cell, rng, row_indices=idx
            )
            random_draw_metrics.append(
                policy_metrics(
                    d, rsel,
                    row_indices=idx,
                    group_ids=groups,
                    n_groups=ngroups,
                )
            )

        vals["MATCHED_RANDOM_EXACT50"] = {
            metric: float(np.mean([x[metric] for x in random_draw_metrics]))
            for metric in metrics
        }

        for a,b in pair_specs + [
            ("FULL_TRANSITION_GATING_EXACT50", "MATCHED_RANDOM_EXACT50")
        ]:
            for metric in metrics:
                va = vals[a][metric]
                vb = vals[b][metric]
                if np.isfinite(va) and np.isfinite(vb):
                    rep_rows.append({
                        "rep": rep,
                        "comparison": f"{a} - {b}",
                        "metric": metric,
                        "delta": float(va - vb),
                    })

    reps = pd.DataFrame(rep_rows)
    summary = []
    for (comp, metric), g in reps.groupby(["comparison","metric"], sort=False):
        v = g["delta"].to_numpy(dtype=np.float64)
        summary.append({
            "comparison": comp,
            "metric": metric,
            "valid_reps": int(len(v)),
            "mean_delta": float(v.mean()),
            "ci95_low": float(np.quantile(v,0.025)),
            "ci95_high": float(np.quantile(v,0.975)),
            "ci_excludes_zero": bool(
                np.quantile(v,0.025) > 0 or np.quantile(v,0.975) < 0
            ),
        })
    return reps, pd.DataFrame(summary)


def self_test():
    # Packed-count sanity on one byte.
    pred = np.asarray([[0b00001111]], dtype=np.uint8)
    gt = np.asarray([[0b00111100]], dtype=np.uint8)
    tp = int(POPCOUNT[np.bitwise_and(pred,gt)].sum())
    fp = int(POPCOUNT[np.bitwise_and(pred,np.bitwise_not(gt))].sum())
    fn = int(POPCOUNT[np.bitwise_and(np.bitwise_not(pred),gt)].sum())
    assert (tp,fp,fn) == (2,2,2)
    assert PACKED_BYTES == 15488
    assert N_ROWS == 8262
    assert int(math.floor(N_SLICES * 0.5)) == 688
    print("SELF_TEST_PACKED_COUNTS=PASS")
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P06B MRI patient-level transition utility."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 172)
    print("SafeTTA B6-P0-6B — MRI transition patient-level utility")
    print("Version                    :", VERSION)
    print("Model fitting              : NO")
    print("Inference/TTA rerun        : NO")
    print("Score flip/calibration     : NO")
    print("Primary coverage           : exact50 within each family x action cell")
    print("Full-target random reps    :", FULL_RANDOM_REPS)
    print("Patient bootstrap reps     :", BOOTSTRAP_REPS)
    print("Bootstrap random draws/rep :", BOOTSTRAP_INNER_RANDOM_DRAWS)
    print("=" * 172)

    print("\n[1/6] Verify P06 protocol and P06A binding gate")
    lineage = verify_protocol_and_p06a()
    print("PROTOCOL_SHA256 =", lineage["protocol_sha256"])
    print("P06A_ROUTE      =", lineage["p06a_route"])
    print("LINEAGE=PASS")

    print("\n[2/6] Bind P01A long-format representations")
    identity = load_identity_map()
    scores, score_meta = load_scores(identity)
    print("representation_column =", score_meta["representation_column"])
    print("representation_labels =", score_meta["representation_labels"])
    print("score rows             =", len(scores))
    print("SCORE_BINDING=PASS")

    print("\n[3/6] Verify packed assets")
    if not GT_PACKED.is_file():
        raise FileNotFoundError(GT_PACKED)
    gt_shape = np.load(GT_PACKED, mmap_mode="r", allow_pickle=False).shape
    print("GT packed shape =", gt_shape)
    for family in FAMILIES:
        base = FINALB1 / "target_promise12" / family
        for name in ["source_current_masks_packbits.npy", *ACTION_TO_MASK.values()]:
            p = base / name
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            print(f"{family:16s} {name:38s} shape={a.shape} dtype={a.dtype}")
            if a.shape != (N_SLICES, PACKED_BYTES) or a.dtype != np.uint8:
                raise RuntimeError(f"Packed mask asset drift: {p}")
    print("PACKED_ASSETS=PASS")

    if args.preflight_only:
        print("\nP06B_PREFLIGHT=PASS")
        print("GT COUNT REPLAY=NOT_RUN")
        print("UTILITY METRICS=NOT_RUN")
        print("RANDOMIZATION=NOT_RUN")
        print("BOOTSTRAP=NOT_RUN")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P06B output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/6] Recount frozen masks against frozen CM6 GT + verify FinalB3 parity")
    d, bind_meta = build_bound_rows()
    print("bound rows =", len(d))
    print("patients   =", d["case_key"].nunique())
    print("cells      =", d[["family","action"]].drop_duplicates().shape[0])
    print("FinalB3 parity =", bind_meta["finalb3_parity"])
    print("RECOUNT_AND_PARITY=PASS")

    p_bound = OUT_DIR / "B6_P06B_BOUND_PATIENT_UTILITY_ROWS.csv"
    d.to_csv(p_bound, index=False)

    print("\n[5/6] Exact50 + coverage curve + full-target matched random")
    point, policies = deterministic_point_table(d)
    curve = coverage_curve(d)

    full_metrics = point[
        point["policy"] == "FULL_TRANSITION_GATING_EXACT50"
    ].iloc[0].to_dict()

    rnd, rnd_summary = full_target_randomization(
        d,
        policies["FULL_TRANSITION_GATING_EXACT50"],
        full_metrics,
    )

    print("\nEXACT50 POINT TABLE")
    print(point.to_string(index=False))
    print("\nMATCHED RANDOM SUMMARY")
    print(rnd_summary.to_string(index=False))

    print("\n[6/6] Paired PROMISE12 patient bootstrap")
    boot_reps, boot_summary = paired_patient_bootstrap(d, policies)
    print("\nBOOTSTRAP DELTAS")
    print(boot_summary.to_string(index=False))

    p_point = OUT_DIR / "B6_P06B_EXACT50_POINT_TABLE.csv"
    p_curve = OUT_DIR / "B6_P06B_COVERAGE_CURVE.csv"
    p_rnd = OUT_DIR / "B6_P06B_EXACT50_RANDOM_REPLICATES.csv"
    p_rnd_sum = OUT_DIR / "B6_P06B_EXACT50_RANDOM_SUMMARY.csv"
    p_boot = OUT_DIR / "B6_P06B_PATIENT_BOOTSTRAP_REPLICATES.csv"
    p_boot_sum = OUT_DIR / "B6_P06B_PATIENT_BOOTSTRAP_SUMMARY.csv"

    point.to_csv(p_point, index=False)
    curve.to_csv(p_curve, index=False)
    rnd.to_csv(p_rnd, index=False)
    rnd_summary.to_csv(p_rnd_sum, index=False)
    boot_reps.to_csv(p_boot, index=False)
    boot_summary.to_csv(p_boot_sum, index=False)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "lineage": lineage,
        "binding": bind_meta,
        "protocol": {
            "primary_coverage": PRIMARY_COVERAGE,
            "coverage_curve": COVERAGES,
            "harm_threshold": HARM_THRESHOLD,
            "benefit_threshold": BENEFIT_THRESHOLD,
            "score_direction": "higher=HARM risk; adapt lowest risk",
            "coverage_optimization": False,
            "target_calibration": False,
            "score_flip": False,
        },
        "headline": {
            "exact50": point.to_dict(orient="records"),
            "full_vs_random": rnd_summary.to_dict(orient="records"),
            "patient_bootstrap": boot_summary.to_dict(orient="records"),
        },
        "artifacts": {},
    }

    for p in [
        p_bound,p_point,p_curve,p_rnd,p_rnd_sum,p_boot,p_boot_sum
    ]:
        audit["artifacts"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    p_audit = OUT_DIR / "B6_P06B_MRI_TRANSITION_PATIENT_UTILITY_AUDIT.json"
    p_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Headline report.
    primary_metric = "mean_patient_3d_deployed_dice"
    bfull_random = boot_summary[
        (boot_summary["comparison"] ==
         "FULL_TRANSITION_GATING_EXACT50 - MATCHED_RANDOM_EXACT50")
        & (boot_summary["metric"] == primary_metric)
    ]
    bfull_source = boot_summary[
        (boot_summary["comparison"] ==
         "FULL_TRANSITION_GATING_EXACT50 - SOURCE_STATE_GATING_EXACT50")
        & (boot_summary["metric"] == primary_metric)
    ]
    bfull_geom = boot_summary[
        (boot_summary["comparison"] ==
         "FULL_TRANSITION_GATING_EXACT50 - SIMPLE_GEOMETRY_GATING_EXACT50")
        & (boot_summary["metric"] == primary_metric)
    ]

    report = "\n".join([
        "=" * 172,
        "SafeTTA B6-P0-6B MRI TRANSITION PATIENT UTILITY COMPLETE",
        "",
        "EXACT50 POINT TABLE:",
        point.to_string(index=False),
        "",
        "PRIMARY PATIENT-3D BOOTSTRAP:",
        pd.concat([bfull_random,bfull_source,bfull_geom]).to_string(index=False),
        "",
        "No coverage optimization: YES",
        "No target calibration/refit/score flip: YES",
        "FinalB3 mask/outcome parity: PASS",
        "",
        "GATE=" + PASS_GATE,
        "audit_json=" + str(p_audit),
        "stage_dir=" + str(OUT_DIR),
        "=" * 172,
        "",
    ])
    p_report = OUT_DIR / "B6_P06B_MRI_TRANSITION_PATIENT_UTILITY_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
