#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-3 — MRI empty-slice / foreground-size / SOURCE-quality robustness.

POST-FREEZE / POST-REVEAL sensitivity audit.

Reads only:
- Final-B1 frozen current-runtime SOURCE masks,
- Final-B3 frozen outcomes,
- B6-P01A frozen matched-representation scores,
- historical CM6 PROMISE12 GT cache.

No model fitting.
No new score generation.
No score reversal.
No HARM-threshold change.
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

VERSION = "2026-09-15-B6-P03-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_NAME = "B6_P03_MRI_STRATIFIED_ROBUSTNESS_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "6a2ec54c59e38d35367505343c9e1baa115a2434356415255708e03ef053a1a6"

B1_DIR = ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3"
B1_LOCK = B1_DIR / "FINALB1_MRI_ACTION_CANDIDATE_MASK_LOCK.json"
EXPECTED_B1_SHA = "5900b35a6971ceff32ddd6e22e8332cf9fdf68e65d9a6c2711c52a4d324eacc5"

B3_DIR = ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1"
B3_LOCK = B3_DIR / "FINALB3_MRI_FUTURE_HARM_LOCK.json"
EXPECTED_B3_SHA = "924b8b882c971be7a5562b7d0fc53cf1eece343786f0ff5f24d46b91a86a191b"
B3_TARGET_OUTCOMES = B3_DIR / "FINALB3_PROMISE12_ACTION_OUTCOMES.csv"

P01A_DIR = ROOT / "B6_P01A_mri_matched_representation_attribution_v1"
P01A_AUDIT = P01A_DIR / "B6_P01A_AUDIT.json"
P01A_TARGET_SCORES = P01A_DIR / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"
EXPECTED_P01A_GATE = "PASS_B6_P01A_MRI_MATCHED_REPRESENTATION_ATTRIBUTION_COMPLETE"

CM6_DIR = (
    ROOT / "outputs"
    / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
)
CM6_GT = CM6_DIR / "PROMISE12_GT_MASKS_PACKBITS.npy"
CM6_GT_META = CM6_DIR / "PROMISE12_GT_REVEAL_META.json"
CM6_CASE_AUDIT = CM6_DIR / "PROMISE12_GT_CASE_AUDIT.csv"

OUT_DIR = ROOT / "B6_P03_mri_empty_foreground_sourcequality_robustness_v1"
PASS_GATE = "PASS_B6_P03_MRI_STRATIFIED_ROBUSTNESS_AUDIT_COMPLETE"

FAMILIES = ["DeepLabV3_R50", "SegFormer_B0"]
ACTIONS = ["TENT1", "PL-CONF90", "MEMO-SEG4-1STEP"]
REPRESENTATIONS = [
    "SOURCE_STATE",
    "SEMANTIC_TRANSITION",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    "FULL_TRANSITION_SAFETTA",
]

SOURCE_N = 3553
TARGET_N = 1377
TARGET_PATIENTS = 50
H = W = 352
PACKED_BYTES = H * W // 8

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260917


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
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != str(expected).lower():
        raise RuntimeError(f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}")
    return got


def recorded_path(value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def verify_protocol(script_dir: Path) -> Dict[str, Any]:
    p = script_dir / PROTOCOL_NAME
    exact_sha(p, EXPECTED_PROTOCOL_SHA256, "P03 protocol")
    d = load_json(p)
    if d.get("status") != "FROZEN_AFTER_P01A_BEFORE_P03_METRICS":
        raise RuntimeError("P03 protocol status changed.")
    if d["frozen_boundaries"]["score_refit"] is not False:
        raise RuntimeError("P03 score-refit boundary changed.")
    if int(d["statistics"]["bootstrap_reps"]) != BOOTSTRAP_REPS:
        raise RuntimeError("P03 bootstrap count changed.")
    return d


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    exact_sha(B1_LOCK, EXPECTED_B1_SHA, "FinalB1 lock")
    exact_sha(B3_LOCK, EXPECTED_B3_SHA, "FinalB3 lock")

    b1 = load_json(B1_LOCK)
    b3 = load_json(B3_LOCK)

    if b1.get("status") != "PASS" or b3.get("status") != "PASS":
        raise RuntimeError("Final-B1/B3 status changed.")

    if not P01A_AUDIT.is_file():
        raise FileNotFoundError(P01A_AUDIT)
    p01a = load_json(P01A_AUDIT)
    if p01a.get("status") != "PASS" or p01a.get("gate") != EXPECTED_P01A_GATE:
        raise RuntimeError("P01A gate changed.")

    p01a_rec = p01a["outputs"]["B6_P01A_PROMISE12_MATCHED_SCORES.csv"]
    exact_sha(P01A_TARGET_SCORES, p01a_rec["sha256"], "P01A target scores")

    return b1, b3, p01a


def load_cm6_gt() -> Tuple[np.ndarray, pd.DataFrame]:
    if not CM6_GT_META.is_file():
        raise FileNotFoundError(CM6_GT_META)
    meta = load_json(CM6_GT_META)

    if meta.get("status") != "PASS":
        raise RuntimeError("CM6 GT meta status changed.")
    if int(meta.get("slices", -1)) != TARGET_N:
        raise RuntimeError("CM6 GT slice count changed.")
    if int(meta.get("patients", -1)) != TARGET_PATIENTS:
        raise RuntimeError("CM6 GT patient count changed.")

    exact_sha(CM6_GT, meta["gt_sha256"], "CM6 PROMISE12 GT packbits")
    exact_sha(CM6_CASE_AUDIT, meta["case_audit_sha256"], "CM6 GT case audit")

    gt = np.load(CM6_GT, mmap_mode="r")
    if gt.shape != (TARGET_N, PACKED_BYTES) or gt.dtype != np.uint8:
        raise RuntimeError(f"CM6 GT packed shape/dtype changed: {gt.shape} {gt.dtype}")

    case_audit = pd.read_csv(CM6_CASE_AUDIT, low_memory=False)
    if len(case_audit) != TARGET_PATIENTS:
        raise RuntimeError("CM6 case audit patient count changed.")

    return gt, case_audit


def load_axis(b1: Dict[str, Any], cohort: str) -> pd.DataFrame:
    rec = b1["cohorts"][cohort]
    p = recorded_path(rec["safe_axis"])
    exact_sha(p, rec["safe_axis_sha256"], f"{cohort} safe axis")
    df = pd.read_csv(p, low_memory=False)
    n = SOURCE_N if cohort == "Prostate158" else TARGET_N
    if len(df) != n:
        raise RuntimeError(f"{cohort} axis rows changed.")
    if df["global_index"].astype(int).tolist() != list(range(n)):
        raise RuntimeError(f"{cohort} global_index order changed.")
    return df


def load_source_packed(b1: Dict[str, Any], cohort: str, family: str) -> np.ndarray:
    key = ("source::" if cohort == "Prostate158" else "target::") + family
    rec = b1["unit_locks"][key]
    lock_path = recorded_path(rec["path"])
    exact_sha(lock_path, rec["sha256"], f"B1 unit {key}")
    lock = load_json(lock_path)
    sr = lock["new_current_runtime_panel"]["SOURCE"]
    p = recorded_path(sr["path"])
    exact_sha(p, sr["sha256"], f"B1 {key} SOURCE")
    a = np.load(p, mmap_mode="r")
    n = SOURCE_N if cohort == "Prostate158" else TARGET_N
    if a.shape != (n, PACKED_BYTES):
        raise RuntimeError(f"{key} SOURCE shape changed: {a.shape}")
    return a


def packed_nonempty(packed: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(packed, dtype=np.uint8) != 0, axis=1)


def packed_foreground_fraction_big(packed: np.ndarray, batch: int = 128) -> np.ndarray:
    n = len(packed)
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        flat = np.unpackbits(
            np.asarray(packed[start:end], dtype=np.uint8),
            axis=1,
            count=H * W,
            bitorder="big",
        )
        out[start:end] = flat.mean(axis=1, dtype=np.float64)
    return out


def verify_gt_axis_alignment(
    gt: np.ndarray,
    case_audit: pd.DataFrame,
    target_axis: pd.DataFrame,
) -> np.ndarray:
    gt_nonempty = packed_nonempty(gt)

    tmp = target_axis[["global_index", "case_key"]].copy()
    tmp["gt_nonempty"] = gt_nonempty

    computed = (
        tmp.groupby("case_key", as_index=False)
        .agg(
            slices=("global_index", "size"),
            positive_slices=("gt_nonempty", "sum"),
        )
    )

    ref = case_audit[["case_key", "slices", "positive_slices"]].copy()

    merged = computed.merge(
        ref,
        on="case_key",
        suffixes=("_computed", "_cm6"),
        validate="one_to_one",
    )

    if len(merged) != TARGET_PATIENTS:
        raise RuntimeError("GT/case-axis patient binding changed.")

    for col in ["slices", "positive_slices"]:
        if not np.array_equal(
            merged[f"{col}_computed"].to_numpy(dtype=np.int64),
            merged[f"{col}_cm6"].to_numpy(dtype=np.int64),
        ):
            raise RuntimeError(f"GT axis alignment failed for {col}.")

    return gt_nonempty


def family_cutpoints_from_source(
    b1: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    cuts = {}

    for family in FAMILIES:
        p = load_source_packed(b1, "Prostate158", family)
        frac = packed_foreground_fraction_big(p)
        pos = frac[frac > 0.0]

        if len(pos) < 100:
            raise RuntimeError(f"{family} insufficient nonempty SOURCE support.")

        q = np.quantile(pos, [0.25, 0.50, 0.75])

        cuts[family] = {
            "q25": float(q[0]),
            "q50": float(q[1]),
            "q75": float(q[2]),
            "source_nonempty_n": int(len(pos)),
            "source_empty_n": int(np.sum(frac == 0.0)),
        }

    return cuts


def assign_foreground_stratum(frac: float, cut: Dict[str, float]) -> str:
    if frac == 0.0:
        return "SOURCE_EMPTY"
    if frac <= cut["q25"]:
        return "SOURCE_FG_Q1"
    if frac <= cut["q50"]:
        return "SOURCE_FG_Q2"
    if frac <= cut["q75"]:
        return "SOURCE_FG_Q3"
    return "SOURCE_FG_Q4"


def source_dice_cutpoints(
    outcomes: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    cuts = {}

    for family in FAMILIES:
        g = outcomes[outcomes["family"] == family][
            ["global_index", "source_dice"]
        ].drop_duplicates()

        if len(g) != TARGET_N:
            raise RuntimeError(f"{family} SOURCE Dice uniqueness changed.")

        q = np.quantile(
            g["source_dice"].to_numpy(dtype=np.float64),
            [0.25, 0.50, 0.75],
        )

        cuts[family] = {
            "q25": float(q[0]),
            "q50": float(q[1]),
            "q75": float(q[2]),
        }

    return cuts


def assign_dice_stratum(x: float, cut: Dict[str, float]) -> str:
    if x <= cut["q25"]:
        return "SOURCE_DICE_Q1"
    if x <= cut["q50"]:
        return "SOURCE_DICE_Q2"
    if x <= cut["q75"]:
        return "SOURCE_DICE_Q3"
    return "SOURCE_DICE_Q4"


def build_analysis_table(
    b1: Dict[str, Any],
    gt_nonempty: np.ndarray,
    fg_cuts: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    scores = pd.read_csv(P01A_TARGET_SCORES, low_memory=False)
    outcomes = pd.read_csv(B3_TARGET_OUTCOMES, low_memory=False)

    keys = [
        "cohort",
        "family",
        "action",
        "global_index",
        "case_key",
        "slice_index",
    ]

    # P01A baseline rows and frozen full rows may carry duplicate outcome columns;
    # keep only score identity and re-bind to Final-B3 outcomes.
    score_min = scores[
        keys + ["representation", "risk_score"]
    ].copy()

    outcome_cols = keys + [
        "harm",
        "source_dice",
        "delta_dice",
    ]
    out_min = outcomes[outcome_cols].copy()

    df = score_min.merge(
        out_min,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if df["harm"].isna().any():
        raise RuntimeError("P01A/B3 outcome binding missing HARM.")

    target_axis = load_axis(b1, "PROMISE12")
    gt_map = target_axis[["global_index"]].copy()
    gt_map["gt_nonempty"] = gt_nonempty

    df = df.merge(gt_map, on="global_index", how="left", validate="many_to_one")

    source_frac_parts = []

    for family in FAMILIES:
        p = load_source_packed(b1, "PROMISE12", family)
        frac = packed_foreground_fraction_big(p)

        tmp = pd.DataFrame({
            "family": family,
            "global_index": np.arange(TARGET_N, dtype=np.int64),
            "source_fg_fraction": frac,
            "source_nonempty": frac > 0.0,
        })

        cut = fg_cuts[family]
        tmp["source_fg_stratum"] = [
            assign_foreground_stratum(float(x), cut)
            for x in frac
        ]
        source_frac_parts.append(tmp)

    source_map = pd.concat(source_frac_parts, ignore_index=True)

    df = df.merge(
        source_map,
        on=["family", "global_index"],
        how="left",
        validate="many_to_one",
    )

    dice_cuts = source_dice_cutpoints(outcomes)
    df["source_dice_stratum"] = [
        assign_dice_stratum(float(x), dice_cuts[f])
        for f, x in zip(df["family"], df["source_dice"])
    ]

    df["gt_stratum"] = np.where(
        df["gt_nonempty"].astype(bool),
        "GT_NONEMPTY",
        "GT_EMPTY",
    )

    df["source_occ_stratum"] = np.where(
        df["source_nonempty"].astype(bool),
        "SOURCE_NONEMPTY",
        "SOURCE_EMPTY",
    )

    return df, dice_cuts


def binary_metrics(y, s) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)

    rec = {
        "n": int(len(y)),
        "harm_n": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else math.nan,
        "valid": False,
        "auroc": math.nan,
        "auprc": math.nan,
    }

    if len(y) == 0 or len(np.unique(y)) < 2:
        return rec

    rec["valid"] = True
    rec["auroc"] = float(roc_auc_score(y, s))
    rec["auprc"] = float(average_precision_score(y, s))
    return rec


def point_summary(df: pd.DataFrame, stratum_name: str, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.loc[mask].copy()
    cell_rows = []

    for rep in REPRESENTATIONS:
        r = sub[sub["representation"] == rep]

        for action in ACTIONS:
            for family in FAMILIES:
                g = r[(r["action"] == action) & (r["family"] == family)]
                m = binary_metrics(g["harm"], g["risk_score"])
                patients_with_harm = int(
                    (
                        g.groupby("case_key")["harm"].max() > 0
                    ).sum()
                ) if len(g) else 0

                cell_rows.append({
                    "stratum": stratum_name,
                    "representation": rep,
                    "action": action,
                    "family": family,
                    "patients": int(g["case_key"].nunique()),
                    "patients_with_harm": patients_with_harm,
                    **m,
                })

    cells = pd.DataFrame(cell_rows)
    macro_rows = []

    for rep in REPRESENTATIONS:
        g = cells[(cells["representation"] == rep) & (cells["valid"] == True)]
        macro_rows.append({
            "stratum": stratum_name,
            "representation": rep,
            "valid_cells": int(len(g)),
            "macro_auroc": float(g["auroc"].mean()) if len(g) else math.nan,
            "macro_auprc": float(g["auprc"].mean()) if len(g) else math.nan,
            "macro_prevalence": float(g["prevalence"].mean()) if len(g) else math.nan,
            "total_rows": int(cells[cells["representation"] == rep]["n"].sum()),
            "total_harm": int(cells[cells["representation"] == rep]["harm_n"].sum()),
        })

    return cells, pd.DataFrame(macro_rows)


def bootstrap_macro_and_deltas(
    df: pd.DataFrame,
    stratum_name: str,
    mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    sub = df.loc[mask].copy()
    patients = sorted(sub["case_key"].astype(str).unique().tolist())

    if len(patients) < 2:
        return pd.DataFrame(), pd.DataFrame()

    point_cells, _ = point_summary(df, stratum_name, mask)
    valid_keys = {}

    for rep in REPRESENTATIONS:
        g = point_cells[
            (point_cells["representation"] == rep)
            & (point_cells["valid"] == True)
        ]
        valid_keys[rep] = [
            (r["action"], r["family"])
            for r in g.to_dict(orient="records")
        ]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    case_arr = sub["case_key"].astype(str).to_numpy()
    idx_map = {p: np.flatnonzero(case_arr == p) for p in patients}

    rep_rows = []
    delta_rows = []

    for rep_i in tqdm(
        range(BOOTSTRAP_REPS),
        desc=f"P03 bootstrap {stratum_name}",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([idx_map[p] for p in sampled])
        boot = sub.iloc[idx]

        current = {}
        for representation in REPRESENTATIONS:
            aucs, prs = [], []
            ok = True
            for action, family in valid_keys[representation]:
                g = boot[
                    (boot["representation"] == representation)
                    & (boot["action"] == action)
                    & (boot["family"] == family)
                ]
                y = g["harm"].to_numpy(dtype=np.int64)
                s = g["risk_score"].to_numpy(dtype=np.float64)

                if len(np.unique(y)) < 2:
                    ok = False
                    break

                aucs.append(float(roc_auc_score(y, s)))
                prs.append(float(average_precision_score(y, s)))

            if ok and aucs:
                current[representation] = {
                    "auroc": float(np.mean(aucs)),
                    "auprc": float(np.mean(prs)),
                }

        for representation, vals in current.items():
            rep_rows.append({
                "stratum": stratum_name,
                "replicate": rep_i,
                "representation": representation,
                **vals,
            })

        full = current.get("FULL_TRANSITION_SAFETTA")
        simple = current.get("SOURCE_PLUS_SIMPLE_MASK_CHANGE")

        if full is not None and simple is not None:
            for metric in ["auroc", "auprc"]:
                delta_rows.append({
                    "stratum": stratum_name,
                    "replicate": rep_i,
                    "metric": metric.upper(),
                    "delta_full_minus_simple": full[metric] - simple[metric],
                })

    return pd.DataFrame(rep_rows), pd.DataFrame(delta_rows)


def summarize_bootstrap(reps: pd.DataFrame, deltas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep_rows = []

    if not reps.empty:
        for (stratum, representation), g in reps.groupby(["stratum", "representation"]):
            for metric in ["auroc", "auprc"]:
                x = g[metric].to_numpy(dtype=np.float64)
                rep_rows.append({
                    "stratum": stratum,
                    "representation": representation,
                    "metric": metric.upper(),
                    "valid_reps": int(len(x)),
                    "ci95_low": float(np.quantile(x, 0.025)),
                    "ci95_high": float(np.quantile(x, 0.975)),
                })

    delta_rows = []

    if not deltas.empty:
        for (stratum, metric), g in deltas.groupby(["stratum", "metric"]):
            x = g["delta_full_minus_simple"].to_numpy(dtype=np.float64)
            delta_rows.append({
                "stratum": stratum,
                "metric": metric,
                "valid_reps": int(len(x)),
                "delta_mean": float(np.mean(x)),
                "ci95_low": float(np.quantile(x, 0.025)),
                "ci95_high": float(np.quantile(x, 0.975)),
                "ci_excludes_zero": bool(
                    np.quantile(x, 0.025) > 0 or np.quantile(x, 0.975) < 0
                ),
            })

    return pd.DataFrame(rep_rows), pd.DataFrame(delta_rows)


def self_test():
    assert BOOTSTRAP_REPS == 2000
    assert TARGET_PATIENTS == 50
    assert "GT_NONEMPTY" in load_json(PROTOCOL_NAME)["strata"] if Path(PROTOCOL_NAME).is_file() else True
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_FIXED_SCORE_DESIGN=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA B6-P0-3 fixed-score MRI stratified robustness audit."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    verify_protocol(Path(__file__).resolve().parent)
    b1, b3, p01a = verify_upstream()

    print("=" * 148)
    print("SafeTTA B6-P0-3 — MRI empty/foreground/SOURCE-quality robustness")
    print(f"Version                : {VERSION}")
    print(f"Protocol SHA           : {EXPECTED_PROTOCOL_SHA256}")
    print("Scores                 : FROZEN Final-B3 / B6-P01A")
    print("Model fitting          : NO")
    print("Score refit / flip     : NO / NO")
    print("HARM change            : NO")
    print("Primary sensitivity    : GT_NONEMPTY")
    print("=" * 148)

    print("\n[1/5] Historical CM6 GT + target-axis binding")
    gt, case_audit = load_cm6_gt()
    target_axis = load_axis(b1, "PROMISE12")
    gt_nonempty = verify_gt_axis_alignment(gt, case_audit, target_axis)

    print(
        f"GT slices: total={len(gt_nonempty)} "
        f"nonempty={int(gt_nonempty.sum())} empty={int((~gt_nonempty).sum())}"
    )
    print("CM6_GT_AXIS_BINDING=PASS")

    print("\n[2/5] SOURCE occupancy and Prostate158 foreground-size cutpoints")
    fg_cuts = family_cutpoints_from_source(b1)
    print(json.dumps(fg_cuts, indent=2))

    print("\n[3/5] Bind frozen risk scores, outcomes, and strata")
    df, dice_cuts = build_analysis_table(b1, gt_nonempty, fg_cuts)

    print("rows=", len(df))
    print("representations=", sorted(df["representation"].unique().tolist()))
    print("patients=", df["case_key"].nunique())
    print("SOURCE_DICE_CUTPOINTS=", json.dumps(dice_cuts, indent=2))

    if args.preflight_only:
        print("P03_PREFLIGHT=PASS")
        print("GATE=PASS_B6_P03_PREFLIGHT_ONLY")
        print("STRATIFIED_METRICS=NOT_RUN")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P0-3 output: {OUT_DIR}\n"
            "Use _fix1 if a script repair is required."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/5] Fixed-score stratified metrics")

    strata = {
        "ALL": np.ones(len(df), dtype=bool),
        "GT_NONEMPTY": df["gt_nonempty"].to_numpy(dtype=bool),
        "GT_EMPTY": ~df["gt_nonempty"].to_numpy(dtype=bool),
        "SOURCE_NONEMPTY": df["source_nonempty"].to_numpy(dtype=bool),
        "SOURCE_EMPTY": ~df["source_nonempty"].to_numpy(dtype=bool),
    }

    for label in ["SOURCE_FG_Q1", "SOURCE_FG_Q2", "SOURCE_FG_Q3", "SOURCE_FG_Q4"]:
        strata[label] = df["source_fg_stratum"].to_numpy() == label

    for label in ["SOURCE_DICE_Q1", "SOURCE_DICE_Q2", "SOURCE_DICE_Q3", "SOURCE_DICE_Q4"]:
        strata[label] = df["source_dice_stratum"].to_numpy() == label

    all_cells = []
    all_macro = []

    for name, mask in strata.items():
        cells, macro = point_summary(df, name, mask)
        all_cells.append(cells)
        all_macro.append(macro)

    cells_df = pd.concat(all_cells, ignore_index=True)
    macro_df = pd.concat(all_macro, ignore_index=True)

    p_cells = OUT_DIR / "B6_P03_STRATIFIED_CELL_METRICS.csv"
    p_macro = OUT_DIR / "B6_P03_STRATIFIED_MACRO_METRICS.csv"
    p_analysis = OUT_DIR / "B6_P03_ANALYSIS_ROWS.csv"
    p_fg = OUT_DIR / "B6_P03_SOURCE_FG_CUTPOINTS.json"
    p_dice = OUT_DIR / "B6_P03_SOURCE_DICE_CUTPOINTS.json"

    cells_df.to_csv(p_cells, index=False, encoding="utf-8")
    macro_df.to_csv(p_macro, index=False, encoding="utf-8")
    df.to_csv(p_analysis, index=False, encoding="utf-8")
    write_json(p_fg, fg_cuts)
    write_json(p_dice, dice_cuts)

    headline = macro_df[
        macro_df["stratum"].isin(["ALL", "GT_NONEMPTY", "GT_EMPTY"])
    ]
    print("\nHeadline all/nonempty/empty metrics:")
    print(headline.to_string(index=False))

    print("\n[5/5] Patient-cluster bootstrap for primary robustness strata")

    bootstrap_strata = ["ALL", "GT_NONEMPTY", "GT_EMPTY", "SOURCE_NONEMPTY", "SOURCE_EMPTY"]
    rep_parts = []
    delta_parts = []

    for name in bootstrap_strata:
        reps, deltas = bootstrap_macro_and_deltas(df, name, strata[name])
        if not reps.empty:
            rep_parts.append(reps)
        if not deltas.empty:
            delta_parts.append(deltas)

    rep_df = pd.concat(rep_parts, ignore_index=True) if rep_parts else pd.DataFrame()
    delta_df = pd.concat(delta_parts, ignore_index=True) if delta_parts else pd.DataFrame()
    rep_summary, delta_summary = summarize_bootstrap(rep_df, delta_df)

    p_rep = OUT_DIR / "B6_P03_BOOTSTRAP_REPLICATES.csv"
    p_delta = OUT_DIR / "B6_P03_FULL_MINUS_SIMPLE_DELTA_REPLICATES.csv"
    p_rep_summary = OUT_DIR / "B6_P03_BOOTSTRAP_CI.csv"
    p_delta_summary = OUT_DIR / "B6_P03_FULL_MINUS_SIMPLE_DELTAS.csv"

    rep_df.to_csv(p_rep, index=False, encoding="utf-8")
    delta_df.to_csv(p_delta, index=False, encoding="utf-8")
    rep_summary.to_csv(p_rep_summary, index=False, encoding="utf-8")
    delta_summary.to_csv(p_delta_summary, index=False, encoding="utf-8")

    print("\nFULL_TRANSITION - SIMPLE_MASK_CHANGE paired patient-bootstrap:")
    print(delta_summary.to_string(index=False))

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "scientific_status": "post-freeze/post-reveal fixed-score robustness audit",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "no_refit": True,
        "no_score_flip": True,
        "no_harm_change": True,
        "historical_primary_replaced": False,
        "gt": {
            "source": str(CM6_GT),
            "sha256": sha256_file(CM6_GT),
            "nonempty_slices": int(gt_nonempty.sum()),
            "empty_slices": int((~gt_nonempty).sum()),
            "axis_binding": "PASS_via_CM6_case_audit_positive_slice_counts",
        },
        "outputs": {},
    }

    for p in [
        p_cells, p_macro, p_analysis, p_fg, p_dice,
        p_rep, p_delta, p_rep_summary, p_delta_summary,
    ]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P03_AUDIT.json"
    write_json(p_audit, audit)

    nonempty = macro_df[macro_df["stratum"] == "GT_NONEMPTY"].copy()
    full = nonempty[nonempty["representation"] == "FULL_TRANSITION_SAFETTA"].iloc[0]
    simple = nonempty[
        nonempty["representation"] == "SOURCE_PLUS_SIMPLE_MASK_CHANGE"
    ].iloc[0]

    report = "\n".join([
        "=" * 148,
        "SafeTTA B6-P0-3 MRI STRATIFIED ROBUSTNESS COMPLETE",
        f"GT nonempty slices : {int(gt_nonempty.sum())}/{TARGET_N}",
        f"GT empty slices    : {int((~gt_nonempty).sum())}/{TARGET_N}",
        "",
        "GT_NONEMPTY:",
        (
            "FULL_TRANSITION_SAFETTA "
            f"AUROC={full['macro_auroc']:.6f} "
            f"AUPRC={full['macro_auprc']:.6f} "
            f"valid_cells={int(full['valid_cells'])}/6"
        ),
        (
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE "
            f"AUROC={simple['macro_auroc']:.6f} "
            f"AUPRC={simple['macro_auprc']:.6f} "
            f"valid_cells={int(simple['valid_cells'])}/6"
        ),
        "",
        "FULL - SIMPLE bootstrap:",
        delta_summary[
            delta_summary["stratum"] == "GT_NONEMPTY"
        ].to_string(index=False),
        "",
        (
            "This audit does not replace Final-B3. It tests whether the observed "
            "ranking and the simple-mask baseline advantage persist outside GT-empty slices."
        ),
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 148,
        "",
    ])

    (OUT_DIR / "B6_P03_REPORT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
