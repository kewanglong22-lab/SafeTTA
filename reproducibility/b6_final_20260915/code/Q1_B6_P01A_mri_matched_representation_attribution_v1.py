#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1A — MRI matched-representation transition attribution.

POST-FREEZE / POST-REVEAL explanatory audit.

For each held-out action:
  fit on Prostate158 other-two-action rows only,
  evaluate held-out action on PROMISE12.

Representations:
  1) SOURCE state only (66-D)
  2) semantic transition only (64-D)
  3) SOURCE state + simple candidate-mask change (70-D)
  4) frozen Final-B3 full transition SafeTTA (130-D score; NOT refit here)

No reliability-change baseline: P0-4 fix1 found no strongly bound historical
probability/logit asset suitable for a fair exact-panel comparison.

No target fit, no target feature selection, no score flip, no HARM change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_NAME = "B6_P01A_MRI_MATCHED_REPRESENTATION_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "2eea6c75886ebe715f5b2e161761c400ae2688702a672c006676232b08b1fb67"

B1_DIR = ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3"
B1_LOCK = B1_DIR / "FINALB1_MRI_ACTION_CANDIDATE_MASK_LOCK.json"
EXPECTED_B1_SHA = "5900b35a6971ceff32ddd6e22e8332cf9fdf68e65d9a6c2711c52a4d324eacc5"

B2_DIR = ROOT / "FinalB2_mri_current_runtime_candidate_conditioned_s64_lock_v1"
B2_LOCK = B2_DIR / "FINALB2_MRI_S64_LOCK.json"
EXPECTED_B2_SHA = "def3a692f50342d78781e7be0a38b5afcf1e34962ce20c21801510a6cf985022"

B3_DIR = ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1"
B3_LOCK = B3_DIR / "FINALB3_MRI_FUTURE_HARM_LOCK.json"
EXPECTED_B3_SHA = "924b8b882c971be7a5562b7d0fc53cf1eece343786f0ff5f24d46b91a86a191b"
B3_SOURCE_OUTCOMES = B3_DIR / "FINALB3_PROSTATE158_ACTION_OUTCOMES.csv"
B3_TARGET_OUTCOMES = B3_DIR / "FINALB3_PROMISE12_ACTION_OUTCOMES.csv"
B3_SOURCE_SCORES = B3_DIR / "FINALB3_SOURCE_LOAO_SCORES.csv"
B3_TARGET_SCORES = B3_DIR / "FINALB3_PROMISE12_JOINT_SHIFT_LOAO_SCORES.csv"

P04_FIX1_DIR = ROOT / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1_fix1"
P04_FIX1_AUDIT = P04_FIX1_DIR / "B6_P04_FIX1_AUDIT.json"
EXPECTED_P04_GATE = "PASS_B6_P04_FIX1_EXACT_ID_AND_PANEL_BINDING_AUDIT_COMPLETE"

OUT_DIR = ROOT / "B6_P01A_mri_matched_representation_attribution_v1"
PASS_GATE = "PASS_B6_P01A_MRI_MATCHED_REPRESENTATION_ATTRIBUTION_COMPLETE"

FAMILIES = ["DeepLabV3_R50", "SegFormer_B0"]
ACTIONS = ["TENT1", "PL-CONF90", "MEMO-SEG4-1STEP"]

SOURCE_N = 3553
TARGET_N = 1377
TARGET_PATIENTS = 50

H = W = 352
PACKED_BYTES = H * W // 8

LR_C = 1.0
LR_MAX_ITER = 5000
LR_RANDOM_STATE = 20260820

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260916

REPRESENTATIONS = [
    "SOURCE_STATE",
    "SEMANTIC_TRANSITION",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    "FULL_TRANSITION_SAFETTA",
]

COMPARATORS = [
    "SOURCE_STATE",
    "SEMANTIC_TRANSITION",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
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
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}"
        )
    return got


def recorded_path(value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def verify_protocol(script_dir: Path) -> Dict[str, Any]:
    p = script_dir / PROTOCOL_NAME
    exact_sha(p, EXPECTED_PROTOCOL_SHA256, "P01A_PROTOCOL")
    d = load_json(p)

    if d.get("status") != "FROZEN_POST_REVEAL_MATCHED_ATTRIBUTION":
        raise RuntimeError("P01A protocol status changed.")
    if d["reliability_change_baseline"]["included"] is not False:
        raise RuntimeError("Reliability-change baseline unexpectedly enabled.")
    if int(d["training"]["target_fit_rows"]) != 0:
        raise RuntimeError("Target-fit boundary changed.")
    if int(d["statistics"]["bootstrap_reps"]) != BOOTSTRAP_REPS:
        raise RuntimeError("Bootstrap count changed.")

    return d


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    exact_sha(B1_LOCK, EXPECTED_B1_SHA, "FinalB1 lock")
    exact_sha(B2_LOCK, EXPECTED_B2_SHA, "FinalB2 lock")
    exact_sha(B3_LOCK, EXPECTED_B3_SHA, "FinalB3 lock")

    b1 = load_json(B1_LOCK)
    b2 = load_json(B2_LOCK)
    b3 = load_json(B3_LOCK)

    if b1.get("status") != "PASS":
        raise RuntimeError("FinalB1 not PASS.")
    if b2.get("status") != "PASS":
        raise RuntimeError("FinalB2 not PASS.")
    if b3.get("status") != "PASS":
        raise RuntimeError("FinalB3 not PASS.")

    if b1.get("families") != FAMILIES:
        raise RuntimeError("FinalB1 family set changed.")
    if b2.get("families") != FAMILIES:
        raise RuntimeError("FinalB2 family set changed.")
    if b2.get("actions") != ACTIONS:
        raise RuntimeError("FinalB2 action set changed.")

    if not P04_FIX1_AUDIT.is_file():
        raise FileNotFoundError(P04_FIX1_AUDIT)

    p04 = load_json(P04_FIX1_AUDIT)
    if p04.get("status") != "PASS" or p04.get("gate") != EXPECTED_P04_GATE:
        raise RuntimeError("P0-4 fix1 gate changed.")

    rel = p04["decisions"]["reliability_change_baseline"]["status"]
    if rel != "NOT_RECOVERABLE_FOR_P0_1_FROM_CURRENT_EXACT_BINDING_AUDIT":
        raise RuntimeError(
            "P0-4 fix1 reliability-change decision changed; "
            f"observed={rel}"
        )

    promise_cluster = (
        p04["decisions"]["PROMISE12"]["P0_1_cluster_unit_decision"]
    )
    if promise_cluster != "patient":
        raise RuntimeError(
            "PROMISE12 cluster decision is no longer patient: "
            f"{promise_cluster}"
        )

    return b1, b2, b3


def unpack_big(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    flat = np.unpackbits(
        p,
        axis=-1,
        count=H * W,
        bitorder="big",
    )
    return flat.reshape(*p.shape[:-1], H, W).astype(bool, copy=False)


def dice_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum(axis=(1, 2), dtype=np.int64)
    den = (
        a.sum(axis=(1, 2), dtype=np.int64)
        + b.sum(axis=(1, 2), dtype=np.int64)
    )
    out = np.ones(len(a), dtype=np.float64)
    nz = den != 0
    out[nz] = 2.0 * inter[nz] / den[nz]
    return out


def iou_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(a, b).sum(axis=(1, 2), dtype=np.int64)
    out = np.ones(len(a), dtype=np.float64)
    nz = union != 0
    out[nz] = inter[nz] / union[nz]
    return out


def mask_morphology_bool(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=bool)
    fg = m.mean(axis=(1, 2), dtype=np.float64)

    denom = float((H - 1) * W + H * (W - 1))
    vertical = np.not_equal(
        m[:, 1:, :],
        m[:, :-1, :],
    ).sum(axis=(1, 2), dtype=np.int64)
    horizontal = np.not_equal(
        m[:, :, 1:],
        m[:, :, :-1],
    ).sum(axis=(1, 2), dtype=np.int64)
    boundary = (vertical + horizontal).astype(np.float64) / denom

    return np.column_stack([fg, boundary])


def source_morphology_from_packed(
    packed: np.ndarray,
    batch: int = 128,
) -> np.ndarray:
    n = len(packed)
    out = np.empty((n, 2), dtype=np.float64)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        m = unpack_big(np.asarray(packed[start:end]))
        out[start:end] = mask_morphology_bool(m)

    return out


def simple_mask_change_from_packed(
    source_packed: np.ndarray,
    candidate_packed: np.ndarray,
    batch: int = 64,
) -> np.ndarray:
    n = len(source_packed)
    out = np.empty((n, 4), dtype=np.float64)

    for start in range(0, n, batch):
        end = min(start + batch, n)

        src = unpack_big(np.asarray(source_packed[start:end]))
        can = unpack_big(np.asarray(candidate_packed[start:end]))

        pred_dice = dice_between(src, can)
        pred_iou = iou_between(src, can)

        sm = mask_morphology_bool(src)
        cm = mask_morphology_bool(can)

        area_delta = cm[:, 0] - sm[:, 0]
        boundary_delta = cm[:, 1] - sm[:, 1]

        out[start:end] = np.column_stack([
            pred_dice,
            pred_iou,
            area_delta,
            boundary_delta,
        ])

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite simple mask-change features.")

    return out


def load_axis_from_b1(
    b1: Dict[str, Any],
    cohort: str,
) -> pd.DataFrame:
    rec = b1["cohorts"][cohort]
    path = recorded_path(rec["safe_axis"])
    exact_sha(
        path,
        rec["safe_axis_sha256"],
        f"{cohort} safe axis",
    )
    df = pd.read_csv(path, low_memory=False)

    n = SOURCE_N if cohort == "Prostate158" else TARGET_N

    if len(df) != n:
        raise RuntimeError(f"{cohort} safe-axis rows changed.")
    if df["global_index"].astype(int).tolist() != list(range(n)):
        raise RuntimeError(f"{cohort} global_index order changed.")

    return df


def load_b1_panel(
    b1: Dict[str, Any],
    cohort: str,
    family: str,
) -> Dict[str, np.ndarray]:
    prefix = "source::" if cohort == "Prostate158" else "target::"
    key = prefix + family
    rec = b1["unit_locks"][key]

    lock_path = recorded_path(rec["path"])
    exact_sha(
        lock_path,
        rec["sha256"],
        f"B1 unit {key}",
    )

    lock = load_json(lock_path)
    panel = lock["new_current_runtime_panel"]

    result = {}

    for state in ["SOURCE", *ACTIONS]:
        sr = panel[state]
        p = recorded_path(sr["path"])
        exact_sha(
            p,
            sr["sha256"],
            f"B1 {key} {state}",
        )

        a = np.load(p, mmap_mode="r")
        n = SOURCE_N if cohort == "Prostate158" else TARGET_N

        if a.shape != (n, PACKED_BYTES):
            raise RuntimeError(
                f"{key}/{state} shape changed: {a.shape}"
            )
        if a.dtype != np.uint8:
            raise RuntimeError(
                f"{key}/{state} dtype changed: {a.dtype}"
            )

        result[state] = a

    return result


def load_b2_arrays(
    b2: Dict[str, Any],
    cohort: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    crec = b2["cohorts"][cohort]
    lock_path = recorded_path(crec["lock_path"])

    exact_sha(
        lock_path,
        crec["lock_sha256"],
        f"B2 {cohort} lock",
    )

    lock = load_json(lock_path)

    zr = lock["semantic64"]["SOURCE"]
    zp = recorded_path(zr["path"])
    exact_sha(
        zp,
        zr["sha256"],
        f"B2 {cohort} SOURCE z64",
    )
    zsrc = np.load(zp, mmap_mode="r")

    s64 = {}

    for action in ACTIONS:
        ar = lock["s64"][action]
        p = recorded_path(ar["path"])
        exact_sha(
            p,
            ar["sha256"],
            f"B2 {cohort} {action} S64",
        )
        s64[action] = np.load(p, mmap_mode="r")

    n = SOURCE_N if cohort == "Prostate158" else TARGET_N
    expected = (len(FAMILIES), n, 64)

    if zsrc.shape != expected:
        raise RuntimeError(
            f"{cohort} z64 shape changed: {zsrc.shape}"
        )

    for action, a in s64.items():
        if a.shape != expected:
            raise RuntimeError(
                f"{cohort}/{action} S64 shape changed: {a.shape}"
            )

    return zsrc, s64


def load_outcomes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        B3_SOURCE_OUTCOMES,
        low_memory=False,
    )
    target = pd.read_csv(
        B3_TARGET_OUTCOMES,
        low_memory=False,
    )

    for name, df, n in [
        (
            "Prostate158",
            source,
            SOURCE_N * len(FAMILIES) * len(ACTIONS),
        ),
        (
            "PROMISE12",
            target,
            TARGET_N * len(FAMILIES) * len(ACTIONS),
        ),
    ]:
        if len(df) != n:
            raise RuntimeError(
                f"{name} outcome rows changed: {len(df)} != {n}"
            )

    return source, target


def load_frozen_full_scores(
    b3: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        B3_SOURCE_SCORES,
        low_memory=False,
    )
    target = pd.read_csv(
        B3_TARGET_SCORES,
        low_memory=False,
    )

    if len(source) != SOURCE_N * len(FAMILIES) * len(ACTIONS):
        raise RuntimeError("Final-B3 source score rows changed.")
    if len(target) != TARGET_N * len(FAMILIES) * len(ACTIONS):
        raise RuntimeError("Final-B3 target score rows changed.")

    # Bind to B3 lock SHA entries.
    exact_sha(
        B3_SOURCE_SCORES,
        b3["scores"]["source_LOAO"]["sha256"],
        "Final-B3 source scores",
    )
    exact_sha(
        B3_TARGET_SCORES,
        b3["scores"]["PROMISE12_joint_shift_LOAO"]["sha256"],
        "Final-B3 target scores",
    )

    return source, target


def build_feature_panels(
    b1: Dict[str, Any],
    b2: Dict[str, Any],
    cohort: str,
) -> Tuple[
    Dict[str, np.ndarray],
    pd.DataFrame,
]:
    axis = load_axis_from_b1(
        b1,
        cohort,
    )
    zsrc, s64 = load_b2_arrays(
        b2,
        cohort,
    )

    n = SOURCE_N if cohort == "Prostate158" else TARGET_N

    q66_rows = []
    s64_rows = []
    simple70_rows = []
    full130_rows = []
    meta_rows = []

    for fi, family in enumerate(FAMILIES):
        panel = load_b1_panel(
            b1,
            cohort,
            family,
        )

        source_morph = source_morphology_from_packed(
            panel["SOURCE"],
        )
        q66 = np.concatenate([
            source_morph,
            np.asarray(zsrc[fi], dtype=np.float64),
        ], axis=1)

        if q66.shape != (n, 66):
            raise RuntimeError(
                f"{cohort}/{family} q66 shape={q66.shape}"
            )

        for action in ACTIONS:
            delta = np.asarray(
                s64[action][fi],
                dtype=np.float64,
            )

            simple = simple_mask_change_from_packed(
                panel["SOURCE"],
                panel[action],
            )

            source_plus_simple = np.concatenate([
                q66,
                simple,
            ], axis=1)

            full = np.concatenate([
                q66,
                delta,
            ], axis=1)

            if source_plus_simple.shape != (n, 70):
                raise RuntimeError("70-D feature shape error.")
            if full.shape != (n, 130):
                raise RuntimeError("130-D feature shape error.")

            q66_rows.append(q66)
            s64_rows.append(delta)
            simple70_rows.append(source_plus_simple)
            full130_rows.append(full)

            for gi in range(n):
                r = axis.iloc[gi]
                rec = {
                    "cohort": cohort,
                    "family": family,
                    "action": action,
                    "global_index": int(gi),
                    "case_key": str(r["case_key"]),
                    "slice_index": int(r["slice_index"]),
                }
                if cohort == "Prostate158":
                    rec["fold"] = int(r["fold"])
                else:
                    rec["case_id"] = str(r["case_id"])
                meta_rows.append(rec)

    features = {
        "SOURCE_STATE": np.concatenate(
            q66_rows,
            axis=0,
        ),
        "SEMANTIC_TRANSITION": np.concatenate(
            s64_rows,
            axis=0,
        ),
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE": np.concatenate(
            simple70_rows,
            axis=0,
        ),
        "FULL_130_REPLAY": np.concatenate(
            full130_rows,
            axis=0,
        ),
    }

    meta = pd.DataFrame(meta_rows)

    expected_rows = n * len(FAMILIES) * len(ACTIONS)

    for name, x in features.items():
        if len(x) != expected_rows:
            raise RuntimeError(
                f"{cohort}/{name} rows changed."
            )
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"{cohort}/{name} has non-finite values."
            )

    if len(meta) != expected_rows:
        raise RuntimeError(
            f"{cohort} feature metadata rows changed."
        )

    return features, meta


def bind_meta_to_outcomes(
    meta: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "cohort",
        "family",
        "action",
        "global_index",
        "case_key",
        "slice_index",
    ]

    merged = meta.merge(
        outcomes,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )

    if len(merged) != len(meta):
        raise RuntimeError("Metadata/outcome row count changed.")
    if merged["harm"].isna().any():
        raise RuntimeError("Metadata/outcome binding has missing HARM.")

    return merged


def fit_ranker(
    X: np.ndarray,
    y: np.ndarray,
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            "Training rows do not contain both HARM classes."
        )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty="l2",
        C=LR_C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=LR_RANDOM_STATE,
    )

    clf.fit(
        Xs,
        y,
    )

    if list(clf.classes_) != [0, 1]:
        raise RuntimeError(
            f"Unexpected classes: {clf.classes_}"
        )

    return scaler, clf


def score_ranker(
    scaler,
    clf,
    X: np.ndarray,
) -> np.ndarray:
    p = clf.predict_proba(
        scaler.transform(X),
    )
    pos = int(
        np.flatnonzero(clf.classes_ == 1)[0]
    )
    score = np.asarray(
        p[:, pos],
        dtype=np.float64,
    )

    if not np.isfinite(score).all():
        raise RuntimeError("Non-finite score.")

    return score


def score_full_replay(
    b3: Dict[str, Any],
    target_meta: pd.DataFrame,
    target_full130: np.ndarray,
    frozen_target_scores: pd.DataFrame,
) -> None:
    """
    Mandatory no-metric preflight:
    reconstruct the exact 130-D features and use the frozen B3 models.
    Scores must reproduce the saved B3 risk scores.
    """
    model_records = b3["ranking_head"]["models"]

    for action in ACTIONS:
        rec = model_records[action]
        p = recorded_path(rec["path"])

        exact_sha(
            p,
            rec["sha256"],
            f"B3 model {action}",
        )

        bundle = joblib.load(p)
        scaler = bundle["scaler"]
        clf = bundle["classifier"]

        mask = (
            target_meta["action"].to_numpy()
            == action
        )

        replay = score_ranker(
            scaler,
            clf,
            target_full130[mask],
        )

        target_subset = target_meta.loc[
            mask,
            [
                "cohort",
                "family",
                "action",
                "global_index",
                "case_key",
                "slice_index",
            ],
        ].copy()

        saved = frozen_target_scores[
            frozen_target_scores["action"] == action
        ].copy()

        keys = [
            "cohort",
            "family",
            "action",
            "global_index",
            "case_key",
            "slice_index",
        ]

        joined = target_subset.merge(
            saved[
                keys
                + ["risk_score"]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
        )

        if joined["risk_score"].isna().any():
            raise RuntimeError(
                f"{action}: missing frozen scores."
            )

        diff = np.abs(
            replay
            - joined["risk_score"].to_numpy(
                dtype=np.float64,
            )
        )

        max_abs = float(diff.max())

        if max_abs > 1e-10:
            raise RuntimeError(
                f"{action}: B3 full-score replay mismatch "
                f"max_abs={max_abs:.3e}"
            )

        print(
            f"FULL_REPLAY {action}: PASS "
            f"max_abs={max_abs:.3e}"
        )


def binary_metrics(
    y: np.ndarray,
    s: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )

    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)

    prev = float(y.mean()) if len(y) else math.nan

    if len(np.unique(y)) < 2:
        return {
            "n": int(len(y)),
            "harm": int(y.sum()),
            "prevalence": prev,
            "auroc": math.nan,
            "auprc": math.nan,
        }

    return {
        "n": int(len(y)),
        "harm": int(y.sum()),
        "prevalence": prev,
        "auroc": float(
            roc_auc_score(
                y,
                s,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                s,
            )
        ),
    }


def cell_metrics(
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for representation in REPRESENTATIONS:
        r_df = score_df[
            score_df["representation"]
            == representation
        ]

        for action in ACTIONS:
            for family in FAMILIES:
                g = r_df[
                    (r_df["action"] == action)
                    & (r_df["family"] == family)
                ]

                m = binary_metrics(
                    g["harm"],
                    g["risk_score"],
                )

                rows.append({
                    "representation": representation,
                    "action": action,
                    "family": family,
                    **m,
                })

    return pd.DataFrame(rows)


def macro_from_cells(
    cells: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for representation in REPRESENTATIONS:
        g = cells[
            cells["representation"]
            == representation
        ]

        rows.append({
            "representation": representation,
            "macro_action_family_auroc":
                float(g["auroc"].mean()),
            "macro_action_family_auprc":
                float(g["auprc"].mean()),
            "macro_action_family_prevalence":
                float(g["prevalence"].mean()),
        })

    return pd.DataFrame(rows)


def bootstrap_paired_deltas(
    score_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Patient-cluster bootstrap on PROMISE12.
    For every replicate, compute macro over 3 actions x 2 families.
    """
    rng = np.random.default_rng(
        BOOTSTRAP_SEED,
    )

    patients = sorted(
        score_df["case_key"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(patients) != TARGET_PATIENTS:
        raise RuntimeError(
            f"PROMISE12 patients={len(patients)} "
            f"!= {TARGET_PATIENTS}"
        )

    case_arr = (
        score_df["case_key"]
        .astype(str)
        .to_numpy()
    )

    patient_rows = {
        p: np.flatnonzero(
            case_arr == p
        )
        for p in patients
    }

    rep_rows = []
    delta_rows = []

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P01A patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )

        idx = np.concatenate([
            patient_rows[p]
            for p in sampled
        ])

        boot = score_df.iloc[idx]

        rep_metrics = {}

        valid_rep = True

        for representation in REPRESENTATIONS:
            vals_auc = []
            vals_pr = []

            r_df = boot[
                boot["representation"]
                == representation
            ]

            for action in ACTIONS:
                for family in FAMILIES:
                    g = r_df[
                        (r_df["action"] == action)
                        & (r_df["family"] == family)
                    ]

                    m = binary_metrics(
                        g["harm"],
                        g["risk_score"],
                    )

                    if (
                        not np.isfinite(m["auroc"])
                        or not np.isfinite(m["auprc"])
                    ):
                        valid_rep = False
                        break

                    vals_auc.append(
                        m["auroc"]
                    )
                    vals_pr.append(
                        m["auprc"]
                    )

                if not valid_rep:
                    break

            if not valid_rep:
                break

            rep_metrics[representation] = {
                "auroc": float(
                    np.mean(vals_auc)
                ),
                "auprc": float(
                    np.mean(vals_pr)
                ),
            }

        if not valid_rep:
            continue

        row = {
            "rep": rep,
        }

        for representation in REPRESENTATIONS:
            row[
                representation
                + "__macro_auroc"
            ] = rep_metrics[
                representation
            ]["auroc"]

            row[
                representation
                + "__macro_auprc"
            ] = rep_metrics[
                representation
            ]["auprc"]

        rep_rows.append(row)

        full = rep_metrics[
            "FULL_TRANSITION_SAFETTA"
        ]

        for comp in COMPARATORS:
            delta_rows.append({
                "rep": rep,
                "comparator": comp,
                "metric": "AUROC",
                "delta_full_minus_comparator":
                    full["auroc"]
                    - rep_metrics[comp]["auroc"],
            })

            delta_rows.append({
                "rep": rep,
                "comparator": comp,
                "metric": "AUPRC",
                "delta_full_minus_comparator":
                    full["auprc"]
                    - rep_metrics[comp]["auprc"],
            })

    reps = pd.DataFrame(rep_rows)
    deltas = pd.DataFrame(delta_rows)

    if len(reps) < int(
        0.95 * BOOTSTRAP_REPS
    ):
        raise RuntimeError(
            "Too many invalid bootstrap replicates: "
            f"{len(reps)}/{BOOTSTRAP_REPS}"
        )

    return reps, deltas


def summarize_deltas(
    deltas: pd.DataFrame,
    point_macro: pd.DataFrame,
) -> pd.DataFrame:
    point = {
        r["representation"]: r
        for r in point_macro.to_dict(
            orient="records"
        )
    }

    rows = []

    full = point[
        "FULL_TRANSITION_SAFETTA"
    ]

    metric_to_col = {
        "AUROC": "macro_action_family_auroc",
        "AUPRC": "macro_action_family_auprc",
    }

    for comp in COMPARATORS:
        for metric in [
            "AUROC",
            "AUPRC",
        ]:
            g = deltas[
                (deltas["comparator"] == comp)
                & (deltas["metric"] == metric)
            ][
                "delta_full_minus_comparator"
            ].to_numpy(
                dtype=np.float64,
            )

            col = metric_to_col[metric]

            point_delta = float(
                full[col]
                - point[comp][col]
            )

            lo = float(
                np.quantile(
                    g,
                    0.025,
                )
            )
            hi = float(
                np.quantile(
                    g,
                    0.975,
                )
            )

            rows.append({
                "comparator": comp,
                "metric": metric,
                "point_delta_full_minus_comparator":
                    point_delta,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_excludes_zero":
                    bool(
                        lo > 0.0
                        or hi < 0.0
                    ),
                "full_better_at_point":
                    bool(
                        point_delta > 0.0
                    ),
            })

    return pd.DataFrame(rows)


def make_method_scores(
    source_features: Dict[str, np.ndarray],
    target_features: Dict[str, np.ndarray],
    source_meta: pd.DataFrame,
    target_meta: pd.DataFrame,
    source_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    frozen_source_scores: pd.DataFrame,
    frozen_target_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    source_parts = []
    target_parts = []
    model_records = {}

    feature_map = {
        "SOURCE_STATE": "SOURCE_STATE",
        "SEMANTIC_TRANSITION": "SEMANTIC_TRANSITION",
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE":
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    }

    # Newly fit matched baselines.
    for heldout in ACTIONS:
        source_train_mask = (
            source_meta["action"].to_numpy()
            != heldout
        )
        source_test_mask = (
            source_meta["action"].to_numpy()
            == heldout
        )
        target_test_mask = (
            target_meta["action"].to_numpy()
            == heldout
        )

        for rep, feature_key in feature_map.items():
            Xs = source_features[
                feature_key
            ]
            Xt = target_features[
                feature_key
            ]

            y_train = source_rows.loc[
                source_train_mask,
                "harm",
            ].to_numpy(
                dtype=np.int64,
            )

            scaler, clf = fit_ranker(
                Xs[source_train_mask],
                y_train,
            )

            src_score = score_ranker(
                scaler,
                clf,
                Xs[source_test_mask],
            )

            tgt_score = score_ranker(
                scaler,
                clf,
                Xt[target_test_mask],
            )

            src = source_rows.loc[
                source_test_mask
            ].copy()

            tgt = target_rows.loc[
                target_test_mask
            ].copy()

            src["representation"] = rep
            tgt["representation"] = rep
            src["risk_score"] = src_score
            tgt["risk_score"] = tgt_score

            source_parts.append(
                src
            )
            target_parts.append(
                tgt
            )

            model_records[
                heldout + "::" + rep
            ] = {
                "heldout_action": heldout,
                "representation": rep,
                "training_rows":
                    int(source_train_mask.sum()),
                "training_harm":
                    int(y_train.sum()),
                "feature_dim":
                    int(
                        Xs.shape[1]
                    ),
                "target_rows_in_fit": 0,
            }

    # Frozen B3 full scores; no refit.
    full_source = frozen_source_scores.copy()
    full_target = frozen_target_scores.copy()

    full_source[
        "representation"
    ] = "FULL_TRANSITION_SAFETTA"

    full_target[
        "representation"
    ] = "FULL_TRANSITION_SAFETTA"

    source_parts.append(
        full_source
    )
    target_parts.append(
        full_target
    )

    source_scores = pd.concat(
        source_parts,
        ignore_index=True,
    )
    target_scores = pd.concat(
        target_parts,
        ignore_index=True,
    )

    return (
        source_scores,
        target_scores,
        model_records,
    )


def bind_features_to_outcomes(
    meta: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "cohort",
        "family",
        "action",
        "global_index",
        "case_key",
        "slice_index",
    ]

    merged = meta.merge(
        outcomes,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )

    if merged["harm"].isna().any():
        raise RuntimeError(
            "Feature/outcome merge produced missing HARM."
        )

    return merged


def self_test():
    assert FAMILIES == [
        "DeepLabV3_R50",
        "SegFormer_B0",
    ]
    assert ACTIONS == [
        "TENT1",
        "PL-CONF90",
        "MEMO-SEG4-1STEP",
    ]
    assert BOOTSTRAP_REPS == 2000
    assert LR_C == 1.0

    a = np.zeros(
        (1, H, W),
        dtype=bool,
    )
    b = np.zeros_like(a)

    assert dice_between(
        a,
        b,
    )[0] == 1.0

    assert iou_between(
        a,
        b,
    )[0] == 1.0

    b[0, 0, 0] = True

    assert dice_between(
        a,
        b,
    )[0] == 0.0

    assert iou_between(
        a,
        b,
    )[0] == 0.0

    print("SELF_TEST_MASK_CHANGE=PASS")
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA B6-P0-1A MRI same-panel matched-representation "
            "attribution audit."
        )
    )

    ap.add_argument(
        "--self-test",
        action="store_true",
    )

    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify locks/features and reproduce frozen B3 full scores. "
            "No new baseline fit or AUROC/AUPRC."
        ),
    )

    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    protocol = verify_protocol(
        Path(__file__).resolve().parent,
    )

    b1, b2, b3 = verify_upstream()

    print("=" * 152)
    print(
        "SafeTTA B6-P0-1A — MRI matched-representation transition attribution"
    )
    print(f"Version                  : {VERSION}")
    print(f"Protocol SHA             : {EXPECTED_PROTOCOL_SHA256}")
    print("Scientific status         : POST-FREEZE / POST-REVEAL attribution audit")
    print("Train                     : Prostate158 other-two actions")
    print("Test                      : PROMISE12 held-out action")
    print("Target rows in fit        : 0")
    print("Test bootstrap cluster    : patient / case_key")
    print("Reliability-change        : NOT INCLUDED (not exactly recoverable)")
    print("HARM threshold            : unchanged Final-B3 definition")
    print("Full transition score     : frozen Final-B3 score; NOT refit")
    print("=" * 152)

    source_outcomes, target_outcomes = load_outcomes()
    frozen_source, frozen_target = load_frozen_full_scores(
        b3,
    )

    print("\n[1/4] Reconstruct frozen feature panels")
    source_features, source_meta = build_feature_panels(
        b1,
        b2,
        "Prostate158",
    )

    target_features, target_meta = build_feature_panels(
        b1,
        b2,
        "PROMISE12",
    )

    source_rows = bind_features_to_outcomes(
        source_meta,
        source_outcomes,
    )

    target_rows = bind_features_to_outcomes(
        target_meta,
        target_outcomes,
    )

    print(
        "SOURCE_STATE:",
        source_features["SOURCE_STATE"].shape,
        target_features["SOURCE_STATE"].shape,
    )
    print(
        "SEMANTIC_TRANSITION:",
        source_features["SEMANTIC_TRANSITION"].shape,
        target_features["SEMANTIC_TRANSITION"].shape,
    )
    print(
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE:",
        source_features[
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE"
        ].shape,
        target_features[
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE"
        ].shape,
    )

    print("\n[2/4] Frozen Final-B3 score replay")
    score_full_replay(
        b3,
        target_meta,
        target_features["FULL_130_REPLAY"],
        frozen_target,
    )

    print("P01A_PREFLIGHT=PASS")

    if args.preflight_only:
        print(
            "GATE=PASS_B6_P01A_PREFLIGHT_ONLY"
        )
        print(
            "NEW_BASELINE_FIT_AND_METRICS=NOT_RUN"
        )
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            "Never overwrite P0-1A output:\n"
            f"  {OUT_DIR}\n"
            "Use _fix1 if the script requires repair."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("\n[3/4] Fit matched source-only baselines and score PROMISE12")

    (
        source_scores,
        target_scores,
        model_records,
    ) = make_method_scores(
        source_features,
        target_features,
        source_meta,
        target_meta,
        source_rows,
        target_rows,
        frozen_source,
        frozen_target,
    )

    p_source_scores = (
        OUT_DIR
        / "B6_P01A_SOURCE_LOAO_MATCHED_SCORES.csv"
    )
    p_target_scores = (
        OUT_DIR
        / "B6_P01A_PROMISE12_MATCHED_SCORES.csv"
    )

    source_scores.to_csv(
        p_source_scores,
        index=False,
        encoding="utf-8",
    )

    target_scores.to_csv(
        p_target_scores,
        index=False,
        encoding="utf-8",
    )

    source_cells = cell_metrics(
        source_scores,
    )
    target_cells = cell_metrics(
        target_scores,
    )

    source_macro = macro_from_cells(
        source_cells,
    )
    target_macro = macro_from_cells(
        target_cells,
    )

    p_source_cells = (
        OUT_DIR
        / "B6_P01A_SOURCE_CELL_METRICS.csv"
    )
    p_target_cells = (
        OUT_DIR
        / "B6_P01A_PROMISE12_CELL_METRICS.csv"
    )
    p_source_macro = (
        OUT_DIR
        / "B6_P01A_SOURCE_MACRO_METRICS.csv"
    )
    p_target_macro = (
        OUT_DIR
        / "B6_P01A_PROMISE12_MACRO_METRICS.csv"
    )

    source_cells.to_csv(
        p_source_cells,
        index=False,
        encoding="utf-8",
    )
    target_cells.to_csv(
        p_target_cells,
        index=False,
        encoding="utf-8",
    )
    source_macro.to_csv(
        p_source_macro,
        index=False,
        encoding="utf-8",
    )
    target_macro.to_csv(
        p_target_macro,
        index=False,
        encoding="utf-8",
    )

    print("\nPROMISE12 matched-representation macro metrics")
    print(
        target_macro.to_string(
            index=False,
        )
    )

    print("\n[4/4] Paired PROMISE12 patient-cluster bootstrap")
    reps, deltas = bootstrap_paired_deltas(
        target_scores,
    )

    delta_summary = summarize_deltas(
        deltas,
        target_macro,
    )

    p_reps = (
        OUT_DIR
        / "B6_P01A_PATIENT_BOOTSTRAP_REPLICATES.csv"
    )
    p_delta_reps = (
        OUT_DIR
        / "B6_P01A_PAIRED_DELTA_REPLICATES.csv"
    )
    p_delta_summary = (
        OUT_DIR
        / "B6_P01A_PAIRED_DELTAS.csv"
    )

    reps.to_csv(
        p_reps,
        index=False,
        encoding="utf-8",
    )
    deltas.to_csv(
        p_delta_reps,
        index=False,
        encoding="utf-8",
    )
    delta_summary.to_csv(
        p_delta_summary,
        index=False,
        encoding="utf-8",
    )

    print("\nPaired full-minus-comparator deltas")
    print(
        delta_summary.to_string(
            index=False,
        )
    )

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "scientific_status": (
            "post-freeze/post-reveal matched-control attribution"
        ),
        "protocol": {
            "path": str(
                Path(__file__).resolve().parent
                / PROTOCOL_NAME
            ),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "upstream": {
            "FinalB1_lock_sha256": EXPECTED_B1_SHA,
            "FinalB2_lock_sha256": EXPECTED_B2_SHA,
            "FinalB3_lock_sha256": EXPECTED_B3_SHA,
            "P0_4_fix1_gate": EXPECTED_P04_GATE,
        },
        "design": {
            "train": "Prostate158 other-two actions",
            "test": "PROMISE12 held-out action",
            "target_fit_rows": 0,
            "test_cluster_unit": "patient/case_key",
            "reliability_change_baseline": "NOT_INCLUDED_NOT_RECOVERABLE",
            "simple_mask_change": [
                "prediction Dice",
                "prediction IoU",
                "signed foreground fraction delta",
                "signed boundary density delta",
            ],
            "full_transition_score": (
                "frozen Final-B3; not refit"
            ),
        },
        "model_records": model_records,
        "outputs": {},
    }

    output_files = [
        p_source_scores,
        p_target_scores,
        p_source_cells,
        p_target_cells,
        p_source_macro,
        p_target_macro,
        p_reps,
        p_delta_reps,
        p_delta_summary,
    ]

    for p in output_files:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01A_AUDIT.json"
    write_json(
        p_audit,
        audit,
    )

    macro_map = {
        r["representation"]: r
        for r in target_macro.to_dict(
            orient="records"
        )
    }

    report_lines = [
        "=" * 152,
        "SafeTTA B6-P0-1A MRI MATCHED ATTRIBUTION COMPLETE",
    ]

    for rep in REPRESENTATIONS:
        r = macro_map[rep]
        report_lines.append(
            f"{rep:36s} "
            f"AUROC={r['macro_action_family_auroc']:.6f} "
            f"AUPRC={r['macro_action_family_auprc']:.6f} "
            f"prev={r['macro_action_family_prevalence']:.6f}"
        )

    report_lines.extend([
        "",
        "PAIRED FULL - COMPARATOR:",
        delta_summary.to_string(index=False),
        "",
        (
            "Execution PASS does not imply semantic transition superiority; "
            "negative or null paired deltas remain valid audit outcomes."
        ),
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 152,
        "",
    ])

    report = "\n".join(
        report_lines
    )

    (
        OUT_DIR
        / "B6_P01A_REPORT.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
