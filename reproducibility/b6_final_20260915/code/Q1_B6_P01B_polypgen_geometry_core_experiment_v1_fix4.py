#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — PolypGen unseen-MEMO geometry-core candidate-conditioning experiment.

Fix4 additionally binds NeoPolyp candidate masks from the pre-existing R38A1A historical recovery decision (not R38 replay output) and PolypGen MEMO masks from the R33A2B pre-HARM lock. No candidate-mask version is chosen by P01B outcome.\n\nThis is the first scientific P01B experiment after:
- source-feature cache loss was established;
- NeoPolyp SOURCE-state 66-D was exactly reconstructed;
- the simple four-feature candidate mask-change descriptor had already been frozen.

Primary matched comparison:
    SOURCE_STATE
vs
    SOURCE_STATE + SIMPLE_MASK_CHANGE

Training:
    NeoPolyp TENT1 + PL-CONF90 only.
Target:
    PolypGen unseen MEMO.
Target fit/calibration:
    ZERO.

The script aggressively binds assets by explicit row keys and expected panel
cardinality before fitting. If exact binding is not unique, it fails BEFORE
scientific metrics rather than guessing.

Optional:
    an already-frozen R33 full semantic-transition score is reported only if
    an exact key match is found. It is not refit here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-GEOM-CORE-v1-fix4"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_NAME = "B6_P01B_GEOMETRY_CORE_AMENDMENT_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "ad36a75760e36112ee373131e0212ad9819128c26aef24db4ec5a91c9a97095a"

PARENT_PROTOCOL = CODE / "B6_P01B_POLYPGEN_MATCHED_CANDIDATE_CONDITIONING_PROTOCOL_v1.json"
EXPECTED_PARENT_PROTOCOL_SHA256 = "64ad2ac72ff6fc04ba07a6492e27c6b1af9a537fd0a49e787b5ae2d0b9ea62f9"

FIRST_BIND_DIR = ROOT / "B6_P01B_polypgen_asset_binding_audit_v1"
FIRST_BIND_AUDIT = FIRST_BIND_DIR / "B6_P01B_ASSET_BINDING_AUDIT.json"
FIRST_INVENTORY = FIRST_BIND_DIR / "B6_P01B_ASSET_INVENTORY.csv"

SOURCE66_DIR = ROOT / "B6_P01B_reconstruct_neopolyp_source_state66_v1"
SOURCE66_AUDIT = SOURCE66_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66_RECONSTRUCTION_AUDIT.json"
SOURCE66_NPY = SOURCE66_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66.npy"
SOURCE66_ROWS = SOURCE66_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66_ROWS.csv"

EXPECTED_BIND_GATE = "PASS_B6_P01B_ASSET_BINDING_AUDIT_COMPLETE"
EXPECTED_SOURCE66_GATE = "PASS_B6_P01B_NEOPOLYP_SOURCE_STATE66_EXACT_RECONSTRUCTION"

OUT_DIR = ROOT / "B6_P01B_polypgen_geometry_core_experiment_v1_fix4"
PASS_GATE = "PASS_B6_P01B_POLYPGEN_GEOMETRY_CORE_EXPERIMENT_COMPLETE"

H = 352
W = 352
PIXELS = H * W
PACKED_BYTES = 15488
BITORDER = "little"

SOURCE_IMAGES = 800
SOURCE_STATES = 9
SOURCE_ROWS_PER_ACTION = 7200
SOURCE_TOTAL_ROWS = 14400

TARGET_IMAGES = 1532
TARGET_STATES = 3
TARGET_ROWS = 4596

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260918

TRAIN_ACTIONS = ["TENT1", "PL-CONF90"]
TARGET_ACTION = "MEMO"

# Exact authoritative PolypGen unseen-MEMO common-support assets recovered
# by B6_P01B_polypgen_target_outcome_schema_audit_v1.
R33A3_TARGET_PANEL = (
    ROOT
    / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MATCHED_SCORE_OUTCOME_PANEL.csv"
)
R33A2B_FEATURE_PRE_HARM = (
    ROOT
    / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
    / "R33A2B_POLYPGEN_MEMO_Q66_DSEM64_FEATURE_PRE_HARM_LOCK.csv"
)
R33A2B_SCORE_PRE_HARM = (
    ROOT
    / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
    / "R33A2B_POLYPGEN_MEMO_EXTERNAL_SAFETTA_SCORE_PRE_HARM_LOCK.csv"
)

# P01B mask adjudication: use the PRE-EXISTING historical recovery decision,
# not replay assets selected by downstream performance.
R38A1A_ROOT = (
    ROOT
    / "R38A1A_historical_candidate_mask_recovery_and_parity_localization_v1"
)
R38A1A_FINAL_LOCK = R38A1A_ROOT / "R38A1A_FINAL_LOCK.json"
R38A1A_HISTORICAL_MANIFEST = (
    R38A1A_ROOT / "R38A1A_HISTORICAL_MASK_ASSET_MANIFEST.csv"
)
EXPECTED_R38A1A_MANIFEST_SHA256 = (
    "a8789346d2c4a42bd5324612a6de31408088e6a6a2345b194726b96fda768471"
)
EXPECTED_R38A1A_STATUS = "PASS_R38A1A_HISTORICAL_CANDIDATE_MASKS_RECOVERED"
EXPECTED_R38A1A_DECISION = (
    "REBUILD_R38A1_WITH_HISTORICAL_TENT_PL_MASKS__"
    "USE_FROZEN_R22A1_S64_AS_BASELINE__"
    "DO_NOT_REQUIRE_NUMERIC_REINFERENCE_IDENTITY"
)
R38A1A_MASK_ROOT = R38A1A_ROOT / "historical_common800_masks"

R33A2B_MASK_ROOT = (
    ROOT
    / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"
)
FINALV2B_MATCHED_PANEL = (
    ROOT
    / "FinalV2B_polypgen_s64_external_outcome_evaluation_v1"
    / "FinalV2B_MATCHED_PANEL.csv"
)

Q66_COLS = [f"qsrc_{i:02d}" for i in range(66)]

KEY_COLS = ["sample_id", "model_state_id"]

SAMPLE_ALIASES = [
    "sample_id", "external_case_id", "image_id", "image_key", "case_key",
    "physical_image_id", "physical_case_id", "frame_id", "case_id",
]
STATE_ALIASES = [
    "model_state_id", "state_id", "model_state", "state",
    "model_state_key", "state_key", "model_case_state",
]
ACTION_ALIASES = [
    "action", "action_name", "tta_action", "adaptation_action", "method",
    "adaptation_method", "tta_method", "action_id", "heldout_action",
    "candidate_action", "update_method",
]
HARM_ALIASES = [
    "harm_label", "harmful", "harm", "is_harm", "future_harm",
    "harm_flag", "y_harm", "memo_harm_label",
]
DELTA_ALIASES = [
    "delta_dice", "dice_delta", "delta", "adaptation_delta_dice",
    "candidate_minus_source_dice", "deltaDice", "memo_delta_dice",
]
SCORE_ALIASES = [
    "risk_score", "frozen_safety_probability", "safety_probability",
    "harm_probability", "score",
]
ROW_INDEX_ALIASES = [
    "state_row_index", "row_index", "global_index", "index",
]
NPZ_REF_ALIASES = [
    "state_prediction_npz", "prediction_npz", "npz_path", "mask_npz",
    "prediction_path", "artifact_path",
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", OUT_DIR.name}

MAX_TABLE_BYTES = 250_000_000


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
    if got.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}")
    return got


def verify_lineage(script_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    protocol_path = script_dir / PROTOCOL_NAME
    exact_sha(protocol_path, EXPECTED_PROTOCOL_SHA256, "P01B geometry protocol")
    exact_sha(PARENT_PROTOCOL, EXPECTED_PARENT_PROTOCOL_SHA256, "P01B parent protocol")

    protocol = load_json(protocol_path)
    if protocol.get("status") != "FROZEN_AFTER_ASSET_LOSS_AUDIT_BEFORE_P01B_SCIENTIFIC_METRICS":
        raise RuntimeError("Geometry protocol status changed.")

    bind = load_json(FIRST_BIND_AUDIT)
    if bind.get("status") != "PASS" or bind.get("gate") != EXPECTED_BIND_GATE:
        raise RuntimeError("First P01B binding gate changed.")

    src = load_json(SOURCE66_AUDIT)
    if src.get("status") != "PASS" or src.get("gate") != EXPECTED_SOURCE66_GATE:
        raise RuntimeError("NeoPolyp SOURCE66 reconstruction gate changed.")

    src_meta = src["outputs"][SOURCE66_NPY.name]
    exact_sha(SOURCE66_NPY, src_meta["sha256"], "NeoPolyp SOURCE66")
    exact_sha(
        SOURCE66_ROWS,
        src["outputs"][SOURCE66_ROWS.name]["sha256"],
        "NeoPolyp SOURCE66 row metadata",
    )

    inv = pd.read_csv(FIRST_INVENTORY, low_memory=False)
    if len(inv) == 0:
        raise RuntimeError("P01B inventory is empty.")

    return protocol, inv


def first_existing(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    for a in aliases:
        if a in columns:
            return a
    return None


def first_existing_ci(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """Case-insensitive exact alias resolver; returns the original column name."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for a in aliases:
        hit = lookup.get(str(a).strip().lower())
        if hit is not None:
            return hit
    return None


def normalize_action_value(x: Any) -> str:
    s = str(x).strip().lower().replace("_", "-").replace(" ", "-")
    if "memo" in s:
        return "MEMO"
    if "tent" in s or s in {"a1", "tent1"}:
        return "TENT1"
    if "conf90" in s or "pl-conf90" in s or "plconf90" in s:
        return "PL-CONF90"
    if s.startswith("pl") or "pseudo" in s:
        return "PL-CONF90"
    return ""


def infer_action_from_path(path: Path) -> str:
    return normalize_action_value(str(path))


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def load_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file() or path.stat().st_size > MAX_TABLE_BYTES:
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
                return pd.DataFrame(obj)
        if path.suffix.lower() == ".jsonl":
            return pd.read_json(path, lines=True)
    except Exception:
        return None
    return None


def standardize_outcome_table(path: Path, df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, Dict[str, Any]]]:
    """
    Robustly standardize historical action-outcome tables.

    Fix2 scientific-preserving binding repair:
    historical R31/R32 OOF tables can repeat the SAME physical
    (sample_id, model_state_id, action) outcome across split seeds,
    held-out folds, representation variants, or prediction rows.

    The HARM outcome is an outcome property of that model-case/action,
    not of the OOF prediction row. Therefore we collapse repeated rows
    ONLY when every duplicate exact key has one and the same HARM label.

    If any duplicate exact key disagrees on HARM, binding fails rather
    than choosing a row.

    This changes no HARM definition, action support, or scientific sample.
    """
    cols = list(df.columns)
    sc = first_existing_ci(cols, SAMPLE_ALIASES)
    stc = first_existing_ci(cols, STATE_ALIASES)
    hc = first_existing_ci(cols, HARM_ALIASES)
    dc = first_existing_ci(cols, DELTA_ALIASES)
    ac = first_existing_ci(cols, ACTION_ALIASES)

    if sc is None or stc is None or (hc is None and dc is None):
        return []

    sample = norm_id(df[sc])
    state = df[stc].astype(str).str.strip()

    if hc is not None:
        harm_num = pd.to_numeric(df[hc], errors="coerce")
        if harm_num.isna().any():
            htxt = df[hc].astype(str).str.strip().str.upper()
            mapping = {
                "HARM": 1, "HARMFUL": 1, "TRUE": 1, "YES": 1,
                "NON-HARM": 0, "NON_HARM": 0, "SAFE": 0,
                "FALSE": 0, "NO": 0, "NEUTRAL": 0, "BENEFIT": 0,
            }
            mapped = htxt.map(mapping)
            if mapped.isna().any():
                return []
            harm = mapped.astype(int)
            harm_source = hc
        else:
            harm = harm_num.astype(int)
            harm_source = hc
    else:
        delta = pd.to_numeric(df[dc], errors="coerce")
        if delta.isna().any():
            return []
        harm = (delta <= -0.02).astype(int)
        harm_source = f"{dc}<=-0.02"

    if not set(harm.unique()).issubset({0, 1}):
        return []

    base = pd.DataFrame({
        "sample_id": sample,
        "model_state_id": state,
        "harm_label": harm,
    })

    if ac is not None:
        actions = df[ac].map(normalize_action_value)
        action_source = ac
        inferred = infer_action_from_path(path)
        if inferred:
            actions = actions.mask(actions == "", inferred)
    else:
        inferred = infer_action_from_path(path)
        actions = pd.Series([inferred] * len(df), index=df.index)
        action_source = "path"

    result = []
    for action in sorted(set(actions) - {""}):
        mask = (actions == action).to_numpy()
        raw = base.loc[mask].copy().reset_index(drop=True)
        if len(raw) == 0:
            continue

        # Exact-key HARM consistency gate.
        harm_nunique = (
            raw.groupby(KEY_COLS, sort=False)["harm_label"]
            .nunique(dropna=False)
        )
        conflicts = harm_nunique[harm_nunique != 1]
        if len(conflicts):
            # Return no candidate from this action/file; choosing among
            # inconsistent labels would alter the science.
            continue

        collapsed = (
            raw.groupby(KEY_COLS, sort=False, as_index=False)["harm_label"]
            .first()
        )

        multiplicity = (
            raw.groupby(KEY_COLS, sort=False)
            .size()
            .to_numpy(dtype=np.int64)
        )

        meta = {
            "path": str(path),
            "sha256": sha256_file(path),
            "action_source": action_source,
            "harm_source": harm_source,
            "raw_rows": int(len(raw)),
            "rows": int(len(collapsed)),
            "samples": int(collapsed["sample_id"].nunique()),
            "states": int(collapsed["model_state_id"].nunique()),
            "unique_keys": int(collapsed[KEY_COLS].drop_duplicates().shape[0]),
            "duplicate_collapse_performed": bool(len(raw) != len(collapsed)),
            "duplicate_multiplicity_min": int(multiplicity.min()) if len(multiplicity) else 0,
            "duplicate_multiplicity_median": float(np.median(multiplicity)) if len(multiplicity) else 0.0,
            "duplicate_multiplicity_max": int(multiplicity.max()) if len(multiplicity) else 0,
            "harm_conflicting_keys": 0,
            "columns": ";".join(map(str, cols)),
        }
        result.append((action, collapsed, meta))
    return result

def all_outcome_candidate_paths(inv: pd.DataFrame) -> List[Path]:
    """
    Fix1 no longer trusts the coarse cohort label in the first heuristic
    inventory. R31/R32 output directories may not contain the literal word
    'NeoPolyp'. We scan every inventoried tabular candidate that is either
    outcome-like/keyed or whose path carries R31/R32/R33/action provenance.
    """
    out: List[Path] = []
    seen = set()

    for _, r in inv.iterrows():
        p = Path(str(r.get("path", "")))
        if not p.is_file() or p.suffix.lower() not in {".csv", ".tsv", ".json", ".jsonl"}:
            continue

        roles = str(r.get("roles", ""))
        low = str(p).lower()
        relevant = (
            "OUTCOME_TABLE_CANDIDATE" in roles
            or "KEYED_TABLE" in roles
            or any(x in low for x in [
                "r31", "r32", "r33", "neopolyp", "polypgen",
                "tent", "conf90", "memo", "outcome", "harm",
            ])
        )
        if not relevant:
            continue

        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)

    return out


def source66_support_set() -> set[tuple]:
    rows = pd.read_csv(SOURCE66_ROWS, low_memory=False)
    sc = first_existing_ci(rows.columns, SAMPLE_ALIASES)
    stc = first_existing_ci(rows.columns, STATE_ALIASES)
    if sc is None or stc is None:
        raise RuntimeError("SOURCE66 row metadata missing sample/state keys.")
    x = pd.DataFrame({
        "sample_id": norm_id(rows[sc]),
        "model_state_id": rows[stc].astype(str).str.strip(),
    })
    if x[KEY_COLS].drop_duplicates().shape[0] != len(x):
        raise RuntimeError("SOURCE66 support keys are not unique.")
    return set(map(tuple, x[KEY_COLS].to_numpy()))

def inventory_paths(inv: pd.DataFrame, cohort: str, role_substr: str) -> List[Path]:
    sub = inv[
        (inv["cohort"].fillna("").astype(str) == cohort)
        & inv["roles"].fillna("").astype(str).str.contains(role_substr, regex=False)
    ]
    out = []
    for x in sub["path"].dropna().astype(str).unique():
        p = Path(x)
        if p.is_file():
            out.append(p)
    return out


def choose_source_outcomes(inv: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fix1 exact source binder.

    The scientific support remains frozen at 800 images x 9 states per action.
    The fix only broadens historical schema/path discovery and requires that
    every selected source key is covered by the already-reconstructed 9000-row
    SOURCE66 panel.
    """
    paths = all_outcome_candidate_paths(inv)
    src66_keys = source66_support_set()

    candidates: Dict[str, List[Tuple[pd.DataFrame, Dict[str, Any]]]] = {
        "TENT1": [],
        "PL-CONF90": [],
    }
    diagnostics = []

    for p in paths:
        df = load_table(p)
        if df is None:
            continue

        standardized = standardize_outcome_table(p, df)
        for action, sub, meta in standardized:
            if action not in candidates:
                continue

            keys = set(map(tuple, sub[KEY_COLS].to_numpy()))
            source66_covered = keys.issubset(src66_keys)

            exact = (
                len(sub) == SOURCE_ROWS_PER_ACTION
                and sub["sample_id"].nunique() == SOURCE_IMAGES
                and sub["model_state_id"].nunique() == SOURCE_STATES
                and sub[KEY_COLS].drop_duplicates().shape[0] == SOURCE_ROWS_PER_ACTION
                and source66_covered
            )

            low = str(p).lower()
            path_score = (
                int("r32" in low) * 20
                + int("r31" in low) * 15
                + int("neopolyp" in low) * 10
                + int(action.lower().replace("-", "") in low.replace("-", "").replace("_", "")) * 3
            )
            meta["source66_covered"] = bool(source66_covered)
            meta["exact_expected_panel"] = bool(exact)
            meta["path_score"] = int(path_score)
            meta["selection_score"] = int((1000 if exact else 0) + path_score)
            candidates[action].append((sub, meta))
            diagnostics.append({"action": action, **meta})

    # Print diagnostic candidates before any failure. This makes any remaining
    # historical-schema mismatch actionable without changing the protocol.
    diag = pd.DataFrame(diagnostics)
    if len(diag):
        show_cols = [
            "action", "raw_rows", "rows", "samples", "states", "unique_keys",
            "duplicate_collapse_performed",
            "duplicate_multiplicity_min", "duplicate_multiplicity_median",
            "duplicate_multiplicity_max",
            "source66_covered", "exact_expected_panel", "selection_score",
            "harm_source", "action_source", "path",
        ]
        print("\nSOURCE outcome binding candidates (top 30):")
        print(
            diag.sort_values(
                ["exact_expected_panel", "selection_score", "rows"],
                ascending=[False, False, False],
            )[show_cols].head(30).to_string(index=False)
        )
    else:
        print("\nSOURCE outcome binding candidates: NONE PARSEABLE")

    chosen: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}
    for action in TRAIN_ACTIONS:
        xs = sorted(
            candidates[action],
            key=lambda t: t[1]["selection_score"],
            reverse=True,
        )
        exacts = [x for x in xs if x[1]["exact_expected_panel"]]
        if not exacts:
            raise RuntimeError(
                f"SOURCE_OUTCOME_BINDING_FAILED_FIX1 {action}: no exact "
                f"{SOURCE_ROWS_PER_ACTION}-row/{SOURCE_IMAGES}-image/"
                f"{SOURCE_STATES}-state candidate covered by SOURCE66. "
                "See the printed candidate table; do not change the protocol."
            )

        # If multiple exact candidates exist, they must have identical keys and
        # identical HARM labels; otherwise fail rather than choose by score.
        ref_sub, ref_meta = exacts[0]
        ref = ref_sub.sort_values(KEY_COLS).reset_index(drop=True)
        for other_sub, other_meta in exacts[1:]:
            oth = other_sub.sort_values(KEY_COLS).reset_index(drop=True)
            same_keys = ref[KEY_COLS].equals(oth[KEY_COLS])
            same_harm = np.array_equal(
                ref["harm_label"].to_numpy(dtype=int),
                oth["harm_label"].to_numpy(dtype=int),
            )
            if not (same_keys and same_harm):
                raise RuntimeError(
                    f"SOURCE_OUTCOME_BINDING_AMBIGUOUS_FIX1 {action}: "
                    "multiple exact candidates disagree."
                )
        chosen[action] = exacts[0]

    k0 = set(map(tuple, chosen["TENT1"][0][KEY_COLS].to_numpy()))
    k1 = set(map(tuple, chosen["PL-CONF90"][0][KEY_COLS].to_numpy()))
    if k0 != k1:
        raise RuntimeError(
            "SOURCE_OUTCOME_BINDING_FAILED_FIX1: TENT1 and PL-CONF90 "
            "do not use the same frozen image/state support."
        )

    parts = []
    meta_out = {}
    for action in TRAIN_ACTIONS:
        sub, meta = chosen[action]
        x = sub.copy()
        x["action"] = action
        parts.append(x)
        meta_out[action] = meta

    out = pd.concat(parts, ignore_index=True)
    if len(out) != SOURCE_TOTAL_ROWS:
        raise RuntimeError("SOURCE outcome total rows changed.")
    return out, meta_out

def choose_target_outcomes(inv: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fix3: bind the exact authoritative R33A3 unseen-MEMO outcome panel.

    This removes the remaining heuristic ambiguity after the target-schema
    audit showed:
      - 4596 rows
      - 1532 physical images
      - 3 DeepLabV3-R50 states
      - action MEMO-SEG4-1STEP
      - memo_harm_label / memo_delta_dice

    The stored memo_harm_label must exactly reproduce the frozen rule
    memo_delta_dice <= -0.02.
    """
    p = R33A3_TARGET_PANEL
    if not p.is_file():
        raise FileNotFoundError(
            f"Authoritative R33A3 target panel not found: {p}"
        )

    df = pd.read_csv(p, low_memory=False)
    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "action",
        "memo_delta_dice",
        "memo_harm_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"R33A3 target panel missing columns: {missing}"
        )

    if len(df) != TARGET_ROWS:
        raise RuntimeError(
            f"R33A3 target rows={len(df)} != {TARGET_ROWS}"
        )

    if df["sample_id"].astype(str).nunique() != TARGET_IMAGES:
        raise RuntimeError("R33A3 physical-image count changed.")
    if df["model_state_id"].astype(str).nunique() != TARGET_STATES:
        raise RuntimeError("R33A3 target state count changed.")
    if set(df["model_family"].astype(str)) != {"DeepLabV3-R50"}:
        raise RuntimeError("R33A3 target family changed from DeepLabV3-R50.")

    actions = df["action"].map(normalize_action_value)
    if set(actions) != {"MEMO"}:
        raise RuntimeError(
            f"R33A3 target action changed: {sorted(set(actions))}"
        )

    delta = pd.to_numeric(
        df["memo_delta_dice"], errors="raise"
    ).to_numpy(dtype=np.float64)
    stored_harm = pd.to_numeric(
        df["memo_harm_label"], errors="raise"
    ).to_numpy(dtype=np.int64)
    expected_harm = (delta <= -0.02).astype(np.int64)

    if not np.array_equal(stored_harm, expected_harm):
        raise RuntimeError(
            "R33A3 memo_harm_label no longer reproduces delta<=-0.02."
        )

    out = pd.DataFrame({
        "sample_id": norm_id(df["sample_id"]),
        "model_state_id": df["model_state_id"].astype(str).str.strip(),
        "harm_label": stored_harm,
        "action": "MEMO",
    })

    if out[KEY_COLS].drop_duplicates().shape[0] != TARGET_ROWS:
        raise RuntimeError("R33A3 target sample/state key not unique.")

    # Optional independent historical consistency sentinel.
    finalv2b_consistency = "NOT_AVAILABLE"
    if FINALV2B_MATCHED_PANEL.is_file():
        v2 = pd.read_csv(FINALV2B_MATCHED_PANEL, low_memory=False)
        need2 = {
            "sample_id", "model_state_id",
            "memo_delta_dice", "memo_harm_label"
        }
        if need2.issubset(v2.columns) and len(v2) == TARGET_ROWS:
            vv = pd.DataFrame({
                "sample_id": norm_id(v2["sample_id"]),
                "model_state_id": v2["model_state_id"].astype(str).str.strip(),
                "memo_harm_label_v2": pd.to_numeric(
                    v2["memo_harm_label"], errors="raise"
                ).astype(int),
                "memo_delta_dice_v2": pd.to_numeric(
                    v2["memo_delta_dice"], errors="raise"
                ).astype(float),
            })
            chk = out.merge(
                vv,
                on=KEY_COLS,
                how="left",
                validate="one_to_one",
            )
            if chk["memo_harm_label_v2"].isna().any():
                raise RuntimeError(
                    "FinalV2B does not cover exact R33A3 target keys."
                )
            if not np.array_equal(
                chk["harm_label"].to_numpy(dtype=int),
                chk["memo_harm_label_v2"].to_numpy(dtype=int),
            ):
                raise RuntimeError(
                    "FinalV2B and R33A3 disagree on MEMO HARM labels."
                )
            finalv2b_consistency = "PASS"

    meta = {
        "path": str(p),
        "sha256": sha256_file(p),
        "rows": TARGET_ROWS,
        "samples": TARGET_IMAGES,
        "states": TARGET_STATES,
        "family": "DeepLabV3-R50",
        "action": "MEMO-SEG4-1STEP",
        "harm_source": "memo_harm_label",
        "delta_source": "memo_delta_dice",
        "harm_rule_reproduction": "PASS",
        "finalv2b_consistency": finalv2b_consistency,
    }

    print("\nAUTHORITATIVE TARGET OUTCOME BINDING:")
    print(" path   =", p)
    print(" rows   =", len(out))
    print(" images =", out["sample_id"].nunique())
    print(" states =", out["model_state_id"].nunique())
    print(" harm_n =", int(out["harm_label"].sum()))
    print(" prev   =", float(out["harm_label"].mean()))
    print(" R33A3 HARM rule reproduction = PASS")
    print(" FinalV2B consistency =", finalv2b_consistency)

    return out.reset_index(drop=True), meta

def load_source66(source_outcomes: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    feat = np.load(SOURCE66_NPY, mmap_mode="r")
    rows = pd.read_csv(SOURCE66_ROWS, low_memory=False)

    if feat.shape != (9000, 66):
        raise RuntimeError(f"Reconstructed SOURCE66 shape changed: {feat.shape}")
    if len(rows) != 9000:
        raise RuntimeError("Reconstructed SOURCE66 row metadata changed.")

    sc = first_existing(rows.columns, SAMPLE_ALIASES)
    stc = first_existing(rows.columns, STATE_ALIASES)
    if not sc or not stc:
        raise RuntimeError("SOURCE66 metadata missing sample/state key.")

    rows = rows.copy()
    rows["sample_id"] = norm_id(rows[sc])
    rows["model_state_id"] = rows[stc].astype(str).str.strip()
    rows["_feature_row"] = np.arange(len(rows), dtype=np.int64)

    if rows[KEY_COLS].drop_duplicates().shape[0] != len(rows):
        raise RuntimeError("SOURCE66 metadata key not unique.")

    support = source_outcomes[KEY_COLS].drop_duplicates()
    joined = support.merge(rows[KEY_COLS + ["_feature_row"]], on=KEY_COLS, how="left", validate="one_to_one")
    if joined["_feature_row"].isna().any():
        raise RuntimeError("SOURCE66 does not cover the frozen source action support.")

    idx = joined["_feature_row"].to_numpy(dtype=np.int64)
    x = np.asarray(feat[idx], dtype=np.float64)
    if x.shape != (SOURCE_ROWS_PER_ACTION, 66):
        raise RuntimeError(f"SOURCE66 support shape unexpected: {x.shape}")

    key_df = joined[KEY_COLS].reset_index(drop=True)
    return x, key_df, {
        "feature_path": str(SOURCE66_NPY),
        "feature_sha256": sha256_file(SOURCE66_NPY),
        "row_path": str(SOURCE66_ROWS),
        "row_sha256": sha256_file(SOURCE66_ROWS),
    }


def candidate_source_state_paths(inv: pd.DataFrame) -> List[Path]:
    paths = inventory_paths(inv, "PolypGen", "SOURCE_STATE_FEATURE_CANDIDATE")
    # Also admit dimension-labelled array candidates if present in inventory.
    extra = inv[
        (inv["cohort"].fillna("").astype(str) == "PolypGen")
        & inv["roles"].fillna("").astype(str).str.contains("DIM66_ARRAY", regex=False)
    ]
    for x in extra["path"].dropna().astype(str).unique():
        p = Path(x)
        if p.is_file() and p not in paths:
            paths.append(p)
    return paths


def find_key_metadata_for_feature(path: Path, n: int) -> List[pd.DataFrame]:
    cands = []
    dirs = [path.parent, path.parent.parent]
    seen = set()

    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.csv"):
            if p in seen:
                continue
            seen.add(p)
            try:
                h = pd.read_csv(p, nrows=0, low_memory=False)
            except Exception:
                continue
            sc = first_existing(h.columns, SAMPLE_ALIASES)
            stc = first_existing(h.columns, STATE_ALIASES)
            if not sc or not stc:
                continue
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue
            if len(df) != n:
                continue
            out = pd.DataFrame({
                "sample_id": norm_id(df[sc]),
                "model_state_id": df[stc].astype(str).str.strip(),
                "_feature_row": np.arange(n, dtype=np.int64),
                "_metadata_path": str(p),
            })
            if out[KEY_COLS].drop_duplicates().shape[0] == n:
                cands.append(out)
    return cands


def bind_feature_array(path: Path, target_keys: pd.DataFrame) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    results = []
    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r")
            arrays = [("", a)]
        elif path.suffix.lower() == ".npz":
            z = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays = [(k, z[k]) for k in z.files if getattr(z[k], "ndim", 0) == 2 and z[k].shape[1] == 66]
        else:
            return []
    except Exception:
        return []

    for member, a in arrays:
        if a.ndim != 2 or a.shape[1] != 66:
            continue
        n = int(a.shape[0])

        # Strategy A: internal keys in NPZ.
        internal_meta = []
        if path.suffix.lower() == ".npz":
            try:
                z = np.load(path, mmap_mode="r", allow_pickle=False)
                sc = next((x for x in SAMPLE_ALIASES if x in z.files), None)
                stc = next((x for x in STATE_ALIASES if x in z.files), None)
                if sc and stc and len(z[sc]) == n and len(z[stc]) == n:
                    im = pd.DataFrame({
                        "sample_id": pd.Series(z[sc].astype(str)).str.strip().str.lower(),
                        "model_state_id": pd.Series(z[stc].astype(str)).str.strip(),
                        "_feature_row": np.arange(n, dtype=np.int64),
                        "_metadata_path": f"{path}::{sc},{stc}",
                    })
                    internal_meta.append(im)
            except Exception:
                pass

        metas = internal_meta + find_key_metadata_for_feature(path, n)

        for meta in metas:
            joined = target_keys.merge(meta, on=KEY_COLS, how="left", validate="one_to_one")
            if joined["_feature_row"].isna().any():
                continue
            idx = joined["_feature_row"].to_numpy(dtype=np.int64)
            x = np.asarray(a[idx], dtype=np.float64)
            if x.shape != (len(target_keys), 66) or not np.isfinite(x).all():
                continue
            results.append((x, {
                "path": str(path),
                "sha256": sha256_file(path),
                "member": member,
                "shape": list(a.shape),
                "metadata_path": str(meta["_metadata_path"].iloc[0]),
            }))
    return results


def choose_target_source66(inv: pd.DataFrame, target_outcomes: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fix3: use the exact pre-HARM R33A2B Q66 feature lock.

    This is stronger than heuristic array discovery because the target-schema
    audit recovered the exact 4596-row table and all qsrc_00..qsrc_65 columns.
    """
    p = R33A2B_FEATURE_PRE_HARM
    if not p.is_file():
        raise FileNotFoundError(
            f"Authoritative R33A2B feature lock not found: {p}"
        )

    df = pd.read_csv(p, low_memory=False)
    required = {
        "sample_id", "model_state_id", "model_family",
        "training_seed", "action", *Q66_COLS
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"R33A2B Q66 feature lock missing columns: {missing[:20]}"
        )

    if len(df) != TARGET_ROWS:
        raise RuntimeError(
            f"R33A2B feature rows={len(df)} != {TARGET_ROWS}"
        )
    if df["sample_id"].astype(str).nunique() != TARGET_IMAGES:
        raise RuntimeError("R33A2B feature image count changed.")
    if df["model_state_id"].astype(str).nunique() != TARGET_STATES:
        raise RuntimeError("R33A2B feature state count changed.")
    if set(df["model_family"].astype(str)) != {"DeepLabV3-R50"}:
        raise RuntimeError("R33A2B feature family changed.")
    if set(df["action"].map(normalize_action_value)) != {"MEMO"}:
        raise RuntimeError("R33A2B feature action changed.")

    feat = pd.DataFrame({
        "sample_id": norm_id(df["sample_id"]),
        "model_state_id": df["model_state_id"].astype(str).str.strip(),
    })
    x = df[Q66_COLS].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)

    if x.shape != (TARGET_ROWS, 66):
        raise RuntimeError(f"R33A2B Q66 shape={x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("R33A2B Q66 contains non-finite values.")
    if feat[KEY_COLS].drop_duplicates().shape[0] != TARGET_ROWS:
        raise RuntimeError("R33A2B Q66 sample/state key not unique.")

    feat["_row"] = np.arange(TARGET_ROWS, dtype=np.int64)
    joined = target_outcomes[KEY_COLS].merge(
        feat,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if joined["_row"].isna().any():
        raise RuntimeError(
            "R33A2B Q66 does not cover exact R33A3 target keys."
        )

    idx = joined["_row"].to_numpy(dtype=np.int64)
    x = x[idx]

    return x, {
        "path": str(p),
        "sha256": sha256_file(p),
        "rows": TARGET_ROWS,
        "dimension": 66,
        "columns": Q66_COLS,
        "pre_harm_lock": True,
    }


def verify_authoritative_mask_adjudication() -> Dict[str, Any]:
    """
    Freeze mask lineage BEFORE any P01B scientific metric.

    NeoPolyp:
      R38A1A previously recovered all 9 historical TENT and PL candidate-mask
      assets and explicitly decided to rebuild with those historical masks.
      Replay-vs-historical mismatch is expected and is NOT a reason to prefer
      replay output.

    PolypGen:
      use the already pre-HARM R33A2B fix1 SOURCE/MEMO lock.
    """
    if not R38A1A_FINAL_LOCK.is_file():
        raise FileNotFoundError(R38A1A_FINAL_LOCK)
    if not R38A1A_HISTORICAL_MANIFEST.is_file():
        raise FileNotFoundError(R38A1A_HISTORICAL_MANIFEST)
    if not R38A1A_MASK_ROOT.is_dir():
        raise FileNotFoundError(R38A1A_MASK_ROOT)
    if not R33A2B_MASK_ROOT.is_dir():
        raise FileNotFoundError(R33A2B_MASK_ROOT)

    lock = load_json(R38A1A_FINAL_LOCK)
    if lock.get("status") != EXPECTED_R38A1A_STATUS:
        raise RuntimeError(
            f"R38A1A historical-mask status changed: {lock.get('status')}"
        )
    if lock.get("decision") != EXPECTED_R38A1A_DECISION:
        raise RuntimeError(
            f"R38A1A historical-mask decision changed: {lock.get('decision')}"
        )

    manifest_sha = sha256_file(R38A1A_HISTORICAL_MANIFEST)
    if manifest_sha.lower() != EXPECTED_R38A1A_MANIFEST_SHA256.lower():
        raise RuntimeError(
            "R38A1A historical-mask manifest SHA changed.\n"
            f"expected={EXPECTED_R38A1A_MANIFEST_SHA256}\n"
            f"observed={manifest_sha}"
        )

    return {
        "neopolyp_historical_lock": {
            "path": str(R38A1A_FINAL_LOCK),
            "sha256": sha256_file(R38A1A_FINAL_LOCK),
            "status": lock.get("status"),
            "decision": lock.get("decision"),
        },
        "neopolyp_historical_manifest": {
            "path": str(R38A1A_HISTORICAL_MANIFEST),
            "sha256": manifest_sha,
        },
        "polypgen_pre_harm_mask_root": str(R33A2B_MASK_ROOT),
        "selection_basis": (
            "pre-existing historical recovery / pre-HARM lock only; "
            "no P01B outcome metric used"
        ),
    }


def split_model_state_id(state_id: str) -> Tuple[str, int]:
    s = str(state_id).strip()
    if "::" not in s:
        raise RuntimeError(f"Unexpected model_state_id={state_id}")
    family, seed = s.rsplit("::", 1)
    if not seed.isdigit():
        raise RuntimeError(f"Unexpected model_state_id seed={state_id}")
    return family, int(seed)


def state_dir_name_r38a1a(state_id: str) -> str:
    family, seed = split_model_state_id(state_id)
    return f"{family}__{seed}"


def exact_index_binding(
    index_csv: Path,
    expected_state_id: str,
    required_state_keys: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
    """
    Bind one packed-mask state file to exact sample IDs.

    Historical index files are authoritative row-order sidecars.
    If no explicit row-index column exists, CSV order is the packed-array row
    order. No outcome column is read or used.
    """
    if not index_csv.is_file():
        raise FileNotFoundError(index_csv)

    idx = pd.read_csv(index_csv, low_memory=False)
    if len(idx) != expected_rows:
        raise RuntimeError(
            f"{index_csv}: rows={len(idx)}, expected={expected_rows}"
        )

    sc = first_existing_ci(idx.columns, SAMPLE_ALIASES)
    if sc is None:
        raise RuntimeError(
            f"{index_csv}: no sample-id column; columns={list(idx.columns)}"
        )

    stc = first_existing_ci(idx.columns, STATE_ALIASES)
    if stc is not None:
        states = idx[stc].astype(str).str.strip()
        if set(states) != {expected_state_id}:
            raise RuntimeError(
                f"{index_csv}: model_state_id mismatch={sorted(set(states))[:10]}"
            )

    ric = first_existing_ci(
        idx.columns,
        [
            "state_row_index", "row_index", "prediction_row",
            "mask_row", "common800_row", "row",
        ],
    )
    if ric is None:
        row_id = np.arange(expected_rows, dtype=np.int64)
        row_source = "csv_order"
    else:
        rr = pd.to_numeric(idx[ric], errors="coerce")
        if rr.isna().any():
            raise RuntimeError(f"{index_csv}: invalid row index column={ric}")
        row_id = rr.astype(np.int64).to_numpy()
        if set(row_id.tolist()) != set(range(expected_rows)):
            raise RuntimeError(
                f"{index_csv}: row index does not form 0..{expected_rows-1}"
            )
        row_source = ric

    meta = pd.DataFrame({
        "sample_id": norm_id(idx[sc]),
        "model_state_id": expected_state_id,
        "_mask_row": row_id,
    })
    if meta[KEY_COLS].drop_duplicates().shape[0] != expected_rows:
        raise RuntimeError(f"{index_csv}: sample/state key not unique.")

    req = required_state_keys[KEY_COLS].copy().reset_index(drop=True)
    merged = req.merge(
        meta,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if merged["_mask_row"].isna().any():
        missing = int(merged["_mask_row"].isna().sum())
        raise RuntimeError(
            f"{index_csv}: missing {missing}/{len(req)} frozen support rows."
        )

    if len(merged) != expected_rows:
        raise RuntimeError(
            f"{index_csv}: required rows={len(merged)}, expected={expected_rows}"
        )

    merged["_mask_row"] = merged["_mask_row"].astype(np.int64)
    merged["_row_source"] = row_source
    return merged


def bind_r38a1a_neopolyp_geometry(
    required: pd.DataFrame,
    action: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if action not in {"TENT1", "PL-CONF90"}:
        raise RuntimeError(f"Unsupported NeoPolyp action={action}")

    candidate_key = (
        "tent1_masks_packed"
        if action == "TENT1"
        else "pl_conf90_masks_packed"
    )
    source_key = "source_masks_packed"

    req = required[KEY_COLS].copy().reset_index(drop=True)
    if len(req) != SOURCE_ROWS_PER_ACTION:
        raise RuntimeError(
            f"NeoPolyp {action}: required rows={len(req)} != {SOURCE_ROWS_PER_ACTION}"
        )
    if req[KEY_COLS].drop_duplicates().shape[0] != len(req):
        raise RuntimeError(f"NeoPolyp {action}: duplicate required keys.")

    parts = []
    assets = []

    states = list(dict.fromkeys(req["model_state_id"].astype(str).tolist()))
    if len(states) != SOURCE_STATES:
        raise RuntimeError(
            f"NeoPolyp {action}: states={len(states)} != {SOURCE_STATES}"
        )

    for state_id in states:
        state_req = req[req["model_state_id"].astype(str) == state_id].copy()
        if len(state_req) != SOURCE_IMAGES:
            raise RuntimeError(
                f"NeoPolyp {action}/{state_id}: rows={len(state_req)} != {SOURCE_IMAGES}"
            )

        d = R38A1A_MASK_ROOT / state_dir_name_r38a1a(state_id)
        npz_path = d / "historical_source_tent_pl_common800.npz"
        index_csv = d / "historical_common800_index.csv"

        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)

        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
        for key in [source_key, candidate_key]:
            if key not in z.files:
                raise RuntimeError(
                    f"{npz_path}: missing key={key}; available={z.files}"
                )
            if z[key].shape != (SOURCE_IMAGES, PACKED_BYTES):
                raise RuntimeError(
                    f"{npz_path}::{key} shape={z[key].shape}, "
                    f"expected={(SOURCE_IMAGES, PACKED_BYTES)}"
                )

        bound = exact_index_binding(
            index_csv,
            state_id,
            state_req,
            SOURCE_IMAGES,
        )
        rr = bound["_mask_row"].to_numpy(dtype=np.int64)

        geom = geometry_from_rows(
            z[source_key],
            z[candidate_key],
            rr,
        )

        part = bound[KEY_COLS].copy()
        part[[
            "pred_dice", "pred_iou", "area_delta", "boundary_delta"
        ]] = geom
        parts.append(part)

        assets.append({
            "model_state_id": state_id,
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "index_path": str(index_csv),
            "index_sha256": sha256_file(index_csv),
            "source_key": source_key,
            "candidate_key": candidate_key,
            "rows": SOURCE_IMAGES,
        })

    allg = pd.concat(parts, ignore_index=True)
    geom_cols = [
        "pred_dice", "pred_iou", "area_delta", "boundary_delta"
    ]

    merged = req.merge(
        allg,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if merged[geom_cols].isna().any().any():
        raise RuntimeError(f"NeoPolyp {action}: geometry incomplete.")

    return merged[geom_cols].to_numpy(dtype=np.float64), {
        "lineage": "R38A1A historical candidate-mask recovery",
        "historical_lock_path": str(R38A1A_FINAL_LOCK),
        "historical_lock_sha256": sha256_file(R38A1A_FINAL_LOCK),
        "historical_manifest_path": str(R38A1A_HISTORICAL_MANIFEST),
        "historical_manifest_sha256": sha256_file(R38A1A_HISTORICAL_MANIFEST),
        "action": action,
        "rows": len(merged),
        "states": len(states),
        "assets": assets,
    }


def find_r33_state_lock_json(idx: int) -> Path:
    candidates = [
        R33A2B_MASK_ROOT
        / f"state_{idx:02d}_polypgen_source_memo_pre_harm.lock.json",
        R33A2B_MASK_ROOT
        / f"state_{idx:02d}_polypgen_source_memo_pre_harm_lock.json",
    ]
    for p in candidates:
        if p.is_file():
            return p

    hits = sorted(R33A2B_MASK_ROOT.glob(f"state_{idx:02d}*.json"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"No state_{idx:02d} lock JSON in {R33A2B_MASK_ROOT}"
        )
    raise RuntimeError(
        f"Ambiguous state_{idx:02d} lock JSONs: {[str(x) for x in hits]}"
    )


def extract_r33_model_state_id(lock_path: Path) -> str:
    txt = lock_path.read_text(encoding="utf-8", errors="ignore")
    hits = sorted(set(re.findall(
        r"DeepLabV3-R50::20\d{6}",
        txt,
    )))
    if len(hits) != 1:
        raise RuntimeError(
            f"{lock_path}: expected one DeepLab model_state_id, got {hits}"
        )
    return hits[0]


def resolve_unique_npz_manifest(
    npz_path: Path,
    expected_state_id: str,
    required_state_keys: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
    """
    Resolve an exact row-order sidecar for one R33A2B packed mask NPZ.

    Multiple sidecars are accepted only if they imply the identical
    sample_id -> packed-row mapping.
    """
    manifests = find_npz_manifest(npz_path, expected_rows)
    valid = []

    for meta in manifests:
        x = meta.copy()
        if len(x) != expected_rows:
            continue

        # State can be absent from the sidecar; this NPZ is already uniquely
        # state-locked by its pre-HARM lock JSON.
        if "model_state_id" in x.columns:
            x["model_state_id"] = expected_state_id
        else:
            x["model_state_id"] = expected_state_id

        req = required_state_keys[KEY_COLS].copy().reset_index(drop=True)
        try:
            merged = req.merge(
                x[["sample_id", "model_state_id", "_mask_row"]],
                on=KEY_COLS,
                how="left",
                validate="one_to_one",
            )
        except Exception:
            continue

        if merged["_mask_row"].isna().any():
            continue
        valid.append(merged)

    if not valid:
        raise RuntimeError(
            f"{npz_path}: no exact-key {expected_rows}-row manifest for {expected_state_id}"
        )

    ref = valid[0][KEY_COLS + ["_mask_row"]].sort_values(KEY_COLS).reset_index(drop=True)
    for other in valid[1:]:
        oth = other[KEY_COLS + ["_mask_row"]].sort_values(KEY_COLS).reset_index(drop=True)
        if not ref.equals(oth):
            raise RuntimeError(
                f"{npz_path}: multiple manifests imply different row mappings."
            )

    return valid[0]


def bind_r33_polypgen_memo_geometry(
    required: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    req = required[KEY_COLS].copy().reset_index(drop=True)
    if len(req) != TARGET_ROWS:
        raise RuntimeError(
            f"PolypGen MEMO: required rows={len(req)} != {TARGET_ROWS}"
        )

    parts = []
    assets = []

    seen_states = set()
    for idx in range(3):
        npz_path = (
            R33A2B_MASK_ROOT
            / f"state_{idx:02d}_polypgen_source_memo_pre_harm.npz"
        )
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)

        lock_path = find_r33_state_lock_json(idx)
        state_id = extract_r33_model_state_id(lock_path)
        if state_id in seen_states:
            raise RuntimeError(f"Duplicate R33 state mapping={state_id}")
        seen_states.add(state_id)

        state_req = req[req["model_state_id"].astype(str) == state_id].copy()
        if len(state_req) != TARGET_IMAGES:
            raise RuntimeError(
                f"PolypGen MEMO/{state_id}: rows={len(state_req)} != {TARGET_IMAGES}"
            )

        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
        for key in ["source_masks_packed", "memo_masks_packed"]:
            if key not in z.files:
                raise RuntimeError(
                    f"{npz_path}: missing key={key}; available={z.files}"
                )
            if z[key].shape != (TARGET_IMAGES, PACKED_BYTES):
                raise RuntimeError(
                    f"{npz_path}::{key} shape={z[key].shape}, "
                    f"expected={(TARGET_IMAGES, PACKED_BYTES)}"
                )

        bound = resolve_unique_npz_manifest(
            npz_path,
            state_id,
            state_req,
            TARGET_IMAGES,
        )
        rr = bound["_mask_row"].to_numpy(dtype=np.int64)

        geom = geometry_from_rows(
            z["source_masks_packed"],
            z["memo_masks_packed"],
            rr,
        )

        part = bound[KEY_COLS].copy()
        part[[
            "pred_dice", "pred_iou", "area_delta", "boundary_delta"
        ]] = geom
        parts.append(part)

        assets.append({
            "state_index": idx,
            "model_state_id": state_id,
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "source_key": "source_masks_packed",
            "candidate_key": "memo_masks_packed",
            "rows": TARGET_IMAGES,
        })

    if len(seen_states) != TARGET_STATES:
        raise RuntimeError(
            f"PolypGen MEMO mapped states={len(seen_states)} != {TARGET_STATES}"
        )

    allg = pd.concat(parts, ignore_index=True)
    geom_cols = [
        "pred_dice", "pred_iou", "area_delta", "boundary_delta"
    ]
    merged = req.merge(
        allg,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if merged[geom_cols].isna().any().any():
        raise RuntimeError("PolypGen MEMO geometry incomplete.")

    return merged[geom_cols].to_numpy(dtype=np.float64), {
        "lineage": "R33A2B fix1 pre-HARM SOURCE/MEMO masks",
        "action": "MEMO",
        "rows": len(merged),
        "states": len(seen_states),
        "assets": assets,
    }


def relevant_mask_paths(inv: pd.DataFrame) -> List[Path]:
    roles = inv["roles"].fillna("").astype(str)
    sub = inv[
        roles.str.contains("SOURCE_MASK_CANDIDATE", regex=False)
        | roles.str.contains("MEMO_MASK_CANDIDATE", regex=False)
        | roles.str.contains("KEYED_TABLE", regex=False)
    ]
    paths = []
    for x in sub["path"].dropna().astype(str).unique():
        p = Path(x)
        if p.is_file() and p.suffix.lower() == ".npz":
            paths.append(p)

    # Add sibling NPZs around candidates, useful when inventory role attached to sidecar.
    for p0 in list(paths):
        for p in p0.parent.glob("*.npz"):
            if p not in paths:
                paths.append(p)
    return paths


def find_npz_manifest(npz_path: Path, n: int) -> List[pd.DataFrame]:
    results = []

    # Search local and parent directories first.
    dirs = [npz_path.parent, npz_path.parent.parent]
    seen = set()

    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.csv"):
            if p in seen:
                continue
            seen.add(p)
            try:
                h = pd.read_csv(p, nrows=0, low_memory=False)
            except Exception:
                continue

            sc = first_existing(h.columns, SAMPLE_ALIASES)
            stc = first_existing(h.columns, STATE_ALIASES)
            if not sc or not stc:
                continue

            refc = next((c for c in NPZ_REF_ALIASES if c in h.columns), None)
            ric = next((c for c in ROW_INDEX_ALIASES if c in h.columns), None)

            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue

            if refc is not None:
                ref = df[refc].astype(str)
                m = ref.map(lambda x: Path(x).name.lower() == npz_path.name.lower())
                sub = df[m].copy()
            else:
                # Conservative fallback: only same-directory table with exact n rows.
                sub = df.copy() if p.parent == npz_path.parent and len(df) == n else df.iloc[0:0].copy()

            if len(sub) != n:
                continue

            out = pd.DataFrame({
                "sample_id": norm_id(sub[sc]),
                "model_state_id": sub[stc].astype(str).str.strip(),
            })

            if ric is not None:
                idx = pd.to_numeric(sub[ric], errors="coerce")
                if idx.isna().any():
                    continue
                out["_mask_row"] = idx.astype(int).to_numpy()
            else:
                out["_mask_row"] = np.arange(n, dtype=np.int64)

            if out["_mask_row"].min() < 0 or out["_mask_row"].max() >= n:
                continue
            if out[KEY_COLS].drop_duplicates().shape[0] != n:
                continue

            out["_manifest_path"] = str(p)
            results.append(out)

    return results


def mask_key_pairs(npz_path: Path) -> List[Tuple[str, str, str]]:
    """
    Return tuples: (source_key, candidate_key, inferred_action).
    """
    try:
        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return []

    keys = list(z.files)
    source_keys = [
        k for k in keys
        if "source" in k.lower() and "mask" in k.lower()
        and getattr(z[k], "ndim", 0) == 2
        and z[k].shape[1] == PACKED_BYTES
    ]
    if not source_keys:
        return []

    pairs = []
    for sk in source_keys:
        for ck in keys:
            if ck == sk:
                continue
            arr = z[ck]
            if getattr(arr, "ndim", 0) != 2 or arr.shape[1] != PACKED_BYTES:
                continue
            low = ck.lower()
            if "mask" not in low:
                continue
            action = normalize_action_value(ck)
            if action == "":
                action = infer_action_from_path(npz_path)
            if action == "" and low.startswith("a1"):
                action = "TENT1"
            if action == "" and any(t in low for t in ["candidate", "adapted"]):
                action = infer_action_from_path(npz_path)
            if action:
                pairs.append((sk, ck, action))
    return pairs


def unpack_little(batch: np.ndarray) -> np.ndarray:
    p = np.asarray(batch, dtype=np.uint8)
    flat = np.unpackbits(
        p,
        axis=1,
        count=PIXELS,
        bitorder=BITORDER,
    )
    return flat.reshape(-1, H, W).astype(bool, copy=False)


def dice_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.logical_and(a, b).sum(axis=(1, 2), dtype=np.int64)
    den = a.sum(axis=(1, 2), dtype=np.int64) + b.sum(axis=(1, 2), dtype=np.int64)
    out = np.ones(len(a), dtype=np.float64)
    nz = den != 0
    out[nz] = 2.0 * inter[nz] / den[nz]
    return out


def iou_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.logical_and(a, b).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(a, b).sum(axis=(1, 2), dtype=np.int64)
    out = np.ones(len(a), dtype=np.float64)
    nz = union != 0
    out[nz] = inter[nz] / union[nz]
    return out


def morphology(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=bool)
    fg = m.mean(axis=(1, 2), dtype=np.float64)
    denom = float((H - 1) * W + H * (W - 1))
    v = np.not_equal(m[:, 1:, :], m[:, :-1, :]).sum(axis=(1, 2), dtype=np.int64)
    h = np.not_equal(m[:, :, 1:], m[:, :, :-1]).sum(axis=(1, 2), dtype=np.int64)
    bd = (v + h).astype(np.float64) / denom
    return np.column_stack([fg, bd])


def geometry_from_rows(source_packed: np.ndarray, cand_packed: np.ndarray, rows: np.ndarray, batch: int = 64) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), 4), dtype=np.float64)

    for start in range(0, len(rows), batch):
        end = min(start + batch, len(rows))
        idx = rows[start:end]
        src = unpack_little(np.asarray(source_packed[idx], dtype=np.uint8))
        can = unpack_little(np.asarray(cand_packed[idx], dtype=np.uint8))

        d = dice_between(src, can)
        j = iou_between(src, can)
        sm = morphology(src)
        cm = morphology(can)

        out[start:end] = np.column_stack([
            d,
            j,
            cm[:, 0] - sm[:, 0],
            cm[:, 1] - sm[:, 1],
        ])

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite simple mask-change geometry.")
    return out


def bind_geometry_panel(
    inv: pd.DataFrame,
    required: pd.DataFrame,
    action: str,
    label: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    required_keys = required[KEY_COLS].copy().reset_index(drop=True)
    required_tuples = set(map(tuple, required_keys.to_numpy()))

    pieces = []

    for p in tqdm(
        relevant_mask_paths(inv),
        desc=f"Bind masks {label}/{action}",
        unit="npz",
        dynamic_ncols=True,
    ):
        pairs = mask_key_pairs(p)
        pairs = [x for x in pairs if x[2] == action]
        if not pairs:
            continue

        try:
            z = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            continue

        for sk, ck, _ in pairs:
            source = z[sk]
            cand = z[ck]
            if source.shape != cand.shape or source.ndim != 2 or source.shape[1] != PACKED_BYTES:
                continue
            n = int(source.shape[0])

            manifests = find_npz_manifest(p, n)
            for meta in manifests:
                keys = set(map(tuple, meta[KEY_COLS].to_numpy()))
                overlap = len(keys & required_tuples)
                if overlap == 0:
                    continue

                sub = meta[
                    meta[KEY_COLS].apply(tuple, axis=1).isin(required_tuples)
                ].copy()
                if len(sub) == 0:
                    continue

                g = geometry_from_rows(
                    source,
                    cand,
                    sub["_mask_row"].to_numpy(dtype=np.int64),
                )

                piece = sub[KEY_COLS].copy()
                piece[["pred_dice", "pred_iou", "area_delta", "boundary_delta"]] = g
                piece["_asset"] = str(p)
                piece["_asset_sha256"] = sha256_file(p)
                piece["_source_key"] = sk
                piece["_candidate_key"] = ck
                piece["_manifest"] = sub["_manifest_path"].iloc[0]
                pieces.append(piece)

    if not pieces:
        raise RuntimeError(f"{label} {action} MASK_BINDING_FAILED: no keyed mask pair overlaps required panel.")

    allp = pd.concat(pieces, ignore_index=True)

    # Duplicated exact keys are allowed only if all four geometry values agree.
    geom_cols = ["pred_dice", "pred_iou", "area_delta", "boundary_delta"]
    conflicts = []
    chosen_rows = []

    for key, g in allp.groupby(KEY_COLS, sort=False):
        arr = g[geom_cols].to_numpy(dtype=np.float64)
        if len(arr) > 1 and not np.allclose(arr, arr[0], rtol=0.0, atol=0.0):
            conflicts.append(key)
            continue
        chosen_rows.append(g.iloc[0])

    if conflicts:
        raise RuntimeError(
            f"{label} {action} MASK_BINDING_AMBIGUOUS: {len(conflicts)} duplicated keys have non-identical geometry."
        )

    panel = pd.DataFrame(chosen_rows)
    if panel[KEY_COLS].drop_duplicates().shape[0] != len(panel):
        raise RuntimeError("Internal geometry de-duplication failed.")

    merged = required_keys.merge(panel, on=KEY_COLS, how="left", validate="one_to_one")
    if merged[geom_cols].isna().any().any():
        missing = int(merged[geom_cols[0]].isna().sum())
        raise RuntimeError(
            f"{label} {action} MASK_BINDING_INCOMPLETE: missing geometry for {missing}/{len(merged)} required rows."
        )

    asset_summary = (
        panel[["_asset", "_asset_sha256", "_source_key", "_candidate_key", "_manifest"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    return merged[geom_cols].to_numpy(dtype=np.float64), {
        "action": action,
        "rows": int(len(merged)),
        "assets": asset_summary,
    }


def choose_frozen_full_score(inv: pd.DataFrame, target_outcomes: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Exact historical reference: R33A2B pre-HARM external SafeTTA score lock.
    No refit and no post-outcome score selection.
    """
    p = R33A2B_SCORE_PRE_HARM
    if not p.is_file():
        return None, None

    df = pd.read_csv(p, low_memory=False)
    required = {
        "sample_id", "model_state_id", "model_family",
        "training_seed", "action", "safettta_external_harm_risk"
    }
    if not required.issubset(df.columns):
        return None, None
    if len(df) != TARGET_ROWS:
        raise RuntimeError("R33A2B frozen score row count changed.")
    if set(df["action"].map(normalize_action_value)) != {"MEMO"}:
        raise RuntimeError("R33A2B frozen score action changed.")

    score = pd.DataFrame({
        "sample_id": norm_id(df["sample_id"]),
        "model_state_id": df["model_state_id"].astype(str).str.strip(),
        "score": pd.to_numeric(
            df["safettta_external_harm_risk"],
            errors="raise",
        ).astype(float),
    })
    if score[KEY_COLS].drop_duplicates().shape[0] != TARGET_ROWS:
        raise RuntimeError("R33A2B frozen score key not unique.")

    joined = target_outcomes[KEY_COLS].merge(
        score,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    if joined["score"].isna().any():
        raise RuntimeError(
            "R33A2B frozen score does not cover exact target keys."
        )
    vals = joined["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise RuntimeError("R33A2B frozen score contains non-finite values.")

    return vals, {
        "path": str(p),
        "sha256": sha256_file(p),
        "score_column": "safettta_external_harm_risk",
        "pre_harm_lock": True,
        "rows": TARGET_ROWS,
    }

def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260820,
        )),
    ])


def metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) != 2:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": float(y.mean())}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "prevalence": float(y.mean()),
    }


def patient_image_bootstrap(
    target: pd.DataFrame,
    score_cols: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.asarray(sorted(target["sample_id"].unique()))
    if len(samples) != TARGET_IMAGES:
        raise RuntimeError("Unexpected target physical-image count.")

    sample_arr = target["sample_id"].to_numpy()
    y = target["harm_label"].to_numpy(dtype=np.int64)
    idx_map = {s: np.flatnonzero(sample_arr == s) for s in samples}

    rep_rows = []
    delta_rows = []

    comparisons = [
        ("SOURCE_PLUS_SIMPLE_MASK_CHANGE", "SOURCE_STATE"),
    ]
    if "FROZEN_FULL_TRANSITION_SAFETTA" in score_cols:
        comparisons += [
            ("FROZEN_FULL_TRANSITION_SAFETTA", "SOURCE_STATE"),
            ("SOURCE_PLUS_SIMPLE_MASK_CHANGE", "FROZEN_FULL_TRANSITION_SAFETTA"),
        ]

    for b in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P01B physical-image bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        draw = rng.choice(samples, size=len(samples), replace=True)
        idx = np.concatenate([idx_map[s] for s in draw])
        yy = y[idx]
        if len(np.unique(yy)) != 2:
            continue

        current = {}
        for name, score in score_cols.items():
            ss = np.asarray(score, dtype=np.float64)[idx]
            current[name] = {
                "auroc": float(roc_auc_score(yy, ss)),
                "auprc": float(average_precision_score(yy, ss)),
            }
            rep_rows.append({
                "replicate": b,
                "representation": name,
                **current[name],
            })

        for a, c in comparisons:
            if a not in current or c not in current:
                continue
            for metric_name in ["auroc", "auprc"]:
                delta_rows.append({
                    "replicate": b,
                    "comparison": f"{a} - {c}",
                    "metric": metric_name.upper(),
                    "delta": current[a][metric_name] - current[c][metric_name],
                })

    return pd.DataFrame(rep_rows), pd.DataFrame(delta_rows)


def summarize_bootstrap(reps: pd.DataFrame, deltas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for (rep, metric), g in reps.melt(
        id_vars=["replicate", "representation"],
        value_vars=["auroc", "auprc"],
        var_name="metric",
        value_name="value",
    ).groupby(["representation", "metric"]):
        x = g["value"].to_numpy(dtype=np.float64)
        ci_rows.append({
            "representation": rep,
            "metric": metric.upper(),
            "valid_reps": int(len(x)),
            "ci95_low": float(np.quantile(x, 0.025)),
            "ci95_high": float(np.quantile(x, 0.975)),
        })

    delta_rows = []
    for (comp, metric), g in deltas.groupby(["comparison", "metric"]):
        x = g["delta"].to_numpy(dtype=np.float64)
        lo = float(np.quantile(x, 0.025))
        hi = float(np.quantile(x, 0.975))
        delta_rows.append({
            "comparison": comp,
            "metric": metric,
            "valid_reps": int(len(x)),
            "bootstrap_mean_delta": float(np.mean(x)),
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def self_test():
    # Exact B6-P01A geometry definitions, adapted only for colonoscopy little-endian packbits.
    a = np.zeros((1, H, W), dtype=bool)
    b = np.zeros((1, H, W), dtype=bool)
    assert dice_between(a, b)[0] == 1.0
    assert iou_between(a, b)[0] == 1.0
    assert morphology(a)[0, 0] == 0.0
    assert SOURCE_ROWS_PER_ACTION * 2 == SOURCE_TOTAL_ROWS
    assert TARGET_IMAGES * TARGET_STATES == TARGET_ROWS
    print("SELF_TEST_GEOMETRY=PASS")
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA B6-P0-1B PolypGen unseen-MEMO geometry-core experiment."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    protocol, inv = verify_lineage(Path(__file__).resolve().parent)

    print("=" * 160)
    print("SafeTTA B6-P0-1B — PolypGen unseen-MEMO geometry-core candidate-conditioning experiment")
    print(f"Version                  : {VERSION}")
    print(f"Protocol SHA             : {EXPECTED_PROTOCOL_SHA256}")
    print("Source actions            : TENT1 + PL-CONF90")
    print("Target action             : unseen MEMO")
    print("New matched models        : SOURCE_STATE vs SOURCE_STATE+SIMPLE_MASK_CHANGE")
    print("Target fit/calibration    : 0 / 0")
    print("Geometry redesign         : NO")
    print("Semantic DINO replay      : NO")
    print("Bootstrap                 : 2000 physical-image clusters")
    print("=" * 160)

    print("\n[1/6] Exact source/target outcome binding")
    source_outcomes, source_outcome_meta = choose_source_outcomes(inv)
    target_outcomes, target_outcome_meta = choose_target_outcomes(inv)

    print("SOURCE rows =", len(source_outcomes))
    print("SOURCE images =", source_outcomes["sample_id"].nunique())
    print("SOURCE states =", source_outcomes["model_state_id"].nunique())
    print("TARGET rows =", len(target_outcomes))
    print("TARGET images =", target_outcomes["sample_id"].nunique())
    print("TARGET states =", target_outcomes["model_state_id"].nunique())
    print("OUTCOME_BINDING=PASS")

    print("\n[2/6] Bind exact SOURCE-state 66-D panels")
    source66_unique, source66_keys, source66_meta = load_source66(source_outcomes)
    target66, target66_meta = choose_target_source66(inv, target_outcomes)

    print("SOURCE66 unique panel =", source66_unique.shape)
    print("TARGET66 =", target66.shape)
    print("SOURCE_STATE_BINDING=PASS")

    print("\n[3/6] Bind adjudicated historical candidate-mask change")
    mask_adjudication_meta = verify_authoritative_mask_adjudication()
    print("NeoPolyp historical mask status  =", EXPECTED_R38A1A_STATUS)
    print("NeoPolyp historical decision     =", EXPECTED_R38A1A_DECISION)
    print("NeoPolyp historical manifest SHA =", EXPECTED_R38A1A_MANIFEST_SHA256)

    source_geom_parts = []
    for action in TRAIN_ACTIONS:
        req = source_outcomes[
            source_outcomes["action"] == action
        ][KEY_COLS]
        g, meta = bind_r38a1a_neopolyp_geometry(req, action)
        source_geom_parts.append((action, g, meta))
        print(f"{action} historical geometry =", g.shape)

    target_geom, target_geom_meta = bind_r33_polypgen_memo_geometry(
        target_outcomes[KEY_COLS]
    )
    print("MEMO pre-HARM geometry =", target_geom.shape)
    print("AUTHORITATIVE_MASK_ADJUDICATION=PASS")
    print("MASK_CHANGE_BINDING=PASS")

    # Expand 66-D source state in exact source_outcome row order.
    source66_map = source66_keys.copy()
    source66_map["_idx66"] = np.arange(len(source66_map), dtype=np.int64)
    source_join = source_outcomes.merge(
        source66_map,
        on=KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    if source_join["_idx66"].isna().any():
        raise RuntimeError("SOURCE66 expansion failed.")
    x_source66 = source66_unique[source_join["_idx66"].to_numpy(dtype=np.int64)]

    geom_by_action = {}
    for action, g, meta in source_geom_parts:
        geom_by_action[action] = g

    # source_outcomes is concatenated in TRAIN_ACTIONS order by choose_source_outcomes.
    x_source_geom = np.vstack([geom_by_action[a] for a in TRAIN_ACTIONS])
    if x_source_geom.shape != (SOURCE_TOTAL_ROWS, 4):
        raise RuntimeError(f"Expanded source geometry shape={x_source_geom.shape}")
    if x_source66.shape != (SOURCE_TOTAL_ROWS, 66):
        raise RuntimeError(f"Expanded source state shape={x_source66.shape}")

    y_source = source_outcomes["harm_label"].to_numpy(dtype=np.int64)
    y_target = target_outcomes["harm_label"].to_numpy(dtype=np.int64)

    if args.preflight_only:
        print("\nP01B_GEOMETRY_PREFLIGHT=PASS")
        print("NEW_MODEL_FIT=NOT_RUN")
        print("SCIENTIFIC_METRICS=NOT_RUN")
        print("GATE=PASS_B6_P01B_GEOMETRY_PREFLIGHT_ONLY")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P01B geometry-core results: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/6] Fit matched NeoPolyp models and score PolypGen")
    model_source = make_model()
    model_geom = make_model()

    model_source.fit(x_source66, y_source)
    model_geom.fit(np.column_stack([x_source66, x_source_geom]), y_source)

    s_source = model_source.predict_proba(target66)[:, 1]
    s_geom = model_geom.predict_proba(np.column_stack([target66, target_geom]))[:, 1]

    full_score, full_score_meta = choose_frozen_full_score(inv, target_outcomes)

    scores = {
        "SOURCE_STATE": s_source,
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE": s_geom,
    }
    if full_score is not None:
        scores["FROZEN_FULL_TRANSITION_SAFETTA"] = full_score

    metric_rows = []
    for name, s in scores.items():
        m = metrics(y_target, s)
        metric_rows.append({
            "representation": name,
            **m,
            "n": len(y_target),
            "harm_n": int(y_target.sum()),
        })

    metric_df = pd.DataFrame(metric_rows)
    print("\nPolypGen unseen-MEMO point metrics")
    print(metric_df.to_string(index=False))

    print("\n[5/6] Paired physical-image bootstrap")
    target_eval = target_outcomes[KEY_COLS + ["harm_label"]].copy()
    reps, deltas = patient_image_bootstrap(target_eval, scores)
    ci_df, delta_df = summarize_bootstrap(reps, deltas)

    # Add point deltas.
    point_map = {
        r["representation"]: r
        for r in metric_df.to_dict(orient="records")
    }
    point_delta = []
    for _, r in delta_df.iterrows():
        a, c = str(r["comparison"]).split(" - ")
        metric_key = str(r["metric"]).lower()
        point_delta.append(
            point_map[a][metric_key] - point_map[c][metric_key]
        )
    if len(delta_df):
        delta_df.insert(3, "point_delta", point_delta)

    print("\nPaired deltas")
    print(delta_df.to_string(index=False))

    print("\n[6/6] Freeze outputs")
    p_metrics = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_METRICS.csv"
    p_ci = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_BOOTSTRAP_CI.csv"
    p_delta = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_PAIRED_DELTAS.csv"
    p_reps = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_BOOTSTRAP_REPLICATES.csv"
    p_scores = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_SCORES.csv"

    metric_df.to_csv(p_metrics, index=False)
    ci_df.to_csv(p_ci, index=False)
    delta_df.to_csv(p_delta, index=False)
    reps.to_csv(p_reps, index=False)

    score_out = target_outcomes[KEY_COLS + ["harm_label"]].copy()
    for name, s in scores.items():
        score_out[name] = s
    score_out.to_csv(p_scores, index=False)

    primary = delta_df[
        (delta_df["comparison"] == "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE")
        & (delta_df["metric"] == "AUROC")
    ]
    if len(primary) != 1:
        raise RuntimeError("Primary paired AUROC delta row missing.")
    pr = primary.iloc[0]

    decision = (
        "CANDIDATE_CONDITIONING_INCREMENT_SUPPORTED"
        if float(pr["point_delta"]) > 0 and float(pr["ci95_low"]) > 0
        else "CANDIDATE_CONDITIONING_INCREMENT_NOT_SUPPORTED"
    )

    binding = {
        "mask_adjudication": mask_adjudication_meta,
        "source_outcomes": source_outcome_meta,
        "target_outcomes": target_outcome_meta,
        "source66": source66_meta,
        "target66": target66_meta,
        "source_geometry": {
            action: meta for action, _, meta in source_geom_parts
        },
        "target_geometry": target_geom_meta,
        "frozen_full_score": full_score_meta,
    }

    p_binding = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_BINDING.json"
    write_json(p_binding, binding)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "decision": decision,
        "version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "parent_protocol_sha256": EXPECTED_PARENT_PROTOCOL_SHA256,
        "target_fit_rows": 0,
        "target_calibration_rows": 0,
        "geometry_features_changed_after_target": False,
        "semantic_dino_replay": False,
        "source_train_rows": int(len(source_outcomes)),
        "target_rows": int(len(target_outcomes)),
        "target_physical_images": int(target_outcomes["sample_id"].nunique()),
        "target_harm_prevalence": float(y_target.mean()),
        "primary_comparison": "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE",
        "primary_auroc_point_delta": float(pr["point_delta"]),
        "primary_auroc_ci95": [float(pr["ci95_low"]), float(pr["ci95_high"])],
        "frozen_full_semantic_reference_included": full_score is not None,
        "outputs": {},
    }

    for p in [p_metrics, p_ci, p_delta, p_reps, p_scores, p_binding]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 160,
        "SafeTTA B6-P0-1B POLYPGEN GEOMETRY CORE EXPERIMENT COMPLETE",
        metric_df.to_string(index=False),
        "",
        "PAIRED DELTAS:",
        delta_df.to_string(index=False),
        "",
        f"DECISION={decision}",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 160,
        "",
    ])
    (OUT_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_REPORT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
