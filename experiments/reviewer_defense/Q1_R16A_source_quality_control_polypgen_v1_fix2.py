#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R16A_source_quality_control_polypgen_v1_fix2.py

SafeTTA R16-A fix2
==================

Why fix2 is needed
------------------
v1/fix1 assumed that one audit-selected PolypGen table already contained every
R16-A field. The local audit shows PolypGen has relevant frozen inputs, but the
strict single-table resolver found no exact table satisfying all invariants.

Fix2 therefore performs a *forensic, invariant-driven reconstruction* from
retained frozen row tables:

1) Search all PolypGen candidate tables from the prior audit manifest.
2) Keep only exact 13,788-row frozen panel tables.
3) Identify physical-image IDs from content (must yield exactly 1,532 images).
4) Identify model/family/state keys from content.
5) First prefer a single exact table containing:
      physical ID + SOURCE Dice + SafeTTA risk + HARM/DeltaDice + family.
6) If no such table exists, reconstruct only by an *exact key join* between
   frozen tables. Row-order joins are prohibited.
7) If exact key alignment cannot be proven, BLOCK and emit a detailed forensic
   report rather than guessing.

Scientific analysis is unchanged from the frozen R16 preregistration:
    M_quality      : HARM ~ z(SOURCE_Dice) + architecture_family
    M_quality+risk : HARM ~ z(SOURCE_Dice) + z(SafeTTA_risk)
                            + architecture_family

This is a POST-REVEAL explanatory audit only.
SOURCE Dice is never a deployable SafeTTA input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-08-Q1-R16A-SOURCE-QUALITY-CONTROL-POLYPGEN-v1-fix2"

DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_AUDIT = DEFAULT_ROOT / r"outputs\Q1_R16A_reviewer_defense_input_audit_v1\CANDIDATE_TABLES.csv"
DEFAULT_OUT = DEFAULT_ROOT / r"outputs\Q1_R16A_source_quality_control_polypgen_v1_fix2"

EXPECTED_ROWS = 13788
EXPECTED_PHYSICAL = 1532
EXPECTED_STATES = 9
EXPECTED_FAMILIES = 3
HARM_MARGIN = -0.02
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260908

SUPPORTED = {".csv", ".tsv", ".parquet", ".feather", ".txt"}

# Conservative naming hints. Content invariants remain mandatory.
PHYSICAL_HINTS = (
    "sample_id", "image_id", "physical_id", "case_id", "patient_id",
    "frame_id", "uid", "image_name", "filename", "file_name", "image_path",
    "sample", "image", "frame", "case",
)
FAMILY_HINTS = (
    "family", "model_family", "architecture", "arch", "backbone",
)
STATE_HINTS = (
    "state", "model_state", "checkpoint", "ckpt", "model_id", "run_id",
    "seed", "model", "run", "checkpoint_sha", "sha",
)
SOURCE_DICE_HINTS = (
    "source_dice", "src_dice", "dice_source", "dice_src", "baseline_dice",
    "pre_dice", "dice_before", "source_slice_dice", "sourceDice", "srcDice",
)
RISK_HINTS = (
    "risk_score", "harm_risk", "safety_score", "final_score", "pred_risk",
    "p_harm", "prob_harm", "frozen_risk", "frozen_score", "risk",
)
DELTA_HINTS = (
    "delta_dice", "dice_delta", "adapt_minus_source_dice", "delta_dice_tent1",
    "ddice", "delta_d", "delta",
)
HARM_HINTS = (
    "harm", "is_harm", "harm_label", "harm_flag", "harm02", "harm_002",
    "y_harm", "label_harm",
)

# Generic "score" is deliberately NOT a default risk hint. It is accepted only
# when the prior audit explicitly classified that column as a risk column.


def norm(x: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower())).strip("_")


def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    if suf == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported table: {path}")


def parse_role_cell(v) -> List[str]:
    if pd.isna(v):
        return []
    return [x for x in str(v).split("|") if x]


def add_unique(out: List[str], values: Iterable[str], valid_cols: Sequence[str]) -> None:
    valid = set(valid_cols)
    seen = set(out)
    for c in values:
        if c in valid and c not in seen:
            out.append(c)
            seen.add(c)


def columns_from_audit_and_hints(
    row: pd.Series,
    df: pd.DataFrame,
    role: str,
    hints: Sequence[str],
) -> List[str]:
    out: List[str] = []
    add_unique(out, parse_role_cell(row.get(f"role_{role}", "")), df.columns)

    nmap = {norm(c): c for c in df.columns}
    for h in hints:
        c = nmap.get(norm(h))
        if c is not None and c not in out:
            out.append(c)

    # Conservative semantic-name fallback.
    for c in df.columns:
        nc = norm(c)
        if role == "source_dice":
            if "dice" in nc and any(t in nc for t in ("source", "src", "baseline", "pre", "before")):
                if c not in out:
                    out.append(c)
        elif role == "risk":
            if any(t in nc for t in ("risk", "harm_risk", "safety")):
                if c not in out:
                    out.append(c)
        elif role == "delta_dice":
            if "dice" in nc and any(t in nc for t in ("delta", "change", "diff")):
                if c not in out:
                    out.append(c)
        elif role == "harm":
            if "harm" in nc:
                if c not in out:
                    out.append(c)
    return out


def finite_numeric(s: pd.Series) -> Tuple[float, np.ndarray]:
    a = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return float(np.isfinite(a).mean()), a


def valid_unit_interval(s: pd.Series) -> bool:
    ff, a = finite_numeric(s)
    if ff < 0.995:
        return False
    a = a[np.isfinite(a)]
    return bool(len(a) and a.min() >= -1e-6 and a.max() <= 1 + 1e-6)


def valid_delta(s: pd.Series) -> bool:
    ff, a = finite_numeric(s)
    if ff < 0.995:
        return False
    a = a[np.isfinite(a)]
    return bool(len(a) and a.min() >= -1.000001 and a.max() <= 1.000001)


def to_binary_harm(s: pd.Series) -> Optional[pd.Series]:
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().mean() >= 0.995:
        vals = set(n.dropna().astype(float).unique().tolist())
        if vals.issubset({0.0, 1.0}) and len(vals) == 2:
            return (n > 0).astype(int)

    x = s.astype(str).str.strip().str.lower()
    vals = set(x.dropna().unique().tolist())
    allowed = {
        "0", "1", "true", "false", "yes", "no", "harm",
        "nonharm", "non-harm", "safe",
    }
    if vals and vals.issubset(allowed):
        y = x.isin({"1", "true", "yes", "harm"}).astype(int)
        if set(y.unique()) == {0, 1}:
            return y
    return None


def family_from_strings(s: pd.Series) -> Optional[pd.Series]:
    x = s.astype(str).str.lower()
    out = pd.Series(index=s.index, dtype="object")
    out[x.str.contains("deeplab", na=False)] = "DeepLabV3-R50"
    out[x.str.contains("pranet", na=False)] = "PraNet"
    out[x.str.contains("segformer", na=False)] = "SegFormer-B0"
    if out.notna().all() and out.nunique() == EXPECTED_FAMILIES:
        return out
    return None


def detect_physical_candidates(row: pd.Series, df: pd.DataFrame) -> List[Dict[str, object]]:
    candidates: List[str] = []
    add_unique(candidates, parse_role_cell(row.get("role_physical_id", "")), df.columns)

    nmap = {norm(c): c for c in df.columns}
    for h in PHYSICAL_HINTS:
        c = nmap.get(norm(h))
        if c is not None and c not in candidates:
            candidates.append(c)

    # Content fallback: any non-floating column with exactly 1532 unique values
    # and no missing values is a candidate.
    for c in df.columns:
        if c in candidates:
            continue
        if df[c].isna().any():
            continue
        nunique = int(df[c].astype(str).nunique())
        if nunique == EXPECTED_PHYSICAL:
            candidates.append(c)

    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c]
        nunique = int(s.astype(str).nunique(dropna=True))
        missing = int(s.isna().sum())
        counts = s.astype(str).value_counts()
        exact_repeat9 = bool(
            nunique == EXPECTED_PHYSICAL
            and missing == 0
            and len(counts) == EXPECTED_PHYSICAL
            and int(counts.min()) == EXPECTED_STATES
            and int(counts.max()) == EXPECTED_STATES
        )
        if nunique == EXPECTED_PHYSICAL and missing == 0:
            nc = norm(c)
            pri = 0
            if nc in {"sample_id", "image_id", "physical_id", "image_uid", "sample_uid"}:
                pri = 5
            elif "id" in nc or "uid" in nc:
                pri = 4
            elif "name" in nc:
                pri = 3
            elif "path" in nc:
                pri = 2
            else:
                pri = 1
            if exact_repeat9:
                pri += 5
            out.append({
                "column": c,
                "priority": pri,
                "unique": nunique,
                "repeat9": exact_repeat9,
            })
    return sorted(out, key=lambda d: (-d["priority"], d["column"]))


def detect_family_candidates(row: pd.Series, df: pd.DataFrame) -> List[Dict[str, object]]:
    candidates: List[str] = []
    add_unique(candidates, parse_role_cell(row.get("role_family", "")), df.columns)
    nmap = {norm(c): c for c in df.columns}
    for h in FAMILY_HINTS + STATE_HINTS:
        c = nmap.get(norm(h))
        if c is not None and c not in candidates:
            candidates.append(c)

    # Also inspect columns with 3 or 9 unique values.
    for c in df.columns:
        if c in candidates or df[c].isna().any():
            continue
        nu = int(df[c].astype(str).nunique())
        if nu in {EXPECTED_FAMILIES, EXPECTED_STATES}:
            candidates.append(c)

    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c]
        nu = int(s.astype(str).nunique(dropna=True))
        if nu == EXPECTED_FAMILIES:
            # Accept exact 3-level column as family only when the column name is
            # architecture/family-like or values explicitly reveal families.
            fam = family_from_strings(s)
            nc = norm(c)
            name_ok = any(tok in nc for tok in ("family", "arch", "backbone", "model"))
            if fam is not None or name_ok:
                series = fam if fam is not None else s.astype(str)
                out.append({
                    "column": c,
                    "series": series,
                    "priority": 5 if fam is not None else 3,
                    "mode": "direct3",
                })
        elif nu == EXPECTED_STATES:
            fam = family_from_strings(s)
            if fam is not None:
                out.append({
                    "column": c,
                    "series": fam,
                    "priority": 4,
                    "mode": "derived_from_9state_strings",
                })
    return sorted(out, key=lambda d: (-d["priority"], d["column"]))


def detect_state_candidates(row: pd.Series, df: pd.DataFrame) -> List[Dict[str, object]]:
    """State key must have exactly 9 levels, each repeated exactly 1532 times."""
    candidates: List[str] = []
    add_unique(candidates, parse_role_cell(row.get("role_family", "")), df.columns)
    nmap = {norm(c): c for c in df.columns}
    for h in STATE_HINTS:
        c = nmap.get(norm(h))
        if c is not None and c not in candidates:
            candidates.append(c)

    for c in df.columns:
        if c in candidates or df[c].isna().any():
            continue
        if int(df[c].astype(str).nunique()) == EXPECTED_STATES:
            candidates.append(c)

    out = []
    for c in candidates:
        if c not in df.columns or df[c].isna().any():
            continue
        s = df[c].astype(str)
        vc = s.value_counts()
        if (
            len(vc) == EXPECTED_STATES
            and int(vc.min()) == EXPECTED_PHYSICAL
            and int(vc.max()) == EXPECTED_PHYSICAL
        ):
            nc = norm(c)
            pri = 1
            if any(tok in nc for tok in ("state", "checkpoint", "ckpt", "model_id", "run_id", "sha")):
                pri = 5
            elif "seed" in nc:
                pri = 3
            elif "model" in nc or "run" in nc:
                pri = 2
            out.append({
                "column": c,
                "priority": pri,
                "values": sorted(s.unique().tolist()),
            })
    return sorted(out, key=lambda d: (-d["priority"], d["column"]))


def detect_numeric_role(row: pd.Series, df: pd.DataFrame, role: str) -> List[Dict[str, object]]:
    hints = {
        "source_dice": SOURCE_DICE_HINTS,
        "risk": RISK_HINTS,
        "delta_dice": DELTA_HINTS,
    }[role]
    candidates = columns_from_audit_and_hints(row, df, role, hints)
    out = []

    for c in candidates:
        ok = valid_delta(df[c]) if role == "delta_dice" else valid_unit_interval(df[c])
        if not ok:
            continue
        a = pd.to_numeric(df[c], errors="coerce")
        if a.nunique(dropna=True) < 20:
            continue

        nc = norm(c)
        pri = 1
        for i, h in enumerate(hints):
            if nc == norm(h):
                pri = 100 - i
                break
        if role == "risk" and any(tok in nc for tok in ("risk", "safety", "harm")):
            pri += 20
        if role == "source_dice" and "dice" in nc:
            pri += 10

        out.append({
            "column": c,
            "priority": pri,
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
        })
    return sorted(out, key=lambda d: (-d["priority"], d["column"]))


def detect_harm_role(row: pd.Series, df: pd.DataFrame) -> List[Dict[str, object]]:
    candidates = columns_from_audit_and_hints(row, df, "harm", HARM_HINTS)
    out = []
    for c in candidates:
        y = to_binary_harm(df[c])
        if y is not None:
            nc = norm(c)
            pri = 1
            for i, h in enumerate(HARM_HINTS):
                if nc == norm(h):
                    pri = 100 - i
                    break
            out.append({
                "column": c,
                "priority": pri,
                "prevalence": float(y.mean()),
            })
    return sorted(out, key=lambda d: (-d["priority"], d["column"]))


def choose_unique_top(items: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not items:
        return None
    top = items[0]["priority"]
    tied = [d for d in items if d["priority"] == top]
    if len(tied) != 1:
        return None
    return tied[0]


def table_profile(row: pd.Series, path: Path, df: pd.DataFrame) -> Dict[str, object]:
    physical = detect_physical_candidates(row, df)
    family = detect_family_candidates(row, df)
    states = detect_state_candidates(row, df)
    source_dice = detect_numeric_role(row, df, "source_dice")
    risk = detect_numeric_role(row, df, "risk")
    delta = detect_numeric_role(row, df, "delta_dice")
    harm = detect_harm_role(row, df)

    return {
        "path": str(path),
        "rows": int(len(df)),
        "audit_relevance_score": float(row.get("relevance_score", -1) if not pd.isna(row.get("relevance_score", -1)) else -1),
        "physical_candidates": physical,
        "family_candidates": [
            {k: v for k, v in d.items() if k != "series"} for d in family
        ],
        "_family_objects": family,
        "state_candidates": states,
        "source_dice_candidates": source_dice,
        "risk_candidates": risk,
        "delta_dice_candidates": delta,
        "harm_candidates": harm,
    }


def load_exact_polypgen_profiles(audit: pd.DataFrame, out_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    x = audit[audit["dataset_guess"].astype(str).str.upper().eq("POLYPGEN")].copy()
    if "relevance_score" in x.columns:
        x["relevance_score"] = pd.to_numeric(x["relevance_score"], errors="coerce").fillna(-1)
        x = x.sort_values("relevance_score", ascending=False)

    profiles: List[Dict[str, object]] = []
    tables: Dict[str, pd.DataFrame] = {}
    audit_rows: Dict[str, pd.Series] = {}

    iterator = list(x.iterrows())
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Forensic PolypGen table scan", unit="table", dynamic_ncols=True)

    seen_paths = set()
    for _, row in iterator:
        path = Path(str(row["path"]))
        key = str(path)
        if key in seen_paths or not path.exists() or path.suffix.lower() not in SUPPORTED:
            continue
        seen_paths.add(key)
        try:
            df = read_any(path)
        except Exception:
            continue
        if len(df) != EXPECTED_ROWS:
            continue

        prof = table_profile(row, path, df)
        profiles.append(prof)
        tables[key] = df
        audit_rows[key] = row

    # Save JSON-safe forensic profile.
    safe_profiles = []
    for p in profiles:
        q = {k: v for k, v in p.items() if k != "_family_objects"}
        safe_profiles.append(q)
    (out_dir / "R16A_FIX2_FORENSIC_PROFILES.json").write_text(
        json.dumps(safe_profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return profiles, tables, audit_rows


def make_single_table_candidate(
    profile: Dict[str, object],
    df: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    pid = choose_unique_top(profile["physical_candidates"])
    fam_items = profile["_family_objects"]
    fam = choose_unique_top(fam_items)
    src = choose_unique_top(profile["source_dice_candidates"])
    risk = choose_unique_top(profile["risk_candidates"])
    harm = choose_unique_top(profile["harm_candidates"])
    delta = choose_unique_top(profile["delta_dice_candidates"])

    if pid is None or fam is None or src is None or risk is None or (harm is None and delta is None):
        return None

    out = pd.DataFrame({
        "physical_id": df[pid["column"]].astype(str),
        "family": fam["series"].astype(str),
        "source_dice": pd.to_numeric(df[src["column"]], errors="coerce"),
        "risk": pd.to_numeric(df[risk["column"]], errors="coerce"),
    })

    if delta is not None:
        out["delta_dice"] = pd.to_numeric(df[delta["column"]], errors="coerce")
        yd = (out["delta_dice"] <= HARM_MARGIN).astype(int)
    else:
        yd = None

    if harm is not None:
        yh = to_binary_harm(df[harm["column"]])
        if yh is None:
            return None
        out["harm"] = yh
        if yd is not None:
            agree = float((yh.to_numpy() == yd.to_numpy()).mean())
            if agree < 0.995:
                return None
    else:
        out["harm"] = yd

    meta = {
        "mode": "SINGLE_TABLE",
        "table": profile["path"],
        "physical_id_column": pid["column"],
        "family_column": fam["column"],
        "source_dice_column": src["column"],
        "risk_column": risk["column"],
        "harm_column": "" if harm is None else harm["column"],
        "delta_dice_column": "" if delta is None else delta["column"],
    }
    return out, meta


def build_key_frame(
    profile: Dict[str, object],
    df: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, Dict[str, object]]]:
    """
    Build every unambiguous physical+state key representation for an exact table.
    State labels must be explicit 9-level identifiers with 1532 repeats each.
    """
    pid_items = profile["physical_candidates"]
    state_items = profile["state_candidates"]
    out = []

    # Limit to high-quality candidates; exact key uniqueness is still verified.
    for pid in pid_items[:4]:
        for st in state_items[:5]:
            k = pd.DataFrame({
                "_physical": df[pid["column"]].astype(str),
                "_state": df[st["column"]].astype(str),
            })
            if k.duplicated(["_physical", "_state"]).any():
                continue
            if len(k) != EXPECTED_ROWS:
                continue
            meta = {
                "physical_col": pid["column"],
                "state_col": st["column"],
            }
            out.append((k, meta))
    return out


def canonical_key_signature(k: pd.DataFrame) -> Tuple[int, int, int]:
    # Stable low-cost signature for prefilter. Exact key-set equality is checked later.
    h = pd.util.hash_pandas_object(
        k.sort_values(["_physical", "_state"]).reset_index(drop=True),
        index=False,
    ).to_numpy(np.uint64)
    return (len(h), int(h.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(h)))


def attach_role_column(
    base_keyed: pd.DataFrame,
    source_profile: Dict[str, object],
    source_df: pd.DataFrame,
    source_key: pd.DataFrame,
    role: str,
) -> Optional[Tuple[pd.Series, Dict[str, object]]]:
    if role == "source_dice":
        item = choose_unique_top(source_profile["source_dice_candidates"])
        if item is None:
            return None
        vals = pd.to_numeric(source_df[item["column"]], errors="coerce")
    elif role == "risk":
        item = choose_unique_top(source_profile["risk_candidates"])
        if item is None:
            return None
        vals = pd.to_numeric(source_df[item["column"]], errors="coerce")
    elif role == "harm":
        item = choose_unique_top(source_profile["harm_candidates"])
        if item is not None:
            vals = to_binary_harm(source_df[item["column"]])
            if vals is None:
                return None
        else:
            item = choose_unique_top(source_profile["delta_dice_candidates"])
            if item is None:
                return None
            delta = pd.to_numeric(source_df[item["column"]], errors="coerce")
            vals = (delta <= HARM_MARGIN).astype(int)
    else:
        raise ValueError(role)

    temp = source_key.copy()
    temp["_value"] = np.asarray(vals)

    merged = base_keyed[["_physical", "_state"]].merge(
        temp,
        on=["_physical", "_state"],
        how="left",
        validate="one_to_one",
    )
    if merged["_value"].isna().any():
        return None

    return merged["_value"], {
        "table": source_profile["path"],
        "column": item["column"],
        "role": role,
    }


def exact_join_candidate(
    profiles: List[Dict[str, object]],
    tables: Dict[str, pd.DataFrame],
) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    """
    Attempt an exact join only when explicit physical+state labels are identical
    across frozen tables. No row-order alignment is ever used.
    """
    keyed = []
    for p in profiles:
        df = tables[p["path"]]
        for k, km in build_key_frame(p, df):
            keyed.append({
                "profile": p,
                "key": k,
                "key_meta": km,
                "sig": canonical_key_signature(k),
            })

    if not keyed:
        return None

    # Group by cheap signature then require exact sorted key equality.
    groups: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for item in keyed:
        groups.setdefault(item["sig"], []).append(item)

    candidates = []
    for _, group in groups.items():
        if len(group) < 2:
            continue

        # Split into exact-equivalence subgroups.
        equiv: List[List[Dict[str, object]]] = []
        for item in group:
            s = item["key"].sort_values(["_physical", "_state"]).reset_index(drop=True)
            placed = False
            for eq in equiv:
                r = eq[0]["key"].sort_values(["_physical", "_state"]).reset_index(drop=True)
                if s.equals(r):
                    eq.append(item)
                    placed = True
                    break
            if not placed:
                equiv.append([item])

        for eq in equiv:
            # Need family plus the three roles across this exact key group.
            # Start from each item that can supply family.
            for base_item in eq:
                p0 = base_item["profile"]
                fam = choose_unique_top(p0["_family_objects"])
                if fam is None:
                    continue

                base = base_item["key"].copy()
                base["family"] = fam["series"].astype(str).to_numpy()
                if base["family"].nunique() != EXPECTED_FAMILIES:
                    continue

                role_sources = {}
                role_values = {}
                ok = True
                for role in ["source_dice", "risk", "harm"]:
                    options = []
                    for item in eq:
                        got = attach_role_column(
                            base,
                            item["profile"],
                            tables[item["profile"]["path"]],
                            item["key"],
                            role,
                        )
                        if got is not None:
                            values, meta = got
                            # Preference: path/name hints and audit relevance.
                            sc = float(item["profile"].get("audit_relevance_score", -1))
                            nm = Path(item["profile"]["path"]).name.lower()
                            if role == "risk" and any(t in nm for t in ("risk", "safety", "score")):
                                sc += 10
                            if role == "source_dice" and "dice" in nm:
                                sc += 5
                            if role == "harm" and any(t in nm for t in ("harm", "outcome", "delta")):
                                sc += 5
                            options.append((sc, values, meta))
                    if not options:
                        ok = False
                        break
                    options.sort(key=lambda z: z[0], reverse=True)
                    top_score = options[0][0]
                    top = [x for x in options if abs(x[0] - top_score) < 1e-12]
                    if len(top) != 1:
                        ok = False
                        break
                    _, values, meta = top[0]
                    role_values[role] = values
                    role_sources[role] = meta

                if not ok:
                    continue

                candidate = pd.DataFrame({
                    "physical_id": base["_physical"].astype(str),
                    "family": base["family"].astype(str),
                    "source_dice": pd.to_numeric(role_values["source_dice"], errors="coerce"),
                    "risk": pd.to_numeric(role_values["risk"], errors="coerce"),
                    "harm": pd.to_numeric(role_values["harm"], errors="coerce").astype(int),
                })

                meta = {
                    "mode": "EXACT_KEY_JOIN",
                    "base_table": p0["path"],
                    "base_physical_col": base_item["key_meta"]["physical_col"],
                    "base_state_col": base_item["key_meta"]["state_col"],
                    "family_column": fam["column"],
                    "role_sources": role_sources,
                }
                candidates.append((candidate, meta))

    # Deduplicate equivalent reconstruction metadata. We still refuse to guess
    # if multiple materially different reconstructions survive.
    if not candidates:
        return None

    valid = []
    for df, meta in candidates:
        try:
            validate_analysis_table(df)
            valid.append((df, meta))
        except Exception:
            continue

    if not valid:
        return None

    # Prefer a reconstruction where risk/source/harm each come from a single,
    # explicit role source and base family is explicit. If multiple remain with
    # identical data, accept the identical data once.
    hashes = []
    for df, meta in valid:
        h = pd.util.hash_pandas_object(
            df.sort_values(["physical_id", "family", "source_dice", "risk", "harm"]).reset_index(drop=True),
            index=False,
        ).to_numpy(np.uint64)
        sig = (len(h), int(h.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(h)))
        hashes.append((sig, df, meta))

    unique_sigs = sorted(set(x[0] for x in hashes))
    if len(unique_sigs) != 1:
        return None

    # All surviving reconstructions produce identical analysis rows.
    return hashes[0][1], hashes[0][2]


def validate_analysis_table(df: pd.DataFrame) -> None:
    required = ["physical_id", "family", "source_dice", "risk", "harm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing analysis columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Rows {len(df)} != {EXPECTED_ROWS}")
    if df["physical_id"].astype(str).nunique() != EXPECTED_PHYSICAL:
        raise RuntimeError("Physical-image count != 1532")
    if df["family"].astype(str).nunique() != EXPECTED_FAMILIES:
        raise RuntimeError("Family count != 3")

    # Each physical image must have exactly 9 model-state rows.
    vc = df["physical_id"].astype(str).value_counts()
    if int(vc.min()) != EXPECTED_STATES or int(vc.max()) != EXPECTED_STATES:
        raise RuntimeError("Each physical image must have exactly 9 rows")

    for c in ["source_dice", "risk"]:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(a).all():
            raise RuntimeError(f"Nonfinite {c}")
        if a.min() < -1e-6 or a.max() > 1.000001:
            raise RuntimeError(f"{c} outside [0,1]")
        if len(np.unique(a)) < 20:
            raise RuntimeError(f"{c} implausibly low variability")

    y = pd.to_numeric(df["harm"], errors="coerce")
    if y.isna().any() or set(y.astype(int).unique().tolist()) != {0, 1}:
        raise RuntimeError("HARM must contain both binary classes")


def resolve_analysis_input(
    profiles: List[Dict[str, object]],
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    singles = []
    for p in profiles:
        got = make_single_table_candidate(p, tables[p["path"]])
        if got is None:
            continue
        df, meta = got
        try:
            validate_analysis_table(df)
            # Prefer source/outcome/risk-named tables and higher audit relevance.
            sc = float(p.get("audit_relevance_score", -1))
            nm = Path(p["path"]).name.lower()
            for tok in ("polypgen", "risk", "safety", "outcome", "tent"):
                if tok in nm:
                    sc += 2
            singles.append((sc, df, meta))
        except Exception:
            continue

    if singles:
        singles.sort(key=lambda x: x[0], reverse=True)
        top_score = singles[0][0]
        top = [x for x in singles if abs(x[0] - top_score) < 1e-12]
        if len(top) == 1:
            return top[0][1], top[0][2]

        # If tied candidates are numerically identical, accept one.
        sigs = []
        for _, df, meta in top:
            z = pd.util.hash_pandas_object(
                df.sort_values(["physical_id", "family", "source_dice", "risk", "harm"]).reset_index(drop=True),
                index=False,
            ).to_numpy(np.uint64)
            sigs.append(((len(z), int(z.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(z))), df, meta))
        if len(set(s[0] for s in sigs)) == 1:
            return sigs[0][1], sigs[0][2]

    joined = exact_join_candidate(profiles, tables)
    if joined is not None:
        return joined

    raise RuntimeError(
        "R16-A fix2 could not prove an exact PolypGen row reconstruction. "
        "No row-order join was attempted. See R16A_FIX2_FORENSIC_PROFILES.json. "
        "GATE=BLOCKED_EXACT_R16A_ALIGNMENT_NOT_PROVEN"
    )


def design(df: pd.DataFrame, include_risk: bool, levels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    parts = [df["z_source_dice"].to_numpy(float).reshape(-1, 1)]
    names = ["z_source_dice"]

    if include_risk:
        parts.append(df["z_risk"].to_numpy(float).reshape(-1, 1))
        names.append("z_risk")

    cat = pd.Categorical(df["family"], categories=list(levels))
    d = pd.get_dummies(cat, drop_first=True, dtype=float)
    if d.shape[1]:
        parts.append(d.to_numpy(float))
        names.extend([f"family[{c}]" for c in d.columns])

    return np.concatenate(parts, axis=1), names


def fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    m = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
    )
    m.fit(X, y)
    return m


def loglik(y: np.ndarray, p: np.ndarray) -> float:
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)
    return float(np.sum(y * np.log(p) + (1-y) * np.log(1-p)))


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def fit_nested(df: pd.DataFrame, levels: Sequence[str]) -> Dict[str, float]:
    y = df["harm"].to_numpy(int)
    X0, _ = design(df, False, levels)
    X1, names1 = design(df, True, levels)

    m0 = fit_lr(X0, y)
    m1 = fit_lr(X1, y)
    p0 = m0.predict_proba(X0)[:, 1]
    p1 = m1.predict_proba(X1)[:, 1]

    ll0 = loglik(y, p0)
    ll1 = loglik(y, p1)
    lr = max(0.0, 2.0 * (ll1 - ll0))
    idx = names1.index("z_risk")

    a0, a1 = safe_auc(y, p0), safe_auc(y, p1)
    r0, r1 = safe_auprc(y, p0), safe_auprc(y, p1)

    return {
        "risk_coefficient": float(m1.coef_.reshape(-1)[idx]),
        "lr_stat": float(lr),
        "lr_p": float(chi2.sf(lr, 1)),
        "auroc_quality": a0,
        "auroc_quality_plus_risk": a1,
        "delta_auroc": a1 - a0,
        "auprc_quality": r0,
        "auprc_quality_plus_risk": r1,
        "delta_auprc": r1 - r0,
        "risk_only_auroc": safe_auc(y, df["risk"].to_numpy(float)),
    }


def cluster_bootstrap(df: pd.DataFrame, reps: int, seed: int, levels: Sequence[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["physical_id"].astype(str).drop_duplicates().to_numpy()
    grouped = {pid: g.copy() for pid, g in df.groupby(df["physical_id"].astype(str), sort=False)}
    rows = []

    iterator = range(reps)
    if tqdm is not None:
        iterator = tqdm(iterator, total=reps, desc="R16-A physical-image bootstrap", unit="rep", dynamic_ncols=True)

    for b in iterator:
        sampled = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([grouped[pid] for pid in sampled], ignore_index=True)
        try:
            r = fit_nested(boot, levels)
            rows.append({
                "rep": b,
                "risk_coefficient": r["risk_coefficient"],
                "delta_auroc": r["delta_auroc"],
                "delta_auprc": r["delta_auprc"],
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def ci(s: pd.Series) -> Tuple[float, float]:
    a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if not len(a):
        return float("nan"), float("nan")
    q = np.percentile(a, [2.5, 97.5])
    return float(q[0]), float(q[1])


def quintile_audit(df: pd.DataFrame) -> pd.DataFrame:
    rank = df["source_dice"].rank(method="first", pct=True)
    q = np.minimum(np.floor(rank * 5).astype(int), 4) + 1
    z = df.copy()
    z["source_dice_quintile"] = q

    rows = []
    for qq, g in z.groupby("source_dice_quintile"):
        y = g["harm"].to_numpy(int)
        s = g["risk"].to_numpy(float)
        rows.append({
            "quintile": int(qq),
            "n_rows": int(len(g)),
            "n_physical": int(g["physical_id"].astype(str).nunique()),
            "source_dice_min": float(g["source_dice"].min()),
            "source_dice_max": float(g["source_dice"].max()),
            "harm_prevalence": float(y.mean()),
            "risk_auroc": safe_auc(y, s),
            "risk_auprc": safe_auprc(y, s),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("===== SAFETTA R16-A SOURCE-QUALITY CONTROL FIX2 =====")
    print("Version:", VERSION)
    print("Resolution strategy: full PolypGen forensic scan + exact-key reconstruction")
    print("Row-order joins allowed: NO")
    print("Expected rows:", EXPECTED_ROWS)
    print("Expected physical images:", EXPECTED_PHYSICAL)
    print("Expected model states/image:", EXPECTED_STATES)
    print("Post-reveal explanatory audit: YES")
    print("SOURCE Dice used for deployment: NO")
    print("SafeTTA refit/recalibration: NO")
    print("Bootstrap reps:", args.bootstrap_reps)
    print()

    if not args.audit_csv.exists():
        raise FileNotFoundError(args.audit_csv)

    audit = pd.read_csv(args.audit_csv, low_memory=False)
    profiles, tables, _ = load_exact_polypgen_profiles(audit, args.out_dir)

    print(f"Exact 13,788-row PolypGen candidate tables found: {len(profiles)}")
    if not profiles:
        raise RuntimeError(
            "No exact 13,788-row PolypGen table exists in the audit manifest. "
            "GATE=BLOCKED_NO_EXACT_POLYPGEN_ROW_TABLE"
        )

    df, provenance = resolve_analysis_input(profiles, tables, args.out_dir)
    validate_analysis_table(df)

    # Keep only columns needed for the explanatory analysis.
    df = df[["physical_id", "family", "source_dice", "risk", "harm"]].copy()
    df["physical_id"] = df["physical_id"].astype(str)
    df["family"] = df["family"].astype(str)
    df["source_dice"] = pd.to_numeric(df["source_dice"], errors="raise")
    df["risk"] = pd.to_numeric(df["risk"], errors="raise")
    df["harm"] = pd.to_numeric(df["harm"], errors="raise").astype(int)

    q_mu = float(df["source_dice"].mean())
    q_sd = float(df["source_dice"].std(ddof=0))
    r_mu = float(df["risk"].mean())
    r_sd = float(df["risk"].std(ddof=0))
    if q_sd <= 0 or r_sd <= 0:
        raise RuntimeError("Zero variance in SOURCE Dice or SafeTTA risk")

    df["z_source_dice"] = (df["source_dice"] - q_mu) / q_sd
    df["z_risk"] = (df["risk"] - r_mu) / r_sd
    levels = sorted(df["family"].unique().tolist())

    main_res = fit_nested(df, levels)
    boot = cluster_bootstrap(df, args.bootstrap_reps, args.seed, levels)
    quint = quintile_audit(df)

    risk_ci = ci(boot["risk_coefficient"]) if len(boot) else (np.nan, np.nan)
    da_ci = ci(boot["delta_auroc"]) if len(boot) else (np.nan, np.nan)
    dp_ci = ci(boot["delta_auprc"]) if len(boot) else (np.nan, np.nan)

    df.to_csv(args.out_dir / "R16A_ANALYSIS_TABLE.csv", index=False)
    boot.to_csv(args.out_dir / "R16A_CLUSTER_BOOTSTRAP.csv", index=False)
    quint.to_csv(args.out_dir / "R16A_SOURCE_DICE_QUINTILES.csv", index=False)
    (args.out_dir / "R16A_INPUT_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "input_provenance": provenance,
        "rows": int(len(df)),
        "physical_images": int(df["physical_id"].nunique()),
        "families": levels,
        "harm_prevalence": float(df["harm"].mean()),
        **main_res,
        "risk_coefficient_ci95": list(risk_ci),
        "delta_auroc_ci95": list(da_ci),
        "delta_auprc_ci95": list(dp_ci),
        "bootstrap_requested": int(args.bootstrap_reps),
        "bootstrap_completed": int(len(boot)),
        "ci_scope": "physical-image uncertainty conditional on frozen model panel",
        "deployment_boundary": "SOURCE Dice used only for post-reveal explanatory audit",
    }
    (args.out_dir / "R16A_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    risk_supported = (
        np.isfinite(risk_ci[0]) and risk_ci[0] > 0
        and main_res["risk_coefficient"] > 0
    )
    auc_supported = np.isfinite(da_ci[0]) and da_ci[0] > 0

    if risk_supported or auc_supported:
        gate = "PASS_R16A_RISK_ADDS_INFORMATION_BEYOND_SOURCE_DICE"
    else:
        gate = "R16A_NO_SUPPORTED_INCREMENT_BEYOND_SOURCE_DICE"

    report = [
        "===== SAFETTA R16-A SOURCE-QUALITY CONTROL SUMMARY =====",
        f"Version: {VERSION}",
        f"Reconstruction mode: {provenance.get('mode')}",
        f"Rows: {len(df)}",
        f"Physical images: {df['physical_id'].nunique()}",
        f"Families: {levels}",
        f"HARM prevalence: {df['harm'].mean():.6f}",
        "",
        f"Quality-only AUROC={main_res['auroc_quality']:.6f} AUPRC={main_res['auprc_quality']:.6f}",
        f"Quality+risk AUROC={main_res['auroc_quality_plus_risk']:.6f} AUPRC={main_res['auprc_quality_plus_risk']:.6f}",
        f"Delta AUROC={main_res['delta_auroc']:.6f} CI95=[{da_ci[0]:.6f}, {da_ci[1]:.6f}]",
        f"Delta AUPRC={main_res['delta_auprc']:.6f} CI95=[{dp_ci[0]:.6f}, {dp_ci[1]:.6f}]",
        f"z(SafeTTA risk) coefficient={main_res['risk_coefficient']:.6f} CI95=[{risk_ci[0]:.6f}, {risk_ci[1]:.6f}]",
        f"Likelihood-ratio stat={main_res['lr_stat']:.6f} p={main_res['lr_p']:.6g}",
        f"Risk-only AUROC={main_res['risk_only_auroc']:.6f}",
        "",
        f"Bootstrap completed: {len(boot)}/{args.bootstrap_reps}",
        "CI scope: physical-image uncertainty conditional on frozen model panel.",
        "SOURCE Dice is post-reveal explanatory only; it is not a deployable SafeTTA input.",
        "",
        f"GATE={gate}",
        "NEXT=COMPLETE_R16B_MATCHED_LEARNED_RELIABILITY_BASELINE",
    ]
    (args.out_dir / "R16A_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print("\nSOURCE-DICE QUINTILES:")
    print(quint.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
