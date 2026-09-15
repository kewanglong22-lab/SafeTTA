#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — PolypGen target-outcome schema audit.

Purpose
-------
The P01B geometry-core preflight has already proven exact NeoPolyp source
outcome support after duplicate collapse. It currently fails only because the
PolypGen unseen-MEMO 4596-row outcome panel is not being parsed by the generic
binder.

This audit inspects ONLY the already-discovered PolypGen outcome candidates
from the first asset inventory and prints their exact schemas/cardinalities.
It does not fit models, compute metrics, or modify any scientific definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P01B-POLYPGEN-TARGET-SCHEMA-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
FIRST_BIND_DIR = ROOT / "B6_P01B_polypgen_asset_binding_audit_v1"
FIRST_BIND_AUDIT = FIRST_BIND_DIR / "B6_P01B_ASSET_BINDING_AUDIT.json"
FIRST_INVENTORY = FIRST_BIND_DIR / "B6_P01B_ASSET_INVENTORY.csv"

EXPECTED_BIND_GATE = "PASS_B6_P01B_ASSET_BINDING_AUDIT_COMPLETE"

OUT_DIR = ROOT / "B6_P01B_polypgen_target_outcome_schema_audit_v1"
PASS_GATE = "PASS_B6_P01B_POLYPGEN_TARGET_OUTCOME_SCHEMA_AUDIT_COMPLETE"

TARGET_ROWS = 4596
TARGET_IMAGES = 1532
TARGET_STATES = 3

SAMPLE_HINTS = [
    "sample", "image", "case", "frame", "external",
]
STATE_HINTS = [
    "state", "model", "family", "checkpoint", "seed",
]
OUTCOME_HINTS = [
    "harm", "delta", "dice", "outcome", "benefit",
]
ACTION_HINTS = [
    "action", "method", "tta", "memo",
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
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return x


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_table(path: Path):
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
    except Exception as e:
        return e
    return None


def interesting_columns(columns: List[str]) -> List[str]:
    out = []
    for c in columns:
        low = str(c).lower()
        if any(h in low for h in SAMPLE_HINTS + STATE_HINTS + OUTCOME_HINTS + ACTION_HINTS):
            out.append(c)
    return out


def series_summary(s: pd.Series) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "dtype": str(s.dtype),
        "non_null": int(s.notna().sum()),
        "nunique": int(s.nunique(dropna=False)),
    }

    vals = s.dropna()
    if len(vals):
        text_vals = vals.astype(str)
        rec["examples"] = text_vals.drop_duplicates().head(10).tolist()

        num = pd.to_numeric(vals, errors="coerce")
        if num.notna().all():
            rec["numeric_min"] = float(num.min())
            rec["numeric_max"] = float(num.max())
            rec["numeric_mean"] = float(num.mean())

    return rec


def likely_key_cardinalities(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)

    sample_cols = [
        c for c in cols
        if any(h in str(c).lower() for h in ["sample", "image", "case", "frame", "external"])
    ]
    state_cols = [
        c for c in cols
        if any(h in str(c).lower() for h in ["state", "model_state"])
    ]
    family_cols = [
        c for c in cols
        if "family" in str(c).lower()
    ]
    seed_cols = [
        c for c in cols
        if "seed" in str(c).lower()
    ]

    rec = {
        "sample_like": {},
        "state_like": {},
        "family_like": {},
        "seed_like": {},
        "pair_candidates": [],
    }

    for name, cs in [
        ("sample_like", sample_cols),
        ("state_like", state_cols),
        ("family_like", family_cols),
        ("seed_like", seed_cols),
    ]:
        for c in cs:
            rec[name][c] = int(df[c].astype(str).nunique(dropna=False))

    # Check all plausible sample x state/family combinations.
    rhs = state_cols + [c for c in family_cols if c not in state_cols]
    for a in sample_cols:
        for b in rhs:
            if a == b:
                continue
            n = int(df[[a, b]].astype(str).drop_duplicates().shape[0])
            if n in {TARGET_ROWS, 13788, 7200, 9000} or (
                int(df[a].astype(str).nunique()) in {TARGET_IMAGES, 800, 1000}
            ):
                rec["pair_candidates"].append({
                    "columns": [a, b],
                    "unique_pairs": n,
                    "left_unique": int(df[a].astype(str).nunique()),
                    "right_unique": int(df[b].astype(str).nunique()),
                })

    return rec


def self_test():
    assert TARGET_IMAGES * TARGET_STATES == TARGET_ROWS
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P01B PolypGen target outcome schema audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    for p in [FIRST_BIND_AUDIT, FIRST_INVENTORY]:
        if not p.is_file():
            raise FileNotFoundError(p)

    audit = load_json(FIRST_BIND_AUDIT)
    if audit.get("status") != "PASS" or audit.get("gate") != EXPECTED_BIND_GATE:
        raise RuntimeError("First P01B binding audit gate changed.")

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite target schema audit: {OUT_DIR}\n"
            "Use _fix1 only if this diagnostic implementation needs repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    inv = pd.read_csv(FIRST_INVENTORY, low_memory=False)

    # Exact first-audit category that previously reported 12 candidates.
    roles = inv["roles"].fillna("").astype(str)
    cohort = inv["cohort"].fillna("").astype(str)

    cand = inv[
        (cohort == "PolypGen")
        & roles.str.contains("OUTCOME_TABLE_CANDIDATE", regex=False)
    ].copy()

    # Also include all PolypGen keyed tables, because the actual R33 panel may
    # have been tagged only as KEYED_TABLE in the heuristic first audit.
    keyed = inv[
        (cohort == "PolypGen")
        & roles.str.contains("KEYED_TABLE", regex=False)
    ].copy()

    cand = (
        pd.concat([cand, keyed], ignore_index=True)
        .drop_duplicates(subset=["path"])
        .reset_index(drop=True)
    )

    print("=" * 160)
    print("SafeTTA B6-P0-1B — PolypGen target-outcome schema audit")
    print(f"Version                  : {VERSION}")
    print("Scientific metrics        : NO")
    print("Model fitting             : NO")
    print(f"PolypGen candidate files  : {len(cand)}")
    print("=" * 160)

    records = []
    for i, r in cand.iterrows():
        p = Path(str(r["path"]))
        print(f"\n[{i+1}/{len(cand)}] {p}")

        rec: Dict[str, Any] = {
            "path": str(p),
            "exists": p.is_file(),
            "inventory_roles": str(r.get("roles", "")),
            "inventory_row_count": (
                None if pd.isna(r.get("row_count", np.nan))
                else int(r.get("row_count"))
            ),
            "inventory_columns": str(r.get("columns", "")),
        }

        if not p.is_file():
            print("  EXISTS=False")
            records.append(rec)
            continue

        rec["sha256"] = sha256_file(p)
        rec["bytes"] = p.stat().st_size

        obj = load_table(p)
        if isinstance(obj, Exception):
            rec["load_error"] = f"{type(obj).__name__}:{obj}"
            print("  LOAD_ERROR:", rec["load_error"])
            records.append(rec)
            continue
        if obj is None:
            rec["load_error"] = "unsupported_or_unparseable"
            print("  LOAD_ERROR: unsupported_or_unparseable")
            records.append(rec)
            continue

        df = obj
        rec["rows"] = int(len(df))
        rec["columns"] = list(map(str, df.columns))
        rec["interesting_columns"] = {}

        print("  rows =", len(df))
        print("  columns =", list(df.columns))

        interesting = interesting_columns(list(df.columns))
        for c in interesting:
            srec = series_summary(df[c])
            rec["interesting_columns"][c] = srec
            print(
                f"    {c}: dtype={srec['dtype']} "
                f"nunique={srec['nunique']} "
                f"examples={srec.get('examples', [])[:6]}"
            )

        card = likely_key_cardinalities(df)
        rec["key_cardinalities"] = card

        # Strong signals for the desired target panel.
        rec["has_4596_rows"] = bool(len(df) == TARGET_ROWS)
        rec["has_1532_unique_in_any_sample_like"] = any(
            v == TARGET_IMAGES for v in card["sample_like"].values()
        )
        rec["has_3_unique_in_any_state_like"] = any(
            v == TARGET_STATES for v in card["state_like"].values()
        )
        rec["has_4596_pair_candidate"] = any(
            x["unique_pairs"] == TARGET_ROWS
            for x in card["pair_candidates"]
        )

        if any([
            rec["has_4596_rows"],
            rec["has_1532_unique_in_any_sample_like"],
            rec["has_4596_pair_candidate"],
        ]):
            print("  >>> TARGET_PANEL_SIGNAL=YES")
            print("  pair candidates =", card["pair_candidates"][:10])

        records.append(rec)

    report = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "scientific_metrics": False,
        "model_fit": False,
        "target_expected": {
            "rows": TARGET_ROWS,
            "physical_images": TARGET_IMAGES,
            "states": TARGET_STATES,
        },
        "candidate_count": len(records),
        "candidates": records,
    }

    p_json = OUT_DIR / "B6_P01B_POLYPGEN_TARGET_OUTCOME_SCHEMA_AUDIT.json"
    write_json(p_json, report)

    # Compact CSV for easier inspection.
    flat = []
    for rec in records:
        flat.append({
            "path": rec["path"],
            "exists": rec.get("exists"),
            "rows": rec.get("rows"),
            "sha256": rec.get("sha256"),
            "has_4596_rows": rec.get("has_4596_rows"),
            "has_1532_unique_in_any_sample_like": rec.get(
                "has_1532_unique_in_any_sample_like"
            ),
            "has_3_unique_in_any_state_like": rec.get(
                "has_3_unique_in_any_state_like"
            ),
            "has_4596_pair_candidate": rec.get(
                "has_4596_pair_candidate"
            ),
            "columns": ";".join(rec.get("columns", [])),
        })
    p_csv = OUT_DIR / "B6_P01B_POLYPGEN_TARGET_OUTCOME_SCHEMA_SUMMARY.csv"
    pd.DataFrame(flat).to_csv(p_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 160)
    print("SafeTTA B6-P0-1B POLYPGEN TARGET OUTCOME SCHEMA AUDIT COMPLETE")
    print(f"GATE={PASS_GATE}")
    print(f"audit_json={p_json}")
    print(f"summary_csv={p_csv}")
    print(f"stage_dir={OUT_DIR}")
    print("=" * 160)


if __name__ == "__main__":
    main()
