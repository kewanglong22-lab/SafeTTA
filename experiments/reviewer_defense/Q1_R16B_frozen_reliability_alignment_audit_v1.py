#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R16B_frozen_reliability_alignment_audit_v1.py

SafeTTA R16-B pre-adaptation reliability-input alignment audit.

Purpose
-------
Determine exactly how the matched learned current-reliability baseline can be
constructed from retained, frozen PRE-ADAPTATION SOURCE outputs without target
GT leakage or approximate surrogate reconstruction.

This is a READ-ONLY audit. It does not train a model.

It consumes the prior R16 input-audit manifest and inspects the strongest
NeoPolyp, PolypGen, and SUN-SEG candidate tables.

It distinguishes:
  DIRECT_SUMMARIES
    Existing entropy/confidence reliability summaries are already retained.
  VALID_ARRAY_REFERENCES
    Table contains resolvable paths to retained SOURCE probability/logit arrays.
  INLINE_SCALAR_ONLY
    A numeric 'probability' column is only a scalar score, not a probability map.
  UNRESOLVED_REFERENCES
    Path-like probability/logit references cannot be resolved.
  NO_RELIABILITY_INPUT
    No usable conventional reliability input is retained.

Expected frozen panel sizes:
  NeoPolyp: 9000 rows / 1000 physical images
  PolypGen: 13788 rows / 1532 physical images
  SUN-SEG: 8820 rows / 980 frames, 49 physical cases (case ID may differ from frame ID)

The audit refuses to use:
- target GT;
- target DeltaDice/HARM to construct input features;
- post-adaptation outputs;
- target-derived normalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-08-Q1-R16B-FROZEN-RELIABILITY-ALIGNMENT-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_AUDIT = DEFAULT_ROOT / r"outputs\Q1_R16A_reviewer_defense_input_audit_v1\CANDIDATE_TABLES.csv"
DEFAULT_OUT = DEFAULT_ROOT / r"outputs\Q1_R16B_frozen_reliability_alignment_audit_v1"

EXPECTED = {
    "NEOPOLYP": {"rows": 9000, "physical": 1000},
    "POLYPGEN": {"rows": 13788, "physical": 1532},
    "SUNSEG": {"rows": 8820, "physical": None},
}

DIRECT_ROLE_COLS = ["role_entropy", "role_confidence"]
PATH_ROLE_COL = "role_probability"
OUTCOME_ROLE_COLS = ["role_harm", "role_delta_dice"]

PATH_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".pkl", ".joblib", ".png", ".tif", ".tiff", ".mha", ".nii", ".gz"}

SKIP_PATH_TOKENS = [
    "adapt", "tent", "plconf", "pl_conf", "tta_", "post_", "after_",
]


def parse_role(v) -> List[str]:
    if pd.isna(v):
        return []
    return [x for x in str(v).split("|") if x]


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


def norm(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(s).lower())).strip("_")


def path_like_fraction(series: pd.Series) -> float:
    s = series.dropna().astype(str).head(500)
    if not len(s):
        return 0.0
    hits = 0
    for x in s:
        xl = x.lower()
        suffix_hit = any(xl.endswith(suf) for suf in PATH_SUFFIXES)
        sep_hit = ("\\" in x) or ("/" in x)
        if suffix_hit or sep_hit:
            hits += 1
    return hits / len(s)


def resolve_one(raw: str, table_path: Path, root: Path) -> Path | None:
    p = Path(str(raw))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            table_path.parent / p,
            root / p,
        ])
    for c in candidates:
        try:
            if c.exists():
                return c
        except OSError:
            pass
    return None


def inspect_array(path: Path) -> Dict[str, object]:
    out = {"sample_path": str(path), "suffix": path.suffix.lower()}
    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r")
            out["shape"] = list(a.shape)
            out["dtype"] = str(a.dtype)
            out["kind"] = "ARRAY"
        elif path.suffix.lower() == ".npz":
            z = np.load(path)
            keys = list(z.files)
            out["keys"] = keys[:20]
            if keys:
                a = z[keys[0]]
                out["shape"] = list(a.shape)
                out["dtype"] = str(a.dtype)
            out["kind"] = "NPZ"
        else:
            out["kind"] = "EXISTS_NONNUMPY"
    except Exception as e:
        out["kind"] = "INSPECTION_ERROR"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def candidate_subset(audit: pd.DataFrame, dataset: str) -> pd.DataFrame:
    x = audit[audit["dataset_guess"].astype(str).str.upper().eq(dataset)].copy()
    if "learned_reliability_ready" in x.columns:
        ready = x["learned_reliability_ready"].astype(str).str.lower().isin(["true", "1", "yes"])
        x = x[ready].copy()
    if "relevance_score" in x.columns:
        x["relevance_score"] = pd.to_numeric(x["relevance_score"], errors="coerce").fillna(-1)
        x = x.sort_values("relevance_score", ascending=False)
    return x.head(25)


def find_physical_col(row: pd.Series, df: pd.DataFrame) -> str | None:
    role = parse_role(row.get("role_physical_id", ""))
    preferred = ["sample_id", "image_id", "physical_id", "case_id", "patient_id", "frame_id"]
    nmap = {norm(c): c for c in df.columns}
    for p in preferred:
        if norm(p) in nmap:
            return nmap[norm(p)]
    for c in role:
        if c in df.columns:
            return c
    return None


def evaluate_candidate(row: pd.Series, root: Path) -> Dict[str, object]:
    table_path = Path(str(row["path"]))
    rec: Dict[str, object] = {
        "dataset": str(row.get("dataset_guess", "UNKNOWN")).upper(),
        "table_path": str(table_path),
        "relevance_score": float(row.get("relevance_score", -1)),
        "classification": "UNASSESSED",
    }

    if not table_path.exists():
        rec["classification"] = "MISSING_TABLE"
        return rec

    try:
        df = read_any(table_path)
    except Exception as e:
        rec["classification"] = "READ_ERROR"
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    rec["rows"] = int(len(df))
    expected = EXPECTED.get(rec["dataset"], {})
    if expected.get("rows") is not None:
        rec["expected_rows_match"] = bool(len(df) == expected["rows"])

    pid_col = find_physical_col(row, df)
    rec["physical_id_col"] = pid_col
    if pid_col:
        rec["unique_physical"] = int(df[pid_col].astype(str).nunique(dropna=True))

    direct_cols = []
    for role_col in DIRECT_ROLE_COLS:
        direct_cols.extend([c for c in parse_role(row.get(role_col, "")) if c in df.columns])
    direct_cols = sorted(set(direct_cols))
    rec["direct_summary_cols"] = "|".join(direct_cols)

    outcome_cols = []
    for role_col in OUTCOME_ROLE_COLS:
        outcome_cols.extend([c for c in parse_role(row.get(role_col, "")) if c in df.columns])
    rec["outcome_cols_present_for_evaluation_only"] = "|".join(sorted(set(outcome_cols)))

    prob_cols = [c for c in parse_role(row.get(PATH_ROLE_COL, "")) if c in df.columns]
    rec["probability_role_cols"] = "|".join(prob_cols)

    # Direct summary route is preferred if conventional metrics truly exist.
    if direct_cols:
        numeric_direct = []
        for c in direct_cols:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() > 0.95:
                numeric_direct.append(c)
        if numeric_direct:
            rec["numeric_direct_summary_cols"] = "|".join(numeric_direct)
            rec["classification"] = "DIRECT_SUMMARIES"
            return rec

    # Otherwise inspect probability/logit role columns.
    best_path_col = None
    best_frac = 0.0
    for c in prob_cols:
        frac = path_like_fraction(df[c])
        rec[f"path_like_fraction__{c}"] = frac
        if frac > best_frac:
            best_frac = frac
            best_path_col = c

    if best_path_col is None:
        # If numeric scalar probability-like columns exist, explicitly reject them as maps.
        numeric = []
        for c in prob_cols:
            arr = pd.to_numeric(df[c], errors="coerce")
            if arr.notna().mean() > 0.95:
                numeric.append(c)
        if numeric:
            rec["numeric_probability_like_cols"] = "|".join(numeric)
            rec["classification"] = "INLINE_SCALAR_ONLY"
        else:
            rec["classification"] = "NO_RELIABILITY_INPUT"
        return rec

    vals = df[best_path_col].dropna().astype(str).drop_duplicates().head(200).tolist()
    resolved = []
    unsafe = []
    for x in vals:
        xl = x.lower()
        if any(tok in xl for tok in SKIP_PATH_TOKENS):
            unsafe.append(x)
            continue
        p = resolve_one(x, table_path, root)
        if p is not None:
            resolved.append(p)

    rec["chosen_path_col"] = best_path_col
    rec["sampled_unique_refs"] = len(vals)
    rec["resolved_safe_refs"] = len(resolved)
    rec["postadapt_suspect_refs_skipped"] = len(unsafe)
    rec["resolved_fraction"] = (len(resolved) / max(1, len(vals)))

    if resolved and rec["resolved_fraction"] >= 0.90:
        rec["classification"] = "VALID_ARRAY_REFERENCES"
        rec["sample_array_inspection"] = json.dumps(inspect_array(resolved[0]), ensure_ascii=False)
    else:
        rec["classification"] = "UNRESOLVED_REFERENCES"

    return rec


def choose_route(records: List[Dict[str, object]]) -> Tuple[str, List[str]]:
    notes = []
    by_ds = {}
    for ds in EXPECTED:
        by_ds[ds] = [r for r in records if r.get("dataset") == ds]

    def has(ds: str, cls: str) -> bool:
        return any(r.get("classification") == cls and r.get("expected_rows_match", True) for r in by_ds.get(ds, []))

    source_direct = has("NEOPOLYP", "DIRECT_SUMMARIES")
    ext_direct = has("POLYPGEN", "DIRECT_SUMMARIES")
    source_arr = has("NEOPOLYP", "VALID_ARRAY_REFERENCES")
    ext_arr = has("POLYPGEN", "VALID_ARRAY_REFERENCES")

    notes.append(f"NeoPolyp direct summaries: {source_direct}")
    notes.append(f"PolypGen direct summaries: {ext_direct}")
    notes.append(f"NeoPolyp valid SOURCE array refs: {source_arr}")
    notes.append(f"PolypGen valid SOURCE array refs: {ext_arr}")

    if source_direct and ext_direct:
        gate = "READY_FOR_R16B_DIRECT_SUMMARY_IMPLEMENTATION"
    elif source_arr and ext_arr:
        gate = "READY_FOR_R16B_PROBABILITY_RECONSTRUCTION"
    elif (source_direct or source_arr) and (ext_direct or ext_arr):
        gate = "READY_FOR_R16B_MIXED_FROZEN_INPUT_IMPLEMENTATION"
    else:
        gate = "BLOCKED_R16B_EXACT_FROZEN_ALIGNMENT_NOT_ESTABLISHED"

    return gate, notes


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("===== SAFETTA R16-B FROZEN RELIABILITY ALIGNMENT AUDIT =====")
    print("Version:", VERSION)
    print("Read-only audit: YES")
    print("Model training: NO")
    print("Target GT used for feature construction: NO")
    print()

    if not args.audit_csv.exists():
        raise FileNotFoundError(args.audit_csv)

    audit = pd.read_csv(args.audit_csv, low_memory=False)
    records = []

    subsets = []
    for ds in ["NEOPOLYP", "POLYPGEN", "SUNSEG"]:
        x = candidate_subset(audit, ds)
        for _, row in x.iterrows():
            subsets.append(row)

    iterator = subsets
    if tqdm is not None:
        iterator = tqdm(subsets, desc="Inspect R16-B candidates", unit="table", dynamic_ncols=True)

    for row in iterator:
        records.append(evaluate_candidate(row, args.root))

    records.sort(
        key=lambda r: (
            r.get("dataset", ""),
            {"DIRECT_SUMMARIES": 3, "VALID_ARRAY_REFERENCES": 2, "INLINE_SCALAR_ONLY": 1}.get(r.get("classification"), 0),
            float(r.get("relevance_score", -1)),
        ),
        reverse=True,
    )

    gate, notes = choose_route(records)
    write_csv(args.out_dir / "R16B_ALIGNMENT_CANDIDATES.csv", records)

    summary = {
        "version": VERSION,
        "gate": gate,
        "notes": notes,
        "records": records[:30],
        "leakage_lock": {
            "target_gt_for_features": False,
            "target_delta_dice_for_features": False,
            "post_adaptation_outputs": False,
            "target_normalization": False,
        },
    }
    (args.out_dir / "R16B_ALIGNMENT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "===== SAFETTA R16-B FROZEN RELIABILITY ALIGNMENT SUMMARY =====",
        f"Version: {VERSION}",
        *notes,
        "",
        f"GATE={gate}",
    ]
    if gate == "READY_FOR_R16B_DIRECT_SUMMARY_IMPLEMENTATION":
        nxt = "NEXT=BUILD_R16B_MATCHED_LR_FROM_RETAINED_DIRECT_SUMMARIES"
    elif gate == "READY_FOR_R16B_PROBABILITY_RECONSTRUCTION":
        nxt = "NEXT=BUILD_R16B_FEATURE_RECONSTRUCTION_FROM_FROZEN_SOURCE_PROBABILITIES"
    elif gate == "READY_FOR_R16B_MIXED_FROZEN_INPUT_IMPLEMENTATION":
        nxt = "NEXT=BUILD_R16B_MIXED_INPUT_EXECUTION_WITH_IDENTICAL_FEATURE_DEFINITIONS"
    else:
        nxt = "NEXT=DO_NOT_RECONSTRUCT_FROM_GT_OR_APPROXIMATIONS; REVIEW_ALIGNMENT_CANDIDATES"
    report.append(nxt)
    report.append("")
    report.append("TOP RECORDS:")
    for r in records[:20]:
        report.extend([
            f"- {r.get('dataset')} | {r.get('classification')} | rows={r.get('rows')}",
            f"  table={r.get('table_path')}",
            f"  direct={r.get('direct_summary_cols','')}",
            f"  probability_cols={r.get('probability_role_cols','')}",
            f"  chosen_path_col={r.get('chosen_path_col','')}",
            f"  resolved_fraction={r.get('resolved_fraction','')}",
        ])

    (args.out_dir / "R16B_ALIGNMENT_REPORT.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("\n".join(report[:8]))
    print(nxt)
    print()
    print("Report:", args.out_dir / "R16B_ALIGNMENT_REPORT.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
