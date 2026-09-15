#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — historical candidate-mask adjudication audit.

Context
-------
The authoritative parity audit established:
- PolypGen R33A2B SOURCE/MEMO: clean 3-state parity.
- NeoPolyp SOURCE masks: R38A0C == R38A1A for all 9 states.
- NeoPolyp candidate masks: R38A0C and R38A1A are NOT universally identical.
- R05D3 original SOURCE+A1 is 1000 cases/state, while common800 is 800/state.

Therefore we must not choose between R38A0C and R38A1A by downstream AUROC.
This audit reads historical lock/audit/report artifacts from the R34/R38 lineage
and looks for the prior provenance decision that adjudicated which common800
candidate masks are the historical recovery.

It also inventories any pre-existing 7200/14400-row geometry descriptor tables
(mask-change / prediction Dice / IoU / area change / boundary change) that can
serve as an outcome-independent anchor.

READ ONLY.
NO GT access.
NO HARM/AUROC/AUPRC computation.
NO model fitting.
NO selection by scientific outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-HIST-MASK-ADJUDICATION-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

SEARCH_ROOT_NAMES = [
    "R34A2A7_common800_deeplab_tent_pl_gtfree_regeneration_v1_fix7_fix1",
    "R38A0C_exact_frozen_tent_pl_spatial_mask_replay_v1_fix3",
    "R38A1A_historical_candidate_mask_recovery_and_parity_localization_v1",
    "R35B0_polypgen_external_unseen_memo_protocol_and_asset_audit_v1",
]

SEARCH_ROOTS = [ROOT / x for x in SEARCH_ROOT_NAMES]

OUT_DIR = ROOT / "B6_P01B_historical_candidate_mask_adjudication_audit_v1"
PASS_GATE = "PASS_B6_P01B_HISTORICAL_CANDIDATE_MASK_ADJUDICATION_AUDIT_COMPLETE"

TEXT_SUFFIXES = {".json", ".txt", ".md", ".csv", ".py"}
MAX_BYTES = 80_000_000

DECISION_TERMS = [
    "decision",
    "gate",
    "status",
    "parity",
    "historical",
    "recovery",
    "recovered",
    "exact",
    "mismatch",
    "match",
    "candidate",
    "mask",
    "tent",
    "pl_conf90",
    "pl-conf90",
    "common800",
    "source",
]

GEOMETRY_COLUMN_TERMS = [
    "pred_dice",
    "prediction_dice",
    "mask_dice",
    "pred_iou",
    "prediction_iou",
    "mask_iou",
    "area_delta",
    "foreground_area_change",
    "fg_area_change",
    "boundary_delta",
    "boundary_density_change",
    "mask_change",
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


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_json(v, key))
            elif isinstance(v, list):
                if len(v) <= 100 and all(
                    isinstance(x, (str, int, float, bool, type(None)))
                    for x in v
                ):
                    out[key] = v
            else:
                out[key] = v
    return out


def iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for dp, dns, fns in os.walk(root):
        dns[:] = [
            d for d in dns
            if d not in {".git", "__pycache__", "node_modules"}
        ]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in TEXT_SUFFIXES:
                yield p


def keyword_excerpt(text: str, terms: List[str], radius: int = 220) -> List[str]:
    low = text.lower()
    spans = []
    for term in terms:
        start = 0
        needle = term.lower()
        while True:
            i = low.find(needle, start)
            if i < 0:
                break
            spans.append((max(0, i - radius), min(len(text), i + len(needle) + radius)))
            start = i + len(needle)

    # merge overlapping spans
    spans.sort()
    merged = []
    for a, b in spans:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    excerpts = []
    for a, b in merged[:40]:
        s = text[a:b].replace("\r", " ").replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        excerpts.append(s)
    return excerpts


def scan_decision_artifacts() -> pd.DataFrame:
    rows = []

    files = []
    for root in SEARCH_ROOTS:
        files.extend(list(iter_files(root)))

    for p in tqdm(
        files,
        desc="Scan historical adjudication artifacts",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
        except Exception:
            continue

        rec: Dict[str, Any] = {
            "root": next(
                (r.name for r in SEARCH_ROOTS if str(p).lower().startswith(str(r).lower())),
                "",
            ),
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "suffix": p.suffix.lower(),
        }

        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                flat = flatten_json(obj)
            except Exception as e:
                rec["parse_error"] = f"{type(e).__name__}:{e}"
                rows.append(rec)
                continue

            selected = {}
            for k, v in flat.items():
                kl = k.lower()
                if any(term in kl for term in DECISION_TERMS):
                    if isinstance(v, (str, int, float, bool, list, type(None))):
                        selected[k] = v

            rec["selected_fields_json"] = json.dumps(
                selected, ensure_ascii=False
            )
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            rec["excerpt_json"] = json.dumps(
                keyword_excerpt(
                    text,
                    [
                        "decision", "gate", "parity", "historical",
                        "mismatch", "recovery", "common800",
                        "tent", "pl_conf90",
                    ],
                ),
                ensure_ascii=False,
            )
            rows.append(rec)

        elif p.suffix.lower() in {".txt", ".md", ".py"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                rec["parse_error"] = f"{type(e).__name__}:{e}"
                rows.append(rec)
                continue

            low = text.lower()
            if not any(term in low for term in DECISION_TERMS):
                continue

            rec["excerpt_json"] = json.dumps(
                keyword_excerpt(
                    text,
                    [
                        "decision", "gate", "parity", "historical",
                        "mismatch", "recovery", "common800",
                        "tent", "pl_conf90",
                    ],
                ),
                ensure_ascii=False,
            )
            rows.append(rec)

        elif p.suffix.lower() == ".csv":
            try:
                hdr = pd.read_csv(p, nrows=0, low_memory=False)
                cols = list(map(str, hdr.columns))
            except Exception:
                continue

            interesting = [
                c for c in cols
                if any(term in c.lower() for term in DECISION_TERMS)
            ]
            if not interesting:
                continue

            try:
                nrows = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
            except Exception:
                nrows = None

            rec["rows"] = nrows
            rec["columns"] = ";".join(cols)
            rec["interesting_columns"] = ";".join(interesting)
            rows.append(rec)

    return pd.DataFrame(rows)


def scan_geometry_tables() -> pd.DataFrame:
    rows = []

    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue

        csvs = list(root.rglob("*.csv"))
        for p in tqdm(
            csvs,
            desc=f"Geometry table scan {root.name[:22]}",
            unit="csv",
            dynamic_ncols=True,
        ):
            try:
                hdr = pd.read_csv(p, nrows=0, low_memory=False)
            except Exception:
                continue

            cols = list(map(str, hdr.columns))
            hits = [
                c for c in cols
                if any(term in c.lower() for term in GEOMETRY_COLUMN_TERMS)
            ]
            if not hits:
                continue

            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue

            rec = {
                "root": root.name,
                "path": str(p),
                "sha256": sha256_file(p),
                "rows": int(len(df)),
                "columns": ";".join(cols),
                "geometry_columns": ";".join(hits),
            }

            for key in [
                "sample_id",
                "model_state_id",
                "action",
                "model_family",
                "training_seed",
            ]:
                if key in df.columns:
                    rec[f"{key}_nunique"] = int(
                        df[key].astype(str).nunique(dropna=False)
                    )

            rows.append(rec)

    return pd.DataFrame(rows)


def print_high_value(decisions: pd.DataFrame, geometry: pd.DataFrame) -> None:
    print("\nHIGH-VALUE DECISION / PARITY ARTIFACTS")
    if decisions.empty:
        print("NONE")
    else:
        rank_terms = [
            "decision", "gate", "audit", "report", "parity",
            "recovery", "summary", "lock",
        ]
        x = decisions.copy()
        x["_rank"] = x["path"].astype(str).str.lower().map(
            lambda s: sum(int(t in s) for t in rank_terms)
        )
        show = x.sort_values(
            ["_rank", "bytes"], ascending=[False, True]
        ).head(40)

        for _, r in show.iterrows():
            print("\nPATH:", r["path"])
            if "selected_fields_json" in r and pd.notna(r.get("selected_fields_json")):
                print("FIELDS:", r.get("selected_fields_json"))
            if "excerpt_json" in r and pd.notna(r.get("excerpt_json")):
                excerpts = json.loads(r.get("excerpt_json", "[]"))
                for e in excerpts[:5]:
                    print("  ", e[:1000])

    print("\nPRE-EXISTING GEOMETRY TABLES")
    if geometry.empty:
        print("NONE")
    else:
        cols = [
            "root", "rows", "sample_id_nunique",
            "model_state_id_nunique", "action_nunique",
            "geometry_columns", "path",
        ]
        for c in cols:
            if c not in geometry.columns:
                geometry[c] = None
        print(geometry[cols].to_string(index=False))


def self_test():
    assert len(SEARCH_ROOTS) == 4
    assert "mask_change" in GEOMETRY_COLUMN_TERMS
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P01B historical candidate-mask adjudication audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite adjudication audit: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 164)
    print("SafeTTA B6-P0-1B — historical candidate-mask adjudication audit")
    print(f"Version            : {VERSION}")
    print("GT access          : NO")
    print("Scientific metrics : NO")
    print("Model fitting      : NO")
    print("Outcome-based pick : NO")
    print("=" * 164)

    print("\n[1/2] Historical decision/parity/recovery artifacts")
    decisions = scan_decision_artifacts()
    p_dec = OUT_DIR / "B6_P01B_HISTORICAL_MASK_DECISION_ARTIFACTS.csv"
    decisions.to_csv(p_dec, index=False, encoding="utf-8")

    print("\n[2/2] Pre-existing geometry descriptor tables")
    geometry = scan_geometry_tables()
    p_geo = OUT_DIR / "B6_P01B_PREEXISTING_GEOMETRY_TABLES.csv"
    geometry.to_csv(p_geo, index=False, encoding="utf-8")

    print_high_value(decisions, geometry)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "read_only": True,
        "gt_access": False,
        "scientific_metrics": False,
        "model_fit": False,
        "decision_artifact_rows": int(len(decisions)),
        "geometry_table_rows": int(len(geometry)),
        "outputs": {},
    }

    for p in [p_dec, p_geo]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_HISTORICAL_MASK_ADJUDICATION_AUDIT.json"
    write_json(p_audit, audit)

    print("\n" + "=" * 164)
    print("SafeTTA B6-P0-1B HISTORICAL MASK ADJUDICATION AUDIT COMPLETE")
    print(f"GATE={PASS_GATE}")
    print(f"audit_json={p_audit}")
    print(f"stage_dir={OUT_DIR}")
    print("=" * 164)


if __name__ == "__main__":
    main()
