#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32B0
Published-Reliability Baseline Asset + Comparability Audit

Purpose
-------
Before extending the published reliability baselines (SicTTA-CCD,
TEGDA-ADIC, MC-dropout) to the new three-action R31B/R32A panel, audit
what exact frozen implementations / score tables / provenance files already
exist locally and which segmentation states are supportable.

This stage DOES NOT:
  - run CCD / ADIC / MC-dropout inference
  - train any model
  - join scores to new HARM outcomes
  - tune score direction
  - select a favorable subset
  - access external cohorts

It only:
  1) binds the completed R32A1-fix1 result;
  2) inventories the exact new 800-case x 9-state x 3-action panel;
  3) discovers retained R17A scripts/tables/provenance by deterministic
     filename/content signatures;
  4) records CSV schemas for candidate row-level score assets;
  5) identifies which frozen model families/states occur in the new panel.

R17A guardrails retained
------------------------
The prior publication audit explicitly required:
  * no target recalibration;
  * no target threshold tuning;
  * no feature selection;
  * no post-hoc score reversal;
  * PraNet ADIC excluded when faithful native nn.Dropout support is absent;
  * regenerated SegFormer CCD states must not be mixed with historical HARM
    labels when SOURCE-state identity is inconsistent.

R32B will preserve those restrictions. R32B0 does not decide the final common
subset; that is frozen only after this asset audit reports support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-12-R32B0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R32A1_SCRIPT = CODE / "Q1_R32A1_three_action_loao_comparison_execution_v1_fix1.py"
EXPECTED_R32A1_SCRIPT_SHA256 = (
    "b618729ab39b83f0afc9d26c8b5e2163742690c127bf0c3a60bbbc596dfa3a78"
)

R32A1_DIR = ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
R32A1_FINAL = R32A1_DIR / "R32A1_FIX1_FINAL_LOCK.json"
EXPECTED_R32A1_FINAL_SHA256 = (
    "0822cf8c6db035e836dcc6d3f67ae4b0e9d58622e72ba1086edf60fa90e8873b"
)
R32A1_OOF = R32A1_DIR / "R32A1_FIX1_OOF_PREDICTIONS.csv"
R32A1_MACRO = R32A1_DIR / "R32A1_FIX1_MACRO_SUMMARY.csv"

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)

DEFAULT_OUT = ROOT / "R32B0_published_reliability_asset_comparability_audit_v1"

EXPECTED_R32A1_STATUS = (
    "PASS_R32A1_FIX1_THREE_ACTION_LOAO_COMPARISON_COMPLETE"
)

R17A_POINT_ANCHORS = {
    "SafeTTA_pooled_AUROC": "0.616419085",
    "SicTTA_CCD_pooled_AUROC": "0.634544123",
    "TEGDA_ADIC_pooled_AUROC": "0.602545392",
    "MC_dropout_pooled_AUROC": "0.446858833",
}

KEYWORDS = (
    "r17a",
    "sict",
    "ccd",
    "tegd",
    "adic",
    "mc-drop",
    "mc_dropout",
    "mcdrop",
    "published_reliability",
    "published-reliability",
    "reliability_baseline",
    "reliability-baseline",
)

TEXT_EXTENSIONS = {
    ".py", ".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".tsv"
}

CONTENT_SCAN_MAX_BYTES = 4 * 1024 * 1024
CSV_HEADER_MAX_BYTES = 128 * 1024 * 1024

PRUNE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".idea",
    ".pytest_cache",
    "node_modules",
}

HEAVY_DATA_DIR_NAMES = {
    "images",
    "image",
    "masks",
    "mask",
    "datasets",
    "dataset",
    "checkpoints",
    "checkpoint",
    "weights",
}

PREFERRED_DIR_TOKENS = (
    "code",
    "experiment",
    "output",
    "provenance",
    "public",
    "repro",
    "release",
    "legacy",
    "r17",
)

SCORE_COLUMN_TOKENS = (
    "score",
    "risk",
    "uncert",
    "entropy",
    "ccd",
    "adic",
    "dropout",
    "confidence",
    "quality",
)

ID_COLUMN_TOKENS = (
    "sample_id",
    "image_id",
    "image",
    "case_id",
    "model_state",
    "model_state_id",
    "state",
    "family",
    "method",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\nexpected={expected}\n"
            f"observed={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def verify_upstream() -> Dict[str, Any]:
    require_sha(R32A1_SCRIPT, EXPECTED_R32A1_SCRIPT_SHA256, "R32A1 fix1 script")
    require_sha(R32A1_FINAL, EXPECTED_R32A1_FINAL_SHA256, "R32A1 fix1 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")

    final = json.loads(R32A1_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != EXPECTED_R32A1_STATUS:
        raise RuntimeError(
            f"R32A1 status={final.get('status')!r}; "
            f"expected={EXPECTED_R32A1_STATUS!r}"
        )
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R32A1 reports external data access.")
    if bool(final.get("hyperparameter_search", True)):
        raise RuntimeError("R32A1 reports hyperparameter search.")

    for p in [R32A1_OOF, R32A1_MACRO, R31B3_TABLE]:
        if not p.is_file():
            raise FileNotFoundError(p)

    if sha256_file(R32A1_OOF) != str(final.get("OOF_predictions_sha256")):
        raise RuntimeError("R32A1 OOF SHA changed.")
    if sha256_file(R32A1_MACRO) != str(final.get("macro_summary_sha256")):
        raise RuntimeError("R32A1 macro-summary SHA changed.")

    return final


def panel_inventory() -> tuple[pd.DataFrame, Dict[str, Any]]:
    d = pd.read_csv(
        R31B3_TABLE,
        usecols=[
            "sample_id",
            "model_state_id",
            "model_family",
            "action",
            "r31b3_harm_label",
        ],
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
            "action": str,
        },
        low_memory=False,
    )

    state = (
        d[
            [
                "model_state_id",
                "model_family",
            ]
        ]
        .drop_duplicates()
        .sort_values(["model_family", "model_state_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    counts = (
        d.groupby(
            ["model_family", "model_state_id", "action"],
            as_index=False,
        )
        .agg(
            rows=("sample_id", "size"),
            physical_cases=("sample_id", "nunique"),
            harm_count=("r31b3_harm_label", "sum"),
            harm_rate=("r31b3_harm_label", "mean"),
        )
        .sort_values(
            ["model_family", "model_state_id", "action"],
            kind="mergesort",
        )
    )

    info = {
        "physical_cases": int(d["sample_id"].nunique()),
        "model_states": int(d["model_state_id"].nunique()),
        "model_families": sorted(d["model_family"].unique().tolist()),
        "actions": sorted(d["action"].unique().tolist()),
        "rows": int(len(d)),
    }

    return counts, info


def candidate_scan_roots(root: Path) -> List[Path]:
    roots: List[Path] = []

    for name in [
        "code",
        "experiments",
        "outputs",
        "provenance",
        "legacy_all_retained_code",
        "method_core",
    ]:
        p = root / name
        if p.is_dir():
            roots.append(p)

    # Add immediate child dirs likely to contain public/repro/R17 assets.
    if root.is_dir():
        for p in root.iterdir():
            if not p.is_dir():
                continue
            low = p.name.lower()
            if any(tok in low for tok in PREFERRED_DIR_TOKENS):
                roots.append(p)

    # De-duplicate resolved paths.
    uniq = []
    seen = set()
    for p in roots:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(rp)

    return uniq


def walk_candidate_files(scan_roots: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []

    for root in scan_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Never descend into obvious non-source heavy trees.
            pruned = []
            for name in dirnames:
                low = name.lower()
                if low in PRUNE_DIR_NAMES:
                    continue
                if low in HEAVY_DATA_DIR_NAMES:
                    continue
                pruned.append(name)
            dirnames[:] = pruned

            for filename in filenames:
                p = Path(dirpath) / filename
                if p.suffix.lower() not in TEXT_EXTENSIONS:
                    continue
                paths.append(p)

    # Exact de-duplication.
    out = []
    seen = set()
    for p in paths:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def text_matches(path: Path) -> tuple[List[str], List[str]]:
    name_low = path.name.lower()
    filename_hits = [k for k in KEYWORDS if k in name_low]

    content_hits: List[str] = []
    try:
        size = path.stat().st_size
    except Exception:
        return filename_hits, content_hits

    if size > CONTENT_SCAN_MAX_BYTES:
        return filename_hits, content_hits

    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return filename_hits, content_hits

    for k in KEYWORDS:
        if k in text:
            content_hits.append(k)

    for label, anchor in R17A_POINT_ANCHORS.items():
        if anchor in text:
            content_hits.append(f"anchor:{label}")

    return sorted(set(filename_hits)), sorted(set(content_hits))


def csv_schema_info(path: Path) -> Dict[str, Any]:
    result = {
        "csv_header_read": False,
        "columns": "",
        "score_like_columns": "",
        "id_like_columns": "",
        "row_level_score_candidate": False,
    }

    try:
        if path.stat().st_size > CSV_HEADER_MAX_BYTES:
            return result

        df = pd.read_csv(path, nrows=5, low_memory=False)
        cols = [str(c) for c in df.columns]
        lowcols = [c.lower() for c in cols]

        score_cols = [
            c for c, lc in zip(cols, lowcols)
            if any(tok in lc for tok in SCORE_COLUMN_TOKENS)
        ]
        id_cols = [
            c for c, lc in zip(cols, lowcols)
            if any(tok in lc for tok in ID_COLUMN_TOKENS)
        ]

        result.update({
            "csv_header_read": True,
            "columns": " | ".join(cols),
            "score_like_columns": " | ".join(score_cols),
            "id_like_columns": " | ".join(id_cols),
            "row_level_score_candidate": bool(score_cols and id_cols),
        })
    except Exception:
        pass

    return result


def discover_assets(scan_roots: Sequence[Path]) -> pd.DataFrame:
    files = walk_candidate_files(scan_roots)
    rows = []

    for p in tqdm(
        files,
        desc="R32B0 retained R17A asset audit",
        unit="file",
        dynamic_ncols=True,
    ):
        fhits, chits = text_matches(p)
        if not fhits and not chits:
            continue

        try:
            rel = str(p.relative_to(ROOT))
        except Exception:
            rel = str(p)

        try:
            size = int(p.stat().st_size)
        except Exception:
            size = -1

        row = {
            "path": str(p),
            "relative_path": rel,
            "suffix": p.suffix.lower(),
            "bytes": size,
            "filename_hits": " | ".join(fhits),
            "content_hits": " | ".join(chits),
            "sha256": sha256_file(p) if size >= 0 else "",
        }

        if p.suffix.lower() == ".csv":
            row.update(csv_schema_info(p))
        else:
            row.update({
                "csv_header_read": False,
                "columns": "",
                "score_like_columns": "",
                "id_like_columns": "",
                "row_level_score_candidate": False,
            })

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "relative_path",
                "suffix",
                "bytes",
                "filename_hits",
                "content_hits",
                "sha256",
                "csv_header_read",
                "columns",
                "score_like_columns",
                "id_like_columns",
                "row_level_score_candidate",
            ]
        )

    out = pd.DataFrame(rows)
    out["priority"] = (
        out["row_level_score_candidate"].astype(int) * 100
        + out["suffix"].eq(".py").astype(int) * 20
        + out["content_hits"].str.contains("anchor:", regex=False).astype(int) * 10
        + out["filename_hits"].ne("").astype(int) * 5
    )
    out = out.sort_values(
        ["priority", "relative_path"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def support_summary(assets: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "SicTTA-CCD": ("sict", "ccd"),
        "TEGDA-ADIC": ("tegd", "adic"),
        "MC-dropout": ("mc-drop", "mc_dropout", "mcdrop"),
        "R17A-general": ("r17a", "published_reliability", "published-reliability"),
    }

    rows = []
    for method, tokens in methods.items():
        if len(assets) == 0:
            g = assets
        else:
            joined = (
                assets["relative_path"].astype(str)
                + " "
                + assets["filename_hits"].astype(str)
                + " "
                + assets["content_hits"].astype(str)
                + " "
                + assets["columns"].astype(str)
            ).str.lower()
            mask = pd.Series(False, index=assets.index)
            for tok in tokens:
                mask = mask | joined.str.contains(re.escape(tok.lower()), regex=True)
            g = assets[mask]

        rows.append({
            "method": method,
            "matching_assets": int(len(g)),
            "python_scripts": int((g["suffix"] == ".py").sum()) if len(g) else 0,
            "csv_assets": int((g["suffix"] == ".csv").sum()) if len(g) else 0,
            "row_level_score_candidates": (
                int(g["row_level_score_candidate"].sum()) if len(g) else 0
            ),
        })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.root.resolve() != ROOT.resolve():
        raise RuntimeError("--root override is intentionally disabled in R32B0-v1; use the frozen project root.")

    print("=" * 124)
    print("SafeTTA R32B0 Published-Reliability Asset + Comparability Audit")
    print("Version                     :", VERSION)
    print("CCD / ADIC / MC inference   : NO")
    print("Training                    : NO")
    print("Score/outcome joining       : NO")
    print("Target recalibration        : NO")
    print("Post-hoc score reversal     : NO")
    print("External cohort access      : NO")
    print("=" * 124)

    final = verify_upstream()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32B0 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    panel_counts, panel_info = panel_inventory()

    scan_roots = candidate_scan_roots(ROOT)
    assets = discover_assets(scan_roots)
    support = support_summary(assets)

    panel_path = out / "R32B0_NEW_PANEL_STATE_ACTION_INVENTORY.csv"
    assets_path = out / "R32B0_R17A_RETAINED_ASSET_INVENTORY.csv"
    support_path = out / "R32B0_BASELINE_ASSET_SUPPORT_SUMMARY.csv"

    atomic_csv(panel_counts, panel_path)
    atomic_csv(assets, assets_path)
    atomic_csv(support, support_path)

    row_score_candidates = (
        assets[assets["row_level_score_candidate"].astype(bool)]
        if len(assets)
        else assets
    )
    script_candidates = (
        assets[assets["suffix"] == ".py"]
        if len(assets)
        else assets
    )

    audit = {
        "status": "PASS_R32B0_ASSET_COMPARABILITY_AUDIT_COMPLETE",
        "version": VERSION,
        "R32A1_fix1_script_sha256": EXPECTED_R32A1_SCRIPT_SHA256,
        "R32A1_fix1_final_sha256": EXPECTED_R32A1_FINAL_SHA256,
        "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        "R32A1_primary_model": final.get("primary_model"),
        "panel": panel_info,
        "scan_roots": [str(p) for p in scan_roots],
        "matching_retained_assets": int(len(assets)),
        "python_script_candidates": int(len(script_candidates)),
        "row_level_score_asset_candidates": int(len(row_score_candidates)),
        "R17A_guardrails": {
            "target_recalibration": False,
            "target_threshold_tuning": False,
            "feature_selection": False,
            "post_hoc_score_reversal": False,
            "inject_dropout_to_create_ADIC": False,
            "mix_inconsistent_regenerated_CCD_states_with_historical_labels": False,
        },
        "interpretation": (
            "This audit does not choose the final R32B common subset. "
            "R32B1 may freeze only baselines/states with faithful retained "
            "implementation and state identity support."
        ),
        "next": "R32B1_FREEZE_FAITHFUL_COMMON_BASELINE_PANEL",
    }

    audit_path = out / "R32B0_AUDIT.json"
    atomic_json(audit_path, audit)

    final_lock = {
        "status": "PASS_R32B0_PUBLISHED_RELIABILITY_ASSET_AUDIT_COMPLETE",
        "version": VERSION,
        "R32A1_fix1_final_sha256": EXPECTED_R32A1_FINAL_SHA256,
        "panel_inventory_sha256": sha256_file(panel_path),
        "asset_inventory_sha256": sha256_file(assets_path),
        "support_summary_sha256": sha256_file(support_path),
        "audit_sha256": sha256_file(audit_path),
        "baseline_inference": False,
        "target_recalibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
        "next": "R32B1_FREEZE_FAITHFUL_COMMON_BASELINE_PANEL",
    }
    final_path = out / "R32B0_FINAL_LOCK.json"
    atomic_json(final_path, final_lock)

    print("\nR32B0 NEW THREE-ACTION PANEL")
    print("  physical cases :", panel_info["physical_cases"])
    print("  model states   :", panel_info["model_states"])
    print("  model families :", panel_info["model_families"])
    print("  actions        :", panel_info["actions"])

    print("\nR32B0 RETAINED BASELINE ASSET SUPPORT")
    print(support.to_string(index=False))

    print("\nR32B0 TOP CANDIDATE ASSETS")
    if len(assets):
        cols = [
            "relative_path",
            "suffix",
            "filename_hits",
            "content_hits",
            "row_level_score_candidate",
        ]
        print(assets[cols].head(30).to_string(index=False))
    else:
        print("  NONE FOUND under audited local roots.")

    print("\nFINAL STATUS : PASS_R32B0_PUBLISHED_RELIABILITY_ASSET_AUDIT_COMPLETE")
    print("CCD / ADIC / MC inference : NO")
    print("Target recalibration      : NO")
    print("Post-hoc score reversal   : NO")
    print("External cohort access    : NO")
    print("Final lock SHA256         :", sha256_file(final_path))
    print("Output                    :", out)
    print("NEXT                      : R32B1_FREEZE_FAITHFUL_COMMON_BASELINE_PANEL")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
