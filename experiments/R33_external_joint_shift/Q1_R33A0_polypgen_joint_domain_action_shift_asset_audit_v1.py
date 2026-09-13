#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A0
PolypGen Joint Domain + Unseen-Action Shift Asset Audit

Scientific motivation
---------------------
R32A established three-action LOAO transfer on NeoPolyp.
R32B established a matched comparison against published current-reliability
baselines and showed:
  - SafeTTA has the strongest macro AUPRC / AUPRC lift;
  - TEGDA-ADIC retains higher macro AUROC.

The next scientifically useful experiment is therefore NOT more NeoPolyp
tuning. It is an external joint-shift test:
    train/freeze on NeoPolyp actions
    -> evaluate on PolypGen under an unseen action.

Primary intended direction:
    NeoPolyp TENT1 + PL-CONF90 -> PolypGen MEMO-SEG4-1STEP

Potential stronger extension, only if exact frozen assets support it:
    NeoPolyp TENT1 + MEMO -> PolypGen PL-CONF90
    NeoPolyp PL + MEMO    -> PolypGen TENT1

R33A0 performs an asset/readiness audit only.

NO:
  - new segmentation inference
  - MEMO inference
  - GT decoding
  - model fitting
  - target calibration
  - target score direction selection
  - HARM evaluation
  - external web/network access

It inventories retained PolypGen assets needed for a prospective-style
post-R32B3 external joint-shift confirmation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-12-R33A0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Exact R32B3 lineage
# ---------------------------------------------------------------------

R32B3_SCRIPT = CODE / "Q1_R32B3_matched_three_action_harm_evaluation_v1_fix1.py"
EXPECTED_R32B3_SCRIPT_SHA256 = (
    "ad4ea4ddd39f7beecae164f1925db1e51e1a7c7de284ac0b2552ab09c0cab9f3"
)

R32B3_DIR = ROOT / "R32B3_matched_three_action_harm_evaluation_v1_fix1"
R32B3_FINAL = R32B3_DIR / "R32B3_FINAL_LOCK.json"
EXPECTED_R32B3_FINAL_SHA256 = (
    "0503a49dc7fe2a9ffe882a8257ed066751d5fdbcc3178bc8898797a489073520"
)
R32B3_BOOTSTRAP = R32B3_DIR / "R32B3_PAIRED_CLUSTERED_BOOTSTRAP.csv"
R32B3_MACRO = R32B3_DIR / "R32B3_MAIN_MACRO_METRICS.csv"

DEFAULT_OUT = ROOT / "R33A0_polypgen_joint_domain_action_shift_asset_audit_v1"

# ---------------------------------------------------------------------
# Audit targets
# ---------------------------------------------------------------------

TARGET_FAMILY = "DeepLabV3-R50"
TARGET_SEEDS = (20260817, 20260818, 20260819)

TOKENS = (
    "polypgen",
    "deeplab",
    "q_source",
    "qsource",
    "dsem",
    "delta_semantic",
    "tent",
    "pl-conf",
    "pl_conf",
    "memo",
    "ccd",
    "adic",
    "dropout",
    "harm",
    "source_feature",
    "feature_table",
    "pregt",
    "postgt",
)

HIGH_VALUE_FILENAME_TOKENS = (
    "polypgen",
    "r17a",
    "deeplab",
    "qsource",
    "q_source",
    "dsem",
    "memo",
    "tent",
    "pl",
)

TEXT_EXT = {".csv", ".json", ".txt", ".md", ".py", ".yaml", ".yml", ".tsv"}
CONTENT_SCAN_MAX = 3 * 1024 * 1024
CSV_SCHEMA_MAX = 256 * 1024 * 1024

PRUNE_DIRS = {
    ".git",
    "__pycache__",
    ".idea",
    ".pytest_cache",
    "node_modules",
}

HEAVY_DIRS = {
    "images",
    "masks",
    "weights",
    "checkpoints",
    "dataset",
    "datasets",
}

SCORE_TOKENS = (
    "score",
    "risk",
    "ccd",
    "adic",
    "entropy",
    "dropout",
    "probability",
)

FEATURE_TOKENS = (
    "q_source",
    "qsource",
    "dsem",
    "delta_semantic",
    "semantic",
    "pca",
)

OUTCOME_TOKENS = (
    "harm",
    "delta_dice",
    "dice_delta",
    "source_dice",
    "adapted_dice",
    "tent",
    "pl",
    "memo",
)

ID_TOKENS = (
    "sample_id",
    "image",
    "model_state",
    "model_state_id",
    "model_family",
    "training_seed",
    "checkpoint",
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
    require_sha(R32B3_SCRIPT, EXPECTED_R32B3_SCRIPT_SHA256, "R32B3 fix1 script")
    require_sha(R32B3_FINAL, EXPECTED_R32B3_FINAL_SHA256, "R32B3 final")

    if not R32B3_BOOTSTRAP.is_file():
        raise FileNotFoundError(R32B3_BOOTSTRAP)
    if not R32B3_MACRO.is_file():
        raise FileNotFoundError(R32B3_MACRO)

    final = json.loads(R32B3_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != (
        "PASS_R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION_COMPLETE"
    ):
        raise RuntimeError("R32B3 final status changed.")

    if sha256_file(R32B3_BOOTSTRAP) != str(final.get("bootstrap_sha256")):
        raise RuntimeError("R32B3 bootstrap SHA changed.")
    if sha256_file(R32B3_MACRO) != str(final.get("macro_metrics_sha256")):
        raise RuntimeError("R32B3 macro metrics SHA changed.")

    if bool(final.get("score_direction_reversal", True)):
        raise RuntimeError("R32B3 score-direction guard changed.")
    if bool(final.get("target_calibration", True)):
        raise RuntimeError("R32B3 target-calibration guard changed.")
    if bool(final.get("model_refitting", True)):
        raise RuntimeError("R32B3 refitting guard changed.")
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R32B3 external-access guard changed.")

    return final


def scan_roots() -> List[Path]:
    candidates = []

    for name in (
        "code",
        "outputs",
        "release",
        "provenance",
        "legacy_all_retained_code",
        "method_core",
    ):
        p = ROOT / name
        if p.is_dir():
            candidates.append(p)

    # Add project children likely to contain retained Rxx assets.
    for p in ROOT.iterdir():
        if not p.is_dir():
            continue
        low = p.name.lower()
        if (
            "polypgen" in low
            or "r17" in low
            or "r1" in low
            or "output" in low
            or "release" in low
        ):
            candidates.append(p)

    out = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p.resolve())
    return out


def iter_candidate_files(roots: Sequence[Path]) -> List[Path]:
    files = []

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            keep_dirs = []
            for d in dirnames:
                low = d.lower()
                if low in PRUNE_DIRS:
                    continue
                # Retained output folders can contain useful small sidecars.
                # Only prune obvious raw-heavy directories.
                if low in HEAVY_DIRS:
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs

            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in TEXT_EXT:
                    files.append(p)

    uniq = []
    seen = set()
    for p in files:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def text_hits(path: Path) -> tuple[List[str], List[str]]:
    lowname = path.name.lower()
    name_hits = [t for t in TOKENS if t in lowname]

    content_hits: List[str] = []
    try:
        size = path.stat().st_size
    except Exception:
        return name_hits, content_hits

    if size > CONTENT_SCAN_MAX:
        return sorted(set(name_hits)), content_hits

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return sorted(set(name_hits)), content_hits

    for t in TOKENS:
        if t in txt:
            content_hits.append(t)

    return sorted(set(name_hits)), sorted(set(content_hits))


def csv_schema(path: Path) -> Dict[str, Any]:
    result = {
        "csv_rows": None,
        "columns": "",
        "id_like_columns": "",
        "score_like_columns": "",
        "feature_like_columns": "",
        "outcome_like_columns": "",
        "deeplab_rows": None,
        "deeplab_states": "",
        "polypgen_sample_count": None,
    }

    try:
        if path.stat().st_size > CSV_SCHEMA_MAX:
            return result

        d = pd.read_csv(path, low_memory=False)
        cols = [str(c) for c in d.columns]
        lows = [c.lower() for c in cols]

        result["csv_rows"] = int(len(d))
        result["columns"] = " | ".join(cols)

        result["id_like_columns"] = " | ".join(
            c for c, lc in zip(cols, lows)
            if any(t in lc for t in ID_TOKENS)
        )
        result["score_like_columns"] = " | ".join(
            c for c, lc in zip(cols, lows)
            if any(t in lc for t in SCORE_TOKENS)
        )
        result["feature_like_columns"] = " | ".join(
            c for c, lc in zip(cols, lows)
            if any(t in lc for t in FEATURE_TOKENS)
        )
        result["outcome_like_columns"] = " | ".join(
            c for c, lc in zip(cols, lows)
            if any(t in lc for t in OUTCOME_TOKENS)
        )

        if "sample_id" in d.columns:
            result["polypgen_sample_count"] = int(
                d["sample_id"].astype(str).nunique()
            )

        if "model_family" in d.columns:
            g = d[d["model_family"].astype(str) == TARGET_FAMILY]
            result["deeplab_rows"] = int(len(g))
            if "model_state_id" in g.columns:
                result["deeplab_states"] = " | ".join(
                    sorted(g["model_state_id"].astype(str).unique().tolist())
                )

    except Exception as exc:
        result["csv_read_error"] = repr(exc)

    return result


def discover_assets(roots: Sequence[Path]) -> pd.DataFrame:
    rows = []

    files = iter_candidate_files(roots)

    for p in tqdm(
        files,
        desc="R33A0 PolypGen joint-shift asset audit",
        unit="file",
        dynamic_ncols=True,
    ):
        nh, ch = text_hits(p)
        if not nh and not ch:
            continue

        try:
            rel = str(p.relative_to(ROOT))
        except Exception:
            rel = str(p)

        row = {
            "relative_path": rel,
            "path": str(p),
            "suffix": p.suffix.lower(),
            "bytes": int(p.stat().st_size),
            "filename_hits": " | ".join(nh),
            "content_hits": " | ".join(ch),
            "sha256": sha256_file(p),
        }

        if p.suffix.lower() == ".csv":
            row.update(csv_schema(p))
        else:
            row.update({
                "csv_rows": None,
                "columns": "",
                "id_like_columns": "",
                "score_like_columns": "",
                "feature_like_columns": "",
                "outcome_like_columns": "",
                "deeplab_rows": None,
                "deeplab_states": "",
                "polypgen_sample_count": None,
            })

        joined = (
            row["relative_path"].lower()
            + " "
            + row["filename_hits"].lower()
            + " "
            + row["content_hits"].lower()
            + " "
            + row["columns"].lower()
        )

        row["has_polypgen"] = "polypgen" in joined
        row["has_deeplab"] = "deeplab" in joined
        row["has_qsource"] = (
            "q_source" in joined or "qsource" in joined
        )
        row["has_dsem"] = (
            "dsem" in joined or "delta_semantic" in joined
        )
        row["has_tent"] = "tent" in joined
        row["has_pl"] = (
            "pl-conf" in joined
            or "pl_conf" in joined
            or re.search(r"\bpl\b", joined) is not None
        )
        row["has_memo"] = "memo" in joined
        row["has_ccd"] = "ccd" in joined
        row["has_adic"] = "adic" in joined
        row["has_mc"] = "dropout" in joined

        # Deterministic ranking for manual inspection only.
        row["priority"] = (
            100 * int(row["has_polypgen"])
            + 50 * int(row["has_deeplab"])
            + 40 * int(row["has_qsource"])
            + 40 * int(row["has_dsem"])
            + 30 * int(row["has_memo"])
            + 20 * int(row["has_tent"])
            + 20 * int(row["has_pl"])
            + 10 * int(row["has_ccd"])
            + 10 * int(row["has_adic"])
            + 10 * int(row["has_mc"])
            + 5 * int(p.suffix.lower() == ".csv")
            + 3 * int(p.suffix.lower() == ".py")
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["priority", "relative_path"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def readiness_summary(assets: pd.DataFrame) -> pd.DataFrame:
    categories = [
        ("PolypGen DeepLab source/index/checkpoint mapping", "has_polypgen & has_deeplab"),
        ("PolypGen q_source / source safety representation", "has_polypgen & has_qsource"),
        ("PolypGen delta-semantic representation", "has_polypgen & has_dsem"),
        ("PolypGen TENT assets/outcomes", "has_polypgen & has_tent"),
        ("PolypGen PL assets/outcomes", "has_polypgen & has_pl"),
        ("PolypGen MEMO assets/outcomes", "has_polypgen & has_memo"),
        ("PolypGen CCD baseline", "has_polypgen & has_ccd"),
        ("PolypGen ADIC baseline", "has_polypgen & has_adic"),
        ("PolypGen MC-dropout baseline", "has_polypgen & has_mc"),
    ]

    rows = []

    if len(assets) == 0:
        for label, _ in categories:
            rows.append({
                "category": label,
                "matching_assets": 0,
                "csv_assets": 0,
                "python_scripts": 0,
                "top_candidate": "",
            })
        return pd.DataFrame(rows)

    for label, expr in categories:
        cols = [x.strip() for x in expr.split("&")]
        mask = pd.Series(True, index=assets.index)
        for c in cols:
            mask &= assets[c].astype(bool)

        g = assets[mask]

        rows.append({
            "category": label,
            "matching_assets": int(len(g)),
            "csv_assets": int((g["suffix"] == ".csv").sum()),
            "python_scripts": int((g["suffix"] == ".py").sum()),
            "top_candidate": (
                str(g.iloc[0]["relative_path"]) if len(g) else ""
            ),
        })

    return pd.DataFrame(rows)


def choose_readiness(readiness: pd.DataFrame) -> Dict[str, Any]:
    def count(prefix: str) -> int:
        g = readiness[
            readiness["category"].astype(str).str.startswith(prefix)
        ]
        return int(g.iloc[0]["matching_assets"]) if len(g) else 0

    # Asset audit only: absence means "new inference/materialization required",
    # not scientific failure.
    primary_support = {
        "deeplab_source_mapping_found": count("PolypGen DeepLab") > 0,
        "q_source_found": count("PolypGen q_source") > 0,
        "tent_found": count("PolypGen TENT") > 0,
        "pl_found": count("PolypGen PL") > 0,
        "memo_found": count("PolypGen MEMO") > 0,
        "ccd_found": count("PolypGen CCD") > 0,
        "adic_found": count("PolypGen ADIC") > 0,
        "mc_found": count("PolypGen MC-dropout") > 0,
        "delta_semantic_found": count("PolypGen delta-semantic") > 0,
    }

    if (
        primary_support["deeplab_source_mapping_found"]
        and primary_support["q_source_found"]
        and primary_support["tent_found"]
        and primary_support["pl_found"]
    ):
        decision = "READY_TO_LOCK_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL"
    else:
        decision = "NEED_EXACT_ASSET_RESOLUTION_BEFORE_R33A1"

    return {
        "decision": decision,
        "support": primary_support,
        "note": (
            "MEMO and delta-semantic may legitimately be absent because they "
            "can be the new prospectively locked inference/materialization "
            "required for the joint domain+action shift confirmation."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R33A0 PolypGen Joint Domain + Unseen-Action Shift Asset Audit")
    print("Version                     :", VERSION)
    print("Scientific role             : POST-R32B3 EXTERNAL JOINT-SHIFT PREPARATION")
    print("New inference               : NO")
    print("MEMO inference              : NO")
    print("GT/HARM evaluation          : NO")
    print("Model fitting               : NO")
    print("Target calibration          : NO")
    print("External web/network access : NO")
    print("=" * 124)

    verify_upstream()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R33A0 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    roots = scan_roots()
    assets = discover_assets(roots)
    readiness = readiness_summary(assets)
    decision = choose_readiness(readiness)

    assets_path = out / "R33A0_POLYPGEN_ASSET_INVENTORY.csv"
    readiness_path = out / "R33A0_READINESS_SUMMARY.csv"

    atomic_csv(assets, assets_path)
    atomic_csv(readiness, readiness_path)

    audit = {
        "status": "PASS_R33A0_POLYPGEN_JOINT_SHIFT_ASSET_AUDIT_COMPLETE",
        "version": VERSION,
        "scientific_status": "POST_R32B3_EXTERNAL_JOINT_SHIFT_PREPARATION",
        "R32B3_final_sha256": EXPECTED_R32B3_FINAL_SHA256,
        "scan_roots": [str(p) for p in roots],
        "candidate_assets": int(len(assets)),
        **decision,
        "primary_intended_external_direction": (
            "NeoPolyp TENT1 + PL-CONF90 -> PolypGen MEMO-SEG4-1STEP"
        ),
        "optional_three_direction_extension": (
            "only if exact retained PolypGen TENT/PL assets and new MEMO "
            "materialization support fair three-action external LOAO"
        ),
        "new_inference": False,
        "GT_HARM_evaluation": False,
        "model_fitting": False,
        "target_calibration": False,
        "external_web_network_access": False,
        "next": (
            "R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL_LOCK"
            if decision["decision"]
            == "READY_TO_LOCK_R33A1_EXTERNAL_JOINT_SHIFT_PROTOCOL"
            else "R33A0B_EXACT_POLYPGEN_ASSET_RESOLUTION"
        ),
    }

    audit_path = out / "R33A0_AUDIT.json"
    atomic_json(audit_path, audit)

    final = {
        "status": "PASS_R33A0_POLYPGEN_JOINT_SHIFT_ASSET_AUDIT_COMPLETE",
        "version": VERSION,
        "decision": decision["decision"],
        "R32B3_final_sha256": EXPECTED_R32B3_FINAL_SHA256,
        "asset_inventory_sha256": sha256_file(assets_path),
        "readiness_summary_sha256": sha256_file(readiness_path),
        "audit_sha256": sha256_file(audit_path),
        "new_inference": False,
        "GT_HARM_evaluation": False,
        "external_web_network_access": False,
        "next": audit["next"],
    }

    final_path = out / "R33A0_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR33A0 READINESS SUMMARY")
    print(readiness.to_string(index=False))

    print("\nR33A0 TOP 30 POLYPGEN CANDIDATE ASSETS")
    if len(assets):
        cols = [
            "relative_path",
            "suffix",
            "priority",
            "has_deeplab",
            "has_qsource",
            "has_dsem",
            "has_tent",
            "has_pl",
            "has_memo",
            "has_ccd",
            "has_adic",
            "has_mc",
        ]
        print(assets[cols].head(30).to_string(index=False))
    else:
        print("  NONE")

    print("\nR33A0 DECISION:", decision["decision"])
    print(
        "Primary intended direction:",
        "NeoPolyp TENT1 + PL-CONF90 -> PolypGen MEMO-SEG4-1STEP",
    )
    print("\nFINAL STATUS : PASS_R33A0_POLYPGEN_JOINT_SHIFT_ASSET_AUDIT_COMPLETE")
    print("New inference               : NO")
    print("GT/HARM evaluation          : NO")
    print("External web/network access : NO")
    print("Final lock SHA256           :", sha256_file(final_path))
    print("Output                      :", out)
    print("NEXT                        :", audit["next"])
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
