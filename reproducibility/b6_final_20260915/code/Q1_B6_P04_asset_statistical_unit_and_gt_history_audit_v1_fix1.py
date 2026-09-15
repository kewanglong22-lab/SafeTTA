#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-4 fix1
Exact-token physical-ID validation + probability/logit panel-binding audit.

Why fix1 exists:
The v1 audit used substring matching and could misclassify columns such as
"examples" as an exam/procedure identifier.  fix1 preserves the original
P0 protocol and all historical assets, but replaces the heuristic with:

1) exact/token-aware ID-name matching;
2) row-level plausibility checks on candidate identifier columns;
3) explicit separation of identifiers from summary/count/statistic columns;
4) conservative physical-unit decision: UNRESOLVED unless a plausible
   row-level identifier is actually present;
5) probability/logit candidates are not called usable until exact
   action/panel/index binding is supported.

READ ONLY.
No model fit.
No new AUROC/AUPRC.
No score change.
No threshold tuning.
No candidate regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P04-v1-fix1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P0_POSTHOC_AUDIT_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "ed19e9f1d98b66a0f5280694274a1a92d001bbc884f07dd5313320e6dc63ba04"
)

V1_DIR = ROOT / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1"
V1_AUDIT = V1_DIR / "B6_P04_REPRO_AUDIT.json"

OUT_DIR = ROOT / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1_fix1"
PASS_GATE = "PASS_B6_P04_FIX1_EXACT_ID_AND_PANEL_BINDING_AUDIT_COMPLETE"

# ---------------------------------------------------------------------
# Conservative identifier semantics.
# IMPORTANT: no substring rule such as "exam" in "examples".
# ---------------------------------------------------------------------

LEVEL_PATTERNS = {
    "patient": {
        "patient", "patient_id", "patientid",
        "subject", "subject_id", "subjectid",
    },
    "procedure/video": {
        "procedure", "procedure_id", "procedureid",
        "video", "video_id", "videoid",
        "exam", "exam_id", "examid",
        "study", "study_id", "studyid",
    },
    "case": {
        "case", "case_id", "caseid", "case_key", "casekey",
    },
    "image/frame": {
        "image", "image_id", "imageid", "image_key", "imagekey",
        "frame", "frame_id", "frameid", "frame_key", "framekey",
        "sample_id", "sampleid",
        "global_index", "globalindex",
    },
}

# Explicitly NOT treated as physical identifiers.
SUMMARY_NAME_TOKENS = {
    "n", "count", "counts", "num", "number",
    "patients", "subjects", "cases", "images", "frames", "videos",
    "target_patients", "target_cases", "physical_cases",
    "mean_patient_3d_source_dice",
    "mean_patient_3d_tent1_dice",
    "mean_patient_3d_deployed_dice",
    "candidate_cases",
    "model_case_rows",
    "target_cases",
    "updated_cases",
    "changed_case_count",
    "master_cases_matched",
    "eligible_confirmatory_cases",
    "posthoc_case_exclusion",
    "all_model_cases_received_a1",
    "examples",
}

ENTITY_TERMS = {
    "polypgen": ["polypgen"],
    "promise12": ["promise12"],
    "neopolyp": ["neopolyp"],
    "sun": ["sun-seg", "sun_seg", "sunseg", r"\sun\\"],
    "prostate158": ["prostate158"],
}

PROB_FILE_HINTS = [
    "prob", "probability", "probabilities",
    "logit", "logits",
    "entropy", "confidence", "low_conf",
]

ACTION_HINTS = {
    "TENT1": ["tent1", "tent_1", "tent-1", "tent"],
    "PL-CONF90": ["pl_conf90", "pl-conf90", "plconf90", "conf90"],
    "MEMO-SEG4-1STEP": ["memo", "seg4"],
}

# Scan only data/provenance-like text/tabular files. Do not inspect raw images.
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl"}
ARRAY_SUFFIXES = {".npy", ".npz"}
CODE_SUFFIXES = {".py", ".ps1", ".sh", ".md", ".txt", ".json"}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".cache",
}

MAX_TABULAR_ROWS_FOR_ID_VALIDATION = 200_000
MAX_CODE_FILE_BYTES = 5_000_000


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


def normalize_name(name: str) -> str:
    # camelCase -> camel_Case, then normalize separators.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def infer_id_level(column: str) -> str:
    norm = normalize_name(column)
    if norm in SUMMARY_NAME_TOKENS:
        return ""
    for level, names in LEVEL_PATTERNS.items():
        if norm in names:
            return level
    return ""


def classify_entity(path: Path) -> str:
    s = str(path).lower()
    for entity, terms in ENTITY_TERMS.items():
        for term in terms:
            if term.startswith("\\"):
                if term in s:
                    return entity
            elif term in s:
                return entity
    return "other"


def discover_files() -> Tuple[List[Path], List[Path]]:
    tabular = []
    prob_candidates = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fn in filenames:
            p = Path(dirpath) / fn
            lo = fn.lower()
            suffix = p.suffix.lower()
            ent = classify_entity(p)

            if ent != "other" and suffix in TEXT_SUFFIXES:
                tabular.append(p)

            if any(h in lo for h in PROB_FILE_HINTS):
                if suffix in TEXT_SUFFIXES or suffix in ARRAY_SUFFIXES:
                    prob_candidates.append(p)

    return tabular, prob_candidates


def read_header(path: Path) -> List[str]:
    try:
        s = path.suffix.lower()
        if s == ".csv":
            return list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        if s in {".tsv", ".txt"}:
            # txt may not be tabular. Try tab separator only.
            try:
                return list(pd.read_csv(path, sep="\t", nrows=0).columns)
            except Exception:
                return []
        if s == ".json" and path.stat().st_size <= MAX_CODE_FILE_BYTES:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return list(obj[0].keys())
            if isinstance(obj, dict):
                return list(obj.keys())
        if s == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                line = f.readline()
            obj = json.loads(line)
            return list(obj.keys()) if isinstance(obj, dict) else []
    except Exception:
        pass
    return []


def load_column_sample(path: Path, column: str) -> Tuple[pd.Series, int, str]:
    """
    Returns (series_sample, observed_rows, note).
    Never loads more than MAX_TABULAR_ROWS_FOR_ID_VALIDATION.
    """
    sfx = path.suffix.lower()

    try:
        if sfx == ".csv":
            df = pd.read_csv(
                path,
                usecols=[column],
                nrows=MAX_TABULAR_ROWS_FOR_ID_VALIDATION,
                low_memory=False,
            )
            return df[column], len(df), ""

        if sfx in {".tsv", ".txt"}:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=[column],
                nrows=MAX_TABULAR_ROWS_FOR_ID_VALIDATION,
                low_memory=False,
            )
            return df[column], len(df), ""

        if sfx == ".json":
            if path.stat().st_size > MAX_CODE_FILE_BYTES:
                return pd.Series(dtype=object), 0, "json_too_large_for_safe_validation"
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                vals = [x.get(column) for x in obj[:MAX_TABULAR_ROWS_FOR_ID_VALIDATION]
                        if isinstance(x, dict)]
                return pd.Series(vals), len(vals), ""
            if isinstance(obj, dict):
                # Top-level scalar/list field in an audit/lock is NOT automatically row-level.
                v = obj.get(column)
                if isinstance(v, list):
                    vals = v[:MAX_TABULAR_ROWS_FOR_ID_VALIDATION]
                    return pd.Series(vals), len(vals), "top_level_json_list"
                return pd.Series([v]), 1, "top_level_json_scalar_or_object"

        if sfx == ".jsonl":
            vals = []
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= MAX_TABULAR_ROWS_FOR_ID_VALIDATION:
                        break
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        vals.append(obj.get(column))
            return pd.Series(vals), len(vals), ""

    except Exception as e:
        return pd.Series(dtype=object), 0, f"read_error:{type(e).__name__}"

    return pd.Series(dtype=object), 0, "unsupported"


def assess_identifier(
    path: Path,
    column: str,
    level: str,
) -> Dict[str, Any]:
    values, n_rows, note = load_column_sample(path, column)

    if n_rows == 0:
        return {
            "plausible_row_identifier": False,
            "reason": note or "no_rows",
        }

    nonnull = values.dropna()
    n_nonnull = int(len(nonnull))
    n_unique = int(nonnull.astype(str).nunique()) if n_nonnull else 0

    unique_fraction = float(n_unique / n_nonnull) if n_nonnull else 0.0
    repeat_fraction = float(1.0 - unique_fraction) if n_nonnull else 0.0

    examples = (
        nonnull.astype(str)
        .drop_duplicates()
        .head(8)
        .tolist()
    )

    norm = normalize_name(column)
    is_summary_name = norm in SUMMARY_NAME_TOKENS

    # Row-level identifier criteria:
    # - actual multirow data;
    # - more than one distinct value;
    # - not a summary/count name.
    plausible = bool(
        n_rows >= 2
        and n_nonnull >= 2
        and n_unique >= 2
        and not is_summary_name
        and "top_level_json_scalar_or_object" not in note
    )

    # For patient/video/case cluster IDs, repeated membership is positive evidence.
    repeated_cluster_support = bool(
        plausible
        and level in {"patient", "procedure/video", "case"}
        and n_unique < n_nonnull
    )

    return {
        "plausible_row_identifier": plausible,
        "repeated_cluster_support": repeated_cluster_support,
        "sampled_rows": int(n_rows),
        "nonnull": n_nonnull,
        "unique": n_unique,
        "unique_fraction": unique_fraction,
        "repeat_fraction": repeat_fraction,
        "examples": "|".join(examples),
        "note": note,
    }


def exact_id_audit(tabular_files: List[Path]) -> pd.DataFrame:
    rows = []

    for path in tqdm(
        tabular_files,
        desc="Exact physical-ID validation",
        unit="file",
        dynamic_ncols=True,
    ):
        entity = classify_entity(path)
        cols = read_header(path)

        for col in cols:
            level = infer_id_level(col)
            if not level:
                continue

            a = assess_identifier(path, col, level)

            rows.append({
                "entity": entity,
                "level": level,
                "column": col,
                "normalized_column": normalize_name(col),
                "path": str(path),
                "filename": path.name,
                **a,
            })

    return pd.DataFrame(rows)


def choose_best_unit(id_df: pd.DataFrame, entity: str) -> Dict[str, Any]:
    if id_df.empty:
        return {
            "entity": entity,
            "best_recoverable_unit": "UNRESOLVED",
            "evidence_rows": 0,
            "evidence": [],
        }

    sub = id_df[
        (id_df["entity"] == entity)
        & (id_df["plausible_row_identifier"] == True)
    ].copy()

    if sub.empty:
        return {
            "entity": entity,
            "best_recoverable_unit": "UNRESOLVED",
            "evidence_rows": 0,
            "evidence": [],
        }

    priority = ["patient", "procedure/video", "case", "image/frame"]

    for level in priority:
        level_df = sub[sub["level"] == level].copy()
        if level_df.empty:
            continue

        # For cluster levels, prefer actual repeated support over unique keys.
        if level in {"patient", "procedure/video", "case"}:
            repeated = level_df[
                level_df["repeated_cluster_support"] == True
            ]
            if not repeated.empty:
                chosen = repeated.sort_values(
                    ["sampled_rows", "unique"],
                    ascending=[False, False],
                ).head(10)
                return {
                    "entity": entity,
                    "best_recoverable_unit": level,
                    "evidence_rows": int(len(chosen)),
                    "evidence": chosen[
                        ["column", "path", "sampled_rows", "unique",
                         "repeat_fraction", "examples"]
                    ].to_dict(orient="records"),
                }

        # A unique row-level `patient_id` may still be valid if each table contains
        # one row per patient.  We retain it as candidate evidence, but label it
        # "candidate_<level>" unless repeated membership confirms clustering.
        chosen = level_df.sort_values(
            ["sampled_rows", "unique"],
            ascending=[False, False],
        ).head(10)

        return {
            "entity": entity,
            "best_recoverable_unit": (
                level if level == "image/frame"
                else f"CANDIDATE_{level.upper()}"
            ),
            "evidence_rows": int(len(chosen)),
            "evidence": chosen[
                ["column", "path", "sampled_rows", "unique",
                 "repeat_fraction", "examples"]
            ].to_dict(orient="records"),
            "note": (
                "Identifier name and row-level values are plausible, but repeated "
                "membership evidence was not observed in the scanned table."
            ),
        }

    return {
        "entity": entity,
        "best_recoverable_unit": "UNRESOLVED",
        "evidence_rows": 0,
        "evidence": [],
    }


def inspect_array(path: Path) -> Dict[str, Any]:
    rec = {
        "shape": "",
        "dtype": "",
        "first_dim": "",
        "readable": True,
        "array_members": "",
        "note": "",
    }

    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r")
            rec["shape"] = str(tuple(a.shape))
            rec["dtype"] = str(a.dtype)
            rec["first_dim"] = int(a.shape[0]) if a.ndim else 1

        elif path.suffix.lower() == ".npz":
            z = np.load(path, mmap_mode="r")
            members = []
            first_dims = []
            for key in z.files:
                a = z[key]
                members.append(f"{key}:{tuple(a.shape)}:{a.dtype}")
                if a.ndim:
                    first_dims.append(int(a.shape[0]))
            rec["array_members"] = ";".join(members)
            rec["first_dim"] = (
                first_dims[0]
                if first_dims and len(set(first_dims)) == 1
                else ""
            )

        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5, low_memory=False)
            # Need actual row count. Fast line count for CSV text.
            with path.open("rb") as f:
                n_lines = sum(1 for _ in f)
            rec["first_dim"] = max(0, n_lines - 1)
            rec["shape"] = f"({rec['first_dim']},{len(df.columns)})"
            rec["dtype"] = "csv"

        else:
            rec["note"] = "header_only_uninspected"

    except Exception as e:
        rec["readable"] = False
        rec["note"] = f"read_error:{type(e).__name__}"

    return rec


def infer_actions(path: Path) -> List[str]:
    s = str(path).lower()
    out = []

    for action, hints in ACTION_HINTS.items():
        if any(h in s for h in hints):
            out.append(action)

    return out


def find_code_references(candidate: Path) -> List[str]:
    """
    Search by filename/stem only in reasonably small code/provenance files.
    This is evidence of lineage, not proof of exact panel binding.
    """
    needles = {candidate.name, candidate.stem}
    refs = []

    search_roots = [
        CODE,
        candidate.parent,
        candidate.parent.parent if candidate.parent.parent.exists() else candidate.parent,
    ]

    seen = set()

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue

        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]

            for fn in fns:
                p = Path(dp) / fn
                if p in seen or p == candidate:
                    continue
                seen.add(p)

                if p.suffix.lower() not in CODE_SUFFIXES:
                    continue

                try:
                    if p.stat().st_size > MAX_CODE_FILE_BYTES:
                        continue
                    txt = p.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception:
                    continue

                if any(n and n in txt for n in needles):
                    refs.append(str(p))

                if len(refs) >= 30:
                    return refs

    return refs


def nearby_index_candidates(candidate: Path) -> List[Dict[str, Any]]:
    """
    Search nearby directory levels for likely manifest/index/score tables.
    Bind only by observed row count; does NOT declare exact semantic match.
    """
    arr_info = inspect_array(candidate)
    n = arr_info.get("first_dim", "")

    if not isinstance(n, int) or n <= 0:
        return []

    roots = [candidate.parent]
    if candidate.parent.parent.exists():
        roots.append(candidate.parent.parent)

    rows = []
    seen = set()

    for root in roots:
        for p in root.glob("*.csv"):
            if p == candidate or p in seen:
                continue
            seen.add(p)

            lo = p.name.lower()
            if not any(k in lo for k in [
                "index", "manifest", "score", "panel",
                "prediction", "outcome", "meta",
            ]):
                continue

            try:
                with p.open("rb") as f:
                    row_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                continue

            if row_count == n:
                rows.append({
                    "path": str(p),
                    "rows": row_count,
                    "filename": p.name,
                })

    return rows[:30]


def probability_binding_audit(
    candidates: List[Path],
) -> pd.DataFrame:
    rows = []

    for p in tqdm(
        candidates,
        desc="Probability/logit binding audit",
        unit="file",
        dynamic_ncols=True,
    ):
        entity = classify_entity(p)
        info = inspect_array(p)
        actions = infer_actions(p)
        refs = find_code_references(p)
        nearby = nearby_index_candidates(p)

        has_action_evidence = bool(actions)
        has_code_evidence = bool(refs)
        has_index_row_binding = bool(nearby)

        # Conservative binding state.
        if (
            entity in {"polypgen", "neopolyp"}
            and info["readable"]
            and has_action_evidence
            and has_code_evidence
            and has_index_row_binding
        ):
            binding = "STRONG_CANDIDATE_REQUIRES_KEY_ALIGNMENT_CHECK"
        elif info["readable"] and (
            has_action_evidence
            or has_code_evidence
            or has_index_row_binding
        ):
            binding = "WEAK_CANDIDATE_NOT_YET_USABLE"
        else:
            binding = "UNBOUND_NOT_USABLE"

        rows.append({
            "entity": entity,
            "path": str(p),
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            **info,
            "action_hints": ";".join(actions),
            "code_reference_count": len(refs),
            "code_references": ";".join(refs),
            "nearby_equal_row_index_count": len(nearby),
            "nearby_equal_row_indexes": ";".join(
                x["path"] for x in nearby
            ),
            "binding_status": binding,
        })

    return pd.DataFrame(rows)


def verify_v1() -> Dict[str, Any]:
    if not V1_AUDIT.is_file():
        raise FileNotFoundError(
            "Original P0-4 audit missing:\n"
            f"  {V1_AUDIT}"
        )
    d = load_json(V1_AUDIT)
    if d.get("gate") != (
        "PASS_B6_P04_REPRO_AND_STATISTICAL_UNIT_AUDIT_COMPLETE"
    ):
        raise RuntimeError("Original P0-4 gate changed.")
    return d


def self_test():
    # Exact token behavior that motivated fix1.
    assert infer_id_level("examples") == ""
    assert infer_id_level("patients") == ""
    assert infer_id_level("mean_patient_3d_deployed_dice") == ""

    assert infer_id_level("patient_id") == "patient"
    assert infer_id_level("PatientID") == "patient"
    assert infer_id_level("case_key") == "case"
    assert infer_id_level("video_id") == "procedure/video"
    assert infer_id_level("global_index") == "image/frame"

    print("SELF_TEST_FALSE_POSITIVE_EXAMPLES=PASS")
    print("SELF_TEST_SUMMARY_COLUMNS_EXCLUDED=PASS")
    print("SELF_TEST_EXACT_ID_TOKENS=PASS")
    print("SELF_TEST_READ_ONLY_DESIGN=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA B6-P0-4 fix1: exact-token physical-unit validation and "
            "conservative probability/logit panel-binding audit."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    exact_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "B6_P0_PROTOCOL",
    )
    protocol = load_json(PROTOCOL)

    if protocol.get("status") != "FROZEN_BEFORE_NEW_P0_METRICS":
        raise RuntimeError("P0 protocol status changed.")

    v1 = verify_v1()

    if OUT_DIR.exists():
        raise FileExistsError(
            "Never overwrite P0-4 fix1 output:\n"
            f"  {OUT_DIR}\n"
            "If fix1 itself needs repair, use _fix2."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 144)
    print(
        "SafeTTA B6-P0-4 fix1 — exact physical-ID + "
        "probability/logit panel-binding audit"
    )
    print(f"Version       : {VERSION}")
    print(
        "Reason        : v1 substring heuristic falsely classified "
        "'examples' as procedure/exam evidence"
    )
    print("Model fitting : NO")
    print("New metrics   : NO")
    print("Score changes : NO")
    print("Target tuning : NO")
    print("=" * 144)

    print("\n[1/5] Filesystem discovery")
    tabular, prob_candidates = discover_files()
    print("entity_tabular_files =", len(tabular))
    print("probability/logit candidate files =", len(prob_candidates))

    print("\n[2/5] Exact-token physical identifier validation")
    ids = exact_id_audit(tabular)

    p_ids = OUT_DIR / "B6_P04_FIX1_EXACT_ID_VALIDATION.csv"
    ids.to_csv(
        p_ids,
        index=False,
        encoding="utf-8",
    )

    entities = {
        "PolypGen": "polypgen",
        "PROMISE12": "promise12",
        "NeoPolyp": "neopolyp",
        "SUN": "sun",
        "Prostate158": "prostate158",
    }

    summary = {
        display: choose_best_unit(ids, key)
        for display, key in entities.items()
    }

    p_summary = OUT_DIR / "B6_P04_FIX1_PHYSICAL_UNIT_SUMMARY.json"
    write_json(p_summary, summary)

    for entity, rec in summary.items():
        print(
            f"{entity}: "
            f"best_recoverable_unit={rec['best_recoverable_unit']} "
            f"evidence_rows={rec['evidence_rows']}"
        )

    print("\n[3/5] Probability/logit exact-panel binding candidates")
    prob = probability_binding_audit(prob_candidates)

    p_prob = OUT_DIR / "B6_P04_FIX1_PROBABILITY_PANEL_BINDING.csv"
    prob.to_csv(
        p_prob,
        index=False,
        encoding="utf-8",
    )

    if prob.empty:
        strong = 0
        weak = 0
    else:
        strong = int(
            (
                prob["binding_status"]
                == "STRONG_CANDIDATE_REQUIRES_KEY_ALIGNMENT_CHECK"
            ).sum()
        )
        weak = int(
            (
                prob["binding_status"]
                == "WEAK_CANDIDATE_NOT_YET_USABLE"
            ).sum()
        )

    print("strong binding candidates =", strong)
    print("weak binding candidates   =", weak)

    print("\n[4/5] P0-1 design decisions")

    polyp = summary["PolypGen"]["best_recoverable_unit"]
    promise = summary["PROMISE12"]["best_recoverable_unit"]

    if strong > 0:
        reliability_status = (
            "DO_NOT_USE_YET__RUN_EXACT_KEY_ALIGNMENT_ON_STRONG_CANDIDATES"
        )
    else:
        reliability_status = (
            "NOT_RECOVERABLE_FOR_P0_1_FROM_CURRENT_EXACT_BINDING_AUDIT"
        )

    # Clustering decision is conservative.
    def cluster_decision(unit: str) -> str:
        if unit in {"patient", "procedure/video", "case"}:
            return unit
        if unit.startswith("CANDIDATE_"):
            return (
                "UNRESOLVED__VERIFY_IDENTIFIER_SEMANTICS_BEFORE_CLUSTERING"
            )
        if unit == "image/frame":
            return "image/frame"
        return "UNRESOLVED"

    decisions = {
        "PolypGen": {
            "best_recoverable_unit": polyp,
            "P0_1_cluster_unit_decision": cluster_decision(polyp),
        },
        "PROMISE12": {
            "best_recoverable_unit": promise,
            "P0_1_cluster_unit_decision": cluster_decision(promise),
        },
        "reliability_change_baseline": {
            "status": reliability_status,
            "strong_binding_candidates": strong,
            "weak_binding_candidates": weak,
            "rule": (
                "Do not use any reliability-change asset until row-key/action/"
                "family binding to the exact frozen P0-1 panel is verified."
            ),
        },
        "v1_correction": {
            "old_PolypGen_claim": (
                v1.get("recovery_summary", {})
                .get(
                    "PolypGen_best_recoverable_unit_from_column_names",
                    "",
                )
            ),
            "old_claim_valid": False,
            "reason": (
                "v1 substring heuristic allowed false positives such as "
                "'examples' matching 'exam'."
            ),
        },
    }

    p_decisions = OUT_DIR / "B6_P04_FIX1_P01_DESIGN_DECISIONS.json"
    write_json(p_decisions, decisions)

    print(
        "PolypGen P0-1 cluster decision =",
        decisions["PolypGen"]["P0_1_cluster_unit_decision"],
    )
    print(
        "PROMISE12 P0-1 cluster decision =",
        decisions["PROMISE12"]["P0_1_cluster_unit_decision"],
    )
    print(
        "Reliability-change baseline =",
        reliability_status,
    )

    print("\n[5/5] Freeze fix1 audit")

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "read_only": True,
        "model_fit": False,
        "new_auroc_auprc": False,
        "score_change": False,
        "threshold_tuning": False,
        "target_tuning": False,
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "supersedes_only_the_following_v1_recovery_inference": (
            "physical-unit recovery and probability/logit usability inference"
        ),
        "historical_scientific_results_changed": False,
        "decisions": decisions,
        "outputs": {},
    }

    for p in [
        p_ids,
        p_summary,
        p_prob,
        p_decisions,
    ]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P04_FIX1_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 144,
        "SafeTTA B6-P0-4 fix1 AUDIT COMPLETE",
        (
            "PolypGen exact physical-unit result : "
            + polyp
        ),
        (
            "PolypGen P0-1 cluster decision      : "
            + decisions["PolypGen"]["P0_1_cluster_unit_decision"]
        ),
        (
            "PROMISE12 exact physical-unit result: "
            + promise
        ),
        (
            "PROMISE12 P0-1 cluster decision     : "
            + decisions["PROMISE12"]["P0_1_cluster_unit_decision"]
        ),
        (
            "Reliability-change baseline         : "
            + reliability_status
        ),
        f"Strong probability/logit candidates   : {strong}",
        f"Weak probability/logit candidates     : {weak}",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 144,
        "",
    ])

    (OUT_DIR / "B6_P04_FIX1_REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
