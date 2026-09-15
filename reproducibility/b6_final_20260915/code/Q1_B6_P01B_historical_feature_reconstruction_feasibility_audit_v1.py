#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — historical feature reconstruction feasibility audit.

Context
-------
P01B exact-binding audit found that final NeoPolyp 66-D SOURCE-state and
64-D semantic-transition caches are not preserved.

This audit asks a narrower question:
Can those features be reconstructed EXACTLY from the already-frozen historical
upstream assets, without using PolypGen outcomes and without redesigning the
representation?

Important historical fact:
R10L0 defined SOURCE state from:
    [M2 morphology (2-D), PCA64(CondDINO SOURCE 1536-D)]
where CondDINO SOURCE 1536-D is:
    foreground_patch_mean 768-D || background_patch_mean 768-D.

Therefore absence of a final 66-D cache is not itself fatal if:
- the R10K2A frozen SOURCE CondDINO arrays survive,
- R10J1 M2 survives,
- the frozen R10L0 PCA survives,
- exact row-key metadata survives.

For semantic transition, this audit searches for historical candidate-conditioned
768/1536-D arrays for TENT1 / PL-CONF90 and code evidence defining the exact
transition transform. If those raw conditioned features survive, 64-D
transition can potentially be reconstructed with the frozen historical PCA.

READ ONLY.
NO fitting.
NO AUROC/AUPRC.
NO HARM recomputation.
NO DINO rerun.
NO TTA rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-RECON-FEAS-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

PROTOCOL = CODE / "B6_P01B_POLYPGEN_MATCHED_CANDIDATE_CONDITIONING_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "64ad2ac72ff6fc04ba07a6492e27c6b1af9a537fd0a49e787b5ae2d0b9ea62f9"

PREV_AUDIT = (
    ROOT
    / "B6_P01B_exact_key_and_source_feature_lineage_audit_v1"
    / "B6_P01B_EXACT_BINDING_AUDIT.json"
)
EXPECTED_PREV_GATE = "PASS_B6_P01B_EXACT_KEY_AND_SOURCE_FEATURE_LINEAGE_AUDIT_COMPLETE"

# Exact historical upstream paths are from the frozen R10L0 source-estimator code.
R10K2A_DIR = (
    OUTPUTS
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1"
)
R10K2A_STATE_META = R10K2A_DIR / "R10K2A_state_metadata_no_labels.csv"
R10K2A_SOURCE_COND = R10K2A_DIR / "R10K2A_mask_conditioned_dinov2_features.npz"

R10J1_DIR = (
    OUTPUTS
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1"
)
R10J1_MORPH = R10J1_DIR / "source_mask_morphology_features_labeled.csv"

R10L0_DIR = (
    OUTPUTS
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
R10L0_PCA = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
R10L0_HEAD = R10L0_DIR / "R10L0_final_safety_head.joblib"

R05D1_MANIFEST = (
    OUTPUTS
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

OUT_DIR = ROOT / "B6_P01B_historical_feature_reconstruction_feasibility_audit_v1"
PASS_GATE = "PASS_B6_P01B_HISTORICAL_FEATURE_RECONSTRUCTION_FEASIBILITY_AUDIT_COMPLETE"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".cache",
    OUT_DIR.name,
}

ARRAY_SUFFIXES = {".npy", ".npz"}
CODE_SUFFIXES = {".py", ".ps1", ".sh", ".md", ".txt", ".json"}

ACTION_PATTERNS = {
    "TENT1": ["tent1", "tent_1", "tent-1", "tent"],
    "PL-CONF90": ["pl_conf90", "pl-conf90", "plconf90", "conf90", "pseudo"],
    "MEMO": ["memo"],
}

TRANSITION_TERMS = [
    "DeltaSemantic64",
    "dSemantic64",
    "delta_semantic",
    "Shared Q66+DeltaS",
    "Q66+DeltaS",
    "semantic transition",
    "candidate-conditioned",
    "foreground_patch_mean",
    "background_patch_mean",
    "pca.transform",
    "PCA64",
]

MAX_TEXT_BYTES = 10_000_000


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


def write_json(path: Path, x: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(x, ensure_ascii=False, indent=2) + "\n",
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


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    exact_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "P01B protocol")
    protocol = load_json(PROTOCOL)

    if protocol.get("status") != "FROZEN_BEFORE_P01B_METRICS":
        raise RuntimeError("P01B protocol status changed.")

    if not PREV_AUDIT.is_file():
        raise FileNotFoundError(PREV_AUDIT)

    prev = load_json(PREV_AUDIT)

    if prev.get("status") != "PASS" or prev.get("gate") != EXPECTED_PREV_GATE:
        raise RuntimeError("Previous exact-binding gate changed.")

    if prev.get("decision") != "SOURCE_FEATURES_NOT_RECOVERED_FROM_EXISTING_ARRAYS":
        raise RuntimeError(
            "This feasibility audit is only valid after final feature-cache recovery failed."
        )

    return protocol, prev


def norm_sid_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def audit_known_source_assets() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []

    for name, p in [
        ("R10K2A_STATE_META", R10K2A_STATE_META),
        ("R10K2A_SOURCE_COND", R10K2A_SOURCE_COND),
        ("R10J1_MORPH", R10J1_MORPH),
        ("R10L0_LOCK", R10L0_LOCK),
        ("R10L0_PCA", R10L0_PCA),
        ("R10L0_HEAD", R10L0_HEAD),
        ("R05D1_MANIFEST", R05D1_MANIFEST),
    ]:
        rows.append({
            "name": name,
            "path": str(p),
            "exists": p.is_file(),
            "bytes": p.stat().st_size if p.is_file() else "",
            "sha256": sha256_file(p) if p.is_file() else "",
        })

    inv = pd.DataFrame(rows)

    result: Dict[str, Any] = {
        "source_reconstruction_ready": False,
        "checks": {},
    }

    required = [
        R10K2A_STATE_META,
        R10K2A_SOURCE_COND,
        R10J1_MORPH,
        R10L0_LOCK,
        R10L0_PCA,
        R05D1_MANIFEST,
    ]

    if not all(p.is_file() for p in required):
        result["checks"]["all_required_source_assets_exist"] = False
        return inv, result

    result["checks"]["all_required_source_assets_exist"] = True

    meta = pd.read_csv(R10K2A_STATE_META, low_memory=False)
    morph = pd.read_csv(R10J1_MORPH, low_memory=False)
    img = pd.read_csv(R05D1_MANIFEST, low_memory=False)

    with np.load(R10K2A_SOURCE_COND, allow_pickle=False) as z:
        keys = set(z.files)
        fg = np.asarray(z["foreground_patch_mean"]) if "foreground_patch_mean" in keys else None
        bg = np.asarray(z["background_patch_mean"]) if "background_patch_mean" in keys else None

    result["checks"]["source_cond_npz_keys"] = sorted(keys)
    result["checks"]["source_fg_shape"] = list(fg.shape) if fg is not None else None
    result["checks"]["source_bg_shape"] = list(bg.shape) if bg is not None else None

    shape_ok = (
        fg is not None
        and bg is not None
        and fg.shape == (9000, 768)
        and bg.shape == (9000, 768)
    )
    result["checks"]["source_conditioned_shape_exact"] = bool(shape_ok)

    meta_required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    }
    morph_required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
        "morph_fg_fraction",
        "morph_boundary_density",
    }

    result["checks"]["state_meta_required_columns"] = bool(
        meta_required.issubset(meta.columns)
    )
    result["checks"]["morph_required_columns"] = bool(
        morph_required.issubset(morph.columns)
    )

    if meta_required.issubset(meta.columns):
        meta["_sid"] = norm_sid_series(meta["sample_id"])
        meta_unique = (
            meta[["_sid", "model_state_id"]]
            .drop_duplicates()
            .shape[0]
            == len(meta)
        )
    else:
        meta_unique = False

    if morph_required.issubset(morph.columns):
        morph["_sid"] = norm_sid_series(morph["sample_id"])
        morph_unique = (
            morph[["_sid", "model_state_id"]]
            .drop_duplicates()
            .shape[0]
            == len(morph)
        )
    else:
        morph_unique = False

    result["checks"]["state_meta_rows"] = int(len(meta))
    result["checks"]["morph_rows"] = int(len(morph))
    result["checks"]["manifest_rows"] = int(len(img))
    result["checks"]["state_meta_key_unique"] = bool(meta_unique)
    result["checks"]["morph_key_unique"] = bool(morph_unique)

    join_ok = False
    if meta_unique and morph_unique:
        joined = morph.merge(
            meta[["_sid", "model_state_id"]],
            on=["_sid", "model_state_id"],
            how="inner",
            validate="one_to_one",
        )
        join_ok = len(joined) == 9000

    result["checks"]["source_meta_morph_exact_join_9000"] = bool(join_ok)

    lock = load_json(R10L0_LOCK)

    lock_ok = (
        lock.get("decision")
        == "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
        and lock.get("primary_method") == "M2_plus_CondDINO_PCA64"
        and lock.get("representation", {}).get("pca_dimension") == 64
        and lock.get("representation", {}).get("final_input_dimension") == 66
    )
    result["checks"]["R10L0_lock_semantics_exact"] = bool(lock_ok)

    pca_ok = False
    pca_info = {}
    try:
        pca = joblib.load(R10L0_PCA)
        pca_info = {
            "class": type(pca).__name__,
            "n_components": int(getattr(pca, "n_components_", getattr(pca, "n_components", -1))),
            "components_shape": list(np.asarray(pca.components_).shape),
            "mean_shape": list(np.asarray(pca.mean_).shape),
            "whiten": bool(getattr(pca, "whiten", False)),
        }
        pca_ok = (
            pca_info["components_shape"] == [64, 1536]
            and pca_info["mean_shape"] == [1536]
            and pca_info["whiten"] is True
        )
    except Exception as e:
        pca_info = {"error": f"{type(e).__name__}:{e}"}

    result["checks"]["R10L0_PCA"] = pca_info
    result["checks"]["R10L0_PCA_exact_shape"] = bool(pca_ok)

    result["source_reconstruction_ready"] = bool(
        shape_ok
        and meta_unique
        and morph_unique
        and join_ok
        and lock_ok
        and pca_ok
        and len(meta) == 9000
        and len(morph) == 9000
        and len(img) == 1000
    )

    return inv, result


def iter_array_files() -> Iterable[Path]:
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in ARRAY_SUFFIXES:
                yield p


def infer_actions(path: Path, member: str = "") -> List[str]:
    s = (str(path) + " " + member).lower()
    out = []

    for action, pats in ACTION_PATTERNS.items():
        if any(p in s for p in pats):
            out.append(action)

    return sorted(set(out))


def infer_cohort(path: Path) -> str:
    s = str(path).lower()
    if "neopolyp" in s:
        return "NeoPolyp"
    if "polypgen" in s:
        return "PolypGen"
    return ""


def scan_conditioned_feature_candidates() -> pd.DataFrame:
    rows = []

    files = list(iter_array_files())

    for p in tqdm(
        files,
        desc="Scan 768/1536-D conditioned feature candidates",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            if p.suffix.lower() == ".npy":
                a = np.load(p, mmap_mode="r")

                if a.ndim and int(a.shape[-1]) in {768, 1536}:
                    rows.append({
                        "path": str(p),
                        "member": "",
                        "cohort_hint": infer_cohort(p),
                        "action_hints": ";".join(infer_actions(p)),
                        "shape": str(tuple(a.shape)),
                        "n0": int(a.shape[0]),
                        "last_dim": int(a.shape[-1]),
                        "dtype": str(a.dtype),
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    })

            elif p.suffix.lower() == ".npz":
                z = np.load(p, mmap_mode="r", allow_pickle=False)

                for key in z.files:
                    a = z[key]

                    if a.ndim and int(a.shape[-1]) in {768, 1536}:
                        rows.append({
                            "path": str(p),
                            "member": key,
                            "cohort_hint": infer_cohort(p),
                            "action_hints": ";".join(infer_actions(p, key)),
                            "shape": str(tuple(a.shape)),
                            "n0": int(a.shape[0]),
                            "last_dim": int(a.shape[-1]),
                            "dtype": str(a.dtype),
                            "bytes": p.stat().st_size,
                            "sha256": sha256_file(p),
                        })

        except Exception:
            continue

    return pd.DataFrame(rows)


def read_small_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def transition_code_evidence() -> pd.DataFrame:
    rows = []

    for dp, dns, fns in os.walk(CODE):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]

        for fn in fns:
            p = Path(dp) / fn

            if p.suffix.lower() not in CODE_SUFFIXES:
                continue

            txt = read_small_text(p)
            if not txt:
                continue

            low = txt.lower()
            hits = [t for t in TRANSITION_TERMS if t.lower() in low]

            if not hits:
                continue

            lines = txt.splitlines()

            for idx, line in enumerate(lines):
                line_low = line.lower()
                line_hits = [t for t in TRANSITION_TERMS if t.lower() in line_low]

                if not line_hits:
                    continue

                start = max(0, idx - 4)
                end = min(len(lines), idx + 5)

                snippet = "\n".join(
                    f"{j+1}: {lines[j]}"
                    for j in range(start, end)
                )

                rows.append({
                    "script": str(p),
                    "script_sha256": sha256_file(p),
                    "line": idx + 1,
                    "term_hits": ";".join(sorted(set(line_hits))),
                    "snippet": snippet,
                })

    return pd.DataFrame(rows)


def historical_r10k2a_script_candidates() -> pd.DataFrame:
    rows = []

    needles = [
        "R10K2A_dinov2_image_mask_representation_lock",
        "mask_conditioned_dinov2_features",
        "foreground_patch_mean",
        "background_patch_mean",
    ]

    for dp, dns, fns in os.walk(CODE):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]

        for fn in fns:
            p = Path(dp) / fn

            if p.suffix.lower() != ".py":
                continue

            txt = read_small_text(p)
            if not txt:
                continue

            hits = [n for n in needles if n.lower() in txt.lower() or n.lower() in fn.lower()]

            if hits:
                rows.append({
                    "path": str(p),
                    "filename": p.name,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                    "hits": ";".join(hits),
                })

    return pd.DataFrame(rows)


def summarize_candidate_transition_assets(
    cond: pd.DataFrame,
    code_ev: pd.DataFrame,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for action in ["TENT1", "PL-CONF90"]:
        if cond.empty:
            sub = cond
        else:
            sub = cond[
                cond["action_hints"].fillna("").str.contains(
                    action,
                    regex=False,
                )
            ].copy()

        # Favor NeoPolyp-path hints and paired FG/BG members.
        neo = sub[sub["cohort_hint"] == "NeoPolyp"] if len(sub) else sub

        if len(neo):
            status = "NEOPOLYP_CANDIDATE_CONDITIONED_RAW_FEATURES_FOUND"
            chosen = neo
        elif len(sub):
            status = "ACTION_CONDITIONED_RAW_FEATURES_FOUND_NEED_COHORT_BINDING"
            chosen = sub
        else:
            status = "NO_ACTION_CONDITIONED_RAW_FEATURE_CACHE_FOUND"
            chosen = sub

        result[action] = {
            "status": status,
            "candidate_count": int(len(chosen)),
            "top_candidates": chosen.head(30).to_dict(orient="records") if len(chosen) else [],
        }

    # Code evidence can establish exact transform even if candidate cache is absent.
    transform_hits = 0
    if not code_ev.empty:
        transform_hits = int(
            code_ev["term_hits"]
            .fillna("")
            .str.contains("pca.transform", case=False, regex=False)
            .sum()
        )

    result["transition_transform_code"] = {
        "pca_transform_evidence_rows": transform_hits,
        "total_transition_code_evidence_rows": int(len(code_ev)),
    }

    return result


def audit_r10k2a_metadata_files() -> pd.DataFrame:
    """
    Inventory JSON/CSV/NPZ files in the exact historical R10K2A output directory.
    This may expose model revision / processor / source artifact hashes needed
    for an exact DINO replay if raw candidate-conditioned features do not survive.
    """
    rows = []

    if not R10K2A_DIR.is_dir():
        return pd.DataFrame(rows)

    for p in sorted(R10K2A_DIR.rglob("*")):
        if not p.is_file():
            continue

        rows.append({
            "path": str(p),
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    return pd.DataFrame(rows)


def self_test():
    assert R10K2A_SOURCE_COND.name == "R10K2A_mask_conditioned_dinov2_features.npz"
    assert R10L0_PCA.name == "R10L0_final_condDINO_PCA64.joblib"
    assert EXPECTED_PROTOCOL_SHA256.startswith("64ad2a")
    print("SELF_TEST_HISTORICAL_PATHS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA B6-P0-1B exact historical feature-reconstruction "
            "feasibility audit."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    verify_upstream()

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite reconstruction audit: {OUT_DIR}\n"
            "Use _fix1 if this audit itself needs repair."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 156)
    print("SafeTTA B6-P0-1B — historical feature reconstruction feasibility audit")
    print(f"Version          : {VERSION}")
    print(f"Protocol SHA     : {EXPECTED_PROTOCOL_SHA256}")
    print("Model fitting    : NO")
    print("Scientific metric: NO")
    print("DINO/TTA rerun   : NO / NO")
    print("=" * 156)

    print("\n[1/5] Exact historical SOURCE-state upstream assets")
    source_inv, source = audit_known_source_assets()

    p_source_inv = OUT_DIR / "B6_P01B_SOURCE_RECONSTRUCTION_ASSET_AUDIT.csv"
    source_inv.to_csv(p_source_inv, index=False, encoding="utf-8")

    p_source = OUT_DIR / "B6_P01B_SOURCE_RECONSTRUCTION_FEASIBILITY.json"
    write_json(p_source, source)

    print(source_inv.to_string(index=False))
    print("SOURCE_STATE_EXACT_RECONSTRUCTION_READY=", source["source_reconstruction_ready"])

    print("\n[2/5] Search historical candidate-conditioned 768/1536-D feature caches")
    cond = scan_conditioned_feature_candidates()

    p_cond = OUT_DIR / "B6_P01B_CANDIDATE_CONDITIONED_RAW_FEATURE_SCAN.csv"
    cond.to_csv(p_cond, index=False, encoding="utf-8")

    print("768/1536-D candidate rows =", len(cond))

    print("\n[3/5] Recover exact historical semantic-transition transform from code")
    code_ev = transition_code_evidence()

    p_code = OUT_DIR / "B6_P01B_TRANSITION_CODE_EVIDENCE.csv"
    code_ev.to_csv(p_code, index=False, encoding="utf-8")

    summary = summarize_candidate_transition_assets(cond, code_ev)

    p_summary = OUT_DIR / "B6_P01B_TRANSITION_RECONSTRUCTION_FEASIBILITY.json"
    write_json(p_summary, summary)

    for action in ["TENT1", "PL-CONF90"]:
        print(
            f"{action:10s}",
            summary[action]["status"],
            "candidates=",
            summary[action]["candidate_count"],
        )

    print(
        "transition code evidence rows =",
        summary["transition_transform_code"]["total_transition_code_evidence_rows"],
    )

    print("\n[4/5] Exact historical R10K2A extractor / metadata inventory")
    scripts = historical_r10k2a_script_candidates()

    p_scripts = OUT_DIR / "B6_P01B_R10K2A_SCRIPT_CANDIDATES.csv"
    scripts.to_csv(p_scripts, index=False, encoding="utf-8")

    r10k2a_files = audit_r10k2a_metadata_files()

    p_r10k2a = OUT_DIR / "B6_P01B_R10K2A_OUTPUT_INVENTORY.csv"
    r10k2a_files.to_csv(p_r10k2a, index=False, encoding="utf-8")

    print("R10K2A script candidates =", len(scripts))
    print("R10K2A output files       =", len(r10k2a_files))

    print("\n[5/5] Reconstruction decision")
    source_ready = bool(source["source_reconstruction_ready"])

    action_raw_ready = all(
        summary[a]["status"]
        in {
            "NEOPOLYP_CANDIDATE_CONDITIONED_RAW_FEATURES_FOUND",
            "ACTION_CONDITIONED_RAW_FEATURES_FOUND_NEED_COHORT_BINDING",
        }
        for a in ["TENT1", "PL-CONF90"]
    )

    exact_extractor_available = len(scripts) > 0
    transform_code_available = (
        summary["transition_transform_code"]["total_transition_code_evidence_rows"] > 0
    )

    if source_ready and action_raw_ready and transform_code_available:
        decision = "READY_TO_RECONSTRUCT_FROM_FROZEN_FEATURES_WITHOUT_DINO_RERUN"
    elif source_ready and exact_extractor_available and transform_code_available:
        decision = "SOURCE_READY_CANDIDATE_SEMANTICS_REQUIRE_EXACT_DINO_REPLAY_FEASIBILITY_CHECK"
    elif source_ready:
        decision = "SOURCE_READY_BUT_SEMANTIC_TRANSITION_RECONSTRUCTION_NOT_YET_PROVEN"
    else:
        decision = "HISTORICAL_SOURCE_STATE_RECONSTRUCTION_NOT_PROVEN"

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "read_only": True,
        "model_fit": False,
        "new_scientific_metrics": False,
        "dino_rerun": False,
        "tta_rerun": False,
        "source_state_exact_reconstruction_ready": source_ready,
        "candidate_raw_features_ready": action_raw_ready,
        "transition_code_available": transform_code_available,
        "exact_r10k2a_extractor_script_available": exact_extractor_available,
        "decision": decision,
        "outputs": {},
    }

    for p in [
        p_source_inv,
        p_source,
        p_cond,
        p_code,
        p_summary,
        p_scripts,
        p_r10k2a,
    ]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_RECONSTRUCTION_FEASIBILITY_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 156,
        "SafeTTA B6-P0-1B HISTORICAL RECONSTRUCTION FEASIBILITY COMPLETE",
        f"SOURCE_STATE exact reconstruction ready : {source_ready}",
        f"TENT1/PL raw candidate semantics ready  : {action_raw_ready}",
        f"historical transition code available   : {transform_code_available}",
        f"R10K2A exact extractor script available: {exact_extractor_available}",
        f"decision                               : {decision}",
        "",
        "No scientific metrics were computed.",
        "No feature was regenerated in this audit.",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 156,
        "",
    ])

    (OUT_DIR / "B6_P01B_RECONSTRUCTION_FEASIBILITY_REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
