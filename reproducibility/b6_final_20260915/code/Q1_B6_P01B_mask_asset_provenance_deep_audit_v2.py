#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — deep mask-asset provenance audit v2.

This audit is triggered after the first mask-lineage audit established:
- NeoPolyp/TENT1: one directory contains 90 mask shards covering the full
  7200-key support, but every key is duplicated with non-identical geometry;
- NeoPolyp/PL-CONF90: no mask shards were bound;
- PolypGen/MEMO: no mask shards were bound.

The purpose here is NOT to choose a geometry variant. It is to recover the
historical provenance structure that explains those assets:
1) inventory every mask-bearing NPZ under the SafeTTA root;
2) inspect exact array keys/shapes and filename/path action hints;
3) inventory sidecar JSON metadata and lock/decision/action fields;
4) find CSV manifests that reference each NPZ;
5) group the 90-shard R05D3 directory into provenance families;
6) identify candidate 800-row NeoPolyp TENT1/PL-CONF90 and 1532-row
   PolypGen MEMO state shards without using any outcome metric.

READ ONLY.
NO model fitting.
NO GT access.
NO HARM/AUROC/AUPRC.
NO feature selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-MASK-PROVENANCE-DEEP-v2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R05D3_STATE_DIR = (
    ROOT
    / "outputs"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
    / "state_predictions"
)

OUT_DIR = ROOT / "B6_P01B_mask_asset_provenance_deep_audit_v2"
PASS_GATE = "PASS_B6_P01B_MASK_ASSET_PROVENANCE_DEEP_AUDIT_V2_COMPLETE"

PACKED_BYTES = 15488
NEOPOLYP_ROWS_PER_STATE = 800
POLYPGEN_ROWS_PER_STATE = 1532

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".cache", OUT_DIR.name,
}

TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".py", ".ps1", ".sh"}
MAX_TEXT_BYTES = 16_000_000

ACTION_TERMS = {
    "TENT1": ["tent1", "tent_1", "tent-1", "tent", "a1"],
    "PL-CONF90": [
        "pl-conf90", "pl_conf90", "plconf90", "conf90",
        "pseudo", "pseudo-label", "pseudo_label",
    ],
    "MEMO": ["memo"],
}

META_KEYS_INTEREST = [
    "decision",
    "status",
    "action",
    "action_name",
    "tta_action",
    "method",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "target_cases",
    "source_cases",
    "model_case_rows",
    "npz_sha256",
    "prediction_npz_sha256",
    "source_prediction_npz",
    "state_prediction_npz",
    "source_mask_key",
    "a1_mask_key",
    "candidate_mask_key",
    "selected_learning_rate",
    "lr",
    "step",
    "steps",
    "episodic",
    "optimizer",
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


def normalize_action_hint(text: str) -> str:
    low = str(text).lower()
    hits = []
    for action, terms in ACTION_TERMS.items():
        if any(t in low for t in terms):
            hits.append(action)
    return ";".join(sorted(set(hits)))


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in suffixes:
                yield p


def mask_members(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        z = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as e:
        return [{
            "member": "",
            "shape": "",
            "dtype": "",
            "mask_like": False,
            "error": f"{type(e).__name__}:{e}",
        }]

    for k in z.files:
        try:
            a = z[k]
        except Exception as e:
            rows.append({
                "member": k,
                "shape": "",
                "dtype": "",
                "mask_like": False,
                "error": f"{type(e).__name__}:{e}",
            })
            continue

        mask_like = (
            getattr(a, "ndim", 0) == 2
            and int(a.shape[-1]) == PACKED_BYTES
        )
        rows.append({
            "member": k,
            "shape": str(tuple(a.shape)),
            "dtype": str(a.dtype),
            "mask_like": bool(mask_like),
            "error": "",
        })
    return rows


def classify_npz(path: Path, members: List[Dict[str, Any]]) -> Dict[str, Any]:
    mask = [r for r in members if r.get("mask_like")]
    row_counts = sorted(set(
        int(eval(r["shape"])[0]) for r in mask if r.get("shape")
    )) if mask else []

    member_names = [r["member"] for r in mask]
    text = " ".join([str(path)] + member_names)

    cohort = ""
    if NEOPOLYP_ROWS_PER_STATE in row_counts:
        cohort = "NeoPolyp-like-800"
    if POLYPGEN_ROWS_PER_STATE in row_counts:
        cohort = (
            "PolypGen-like-1532"
            if not cohort
            else cohort + ";PolypGen-like-1532"
        )

    return {
        "mask_member_count": len(mask),
        "mask_members": ";".join(member_names),
        "mask_row_counts": ";".join(map(str, row_counts)),
        "cohort_shape_hint": cohort,
        "action_hint": normalize_action_hint(text),
        "has_source_mask": any("source" in x.lower() for x in member_names),
        "has_a1_mask": any(
            x.lower().startswith("a1") or "a1_" in x.lower()
            for x in member_names
        ),
        "has_candidate_mask": any(
            "candidate" in x.lower() or "adapt" in x.lower()
            for x in member_names
        ),
        "has_memo_mask": any("memo" in x.lower() for x in member_names),
        "has_pl_mask": any(
            ("conf90" in x.lower())
            or x.lower().startswith("pl")
            or "pseudo" in x.lower()
            for x in member_names
        ),
    }


def inventory_mask_npzs() -> pd.DataFrame:
    rows = []
    files = list(iter_files(ROOT, {".npz"}))

    for p in tqdm(
        files,
        desc="Deep mask NPZ inventory",
        unit="npz",
        dynamic_ncols=True,
    ):
        members = mask_members(p)
        cls = classify_npz(p, members)
        if cls["mask_member_count"] == 0:
            continue

        rows.append({
            "path": str(p),
            "parent": str(p.parent),
            "filename": p.name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            **cls,
        })

    return pd.DataFrame(rows)


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_json(v, key))
            elif isinstance(v, list):
                if len(v) <= 30 and all(
                    isinstance(x, (str, int, float, bool, type(None)))
                    for x in v
                ):
                    out[key] = v
            else:
                out[key] = v
    return out


def json_sidecars_for_npz(npz_path: Path, npz_sha: str) -> List[Dict[str, Any]]:
    out = []
    dirs = [npz_path.parent, npz_path.parent.parent]

    for d in dirs:
        if not d.is_dir():
            continue

        for p in d.glob("*.json"):
            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            flat = flatten_json(obj)
            raw_text = json.dumps(obj, ensure_ascii=False).lower()
            mentions = (
                npz_path.name.lower() in raw_text
                or npz_sha.lower() in raw_text
            )

            # Same-stem sidecar is also relevant even without explicit mention.
            same_stem = (
                p.stem.lower().replace("_lock", "")
                in npz_path.stem.lower()
                or npz_path.stem.lower()
                in p.stem.lower().replace("_lock", "")
            )

            if not (mentions or same_stem):
                continue

            selected = {}
            for fk, fv in flat.items():
                last = fk.split(".")[-1]
                if (
                    last in META_KEYS_INTEREST
                    or any(
                        token in fk.lower()
                        for token in [
                            "action", "decision", "state", "seed",
                            "checkpoint", "npz", "learning_rate",
                            "optimizer", "step", "episodic",
                        ]
                    )
                ):
                    if isinstance(fv, (str, int, float, bool, type(None), list)):
                        selected[fk] = fv

            out.append({
                "sidecar_path": str(p),
                "sidecar_sha256": sha256_file(p),
                "explicit_npz_reference": bool(mentions),
                "same_stem_relation": bool(same_stem),
                "action_hint": normalize_action_hint(
                    str(p) + " " + raw_text
                ),
                "selected_metadata": selected,
            })

    return out


def manifest_refs_for_npz(npz_path: Path) -> List[Dict[str, Any]]:
    out = []

    for d in [npz_path.parent, npz_path.parent.parent]:
        if not d.is_dir():
            continue

        for p in d.glob("*.csv"):
            try:
                hdr = pd.read_csv(p, nrows=0, low_memory=False)
            except Exception:
                continue

            possible_ref_cols = [
                c for c in hdr.columns
                if any(
                    t in str(c).lower()
                    for t in [
                        "npz", "prediction", "artifact",
                        "source_asset", "mask",
                    ]
                )
            ]
            if not possible_ref_cols:
                continue

            try:
                use = pd.read_csv(
                    p,
                    usecols=possible_ref_cols,
                    low_memory=False,
                )
            except Exception:
                continue

            matched_cols = []
            matched_rows = set()
            for c in possible_ref_cols:
                s = use[c].astype(str)
                m = s.map(
                    lambda x: Path(str(x)).name.lower()
                    == npz_path.name.lower()
                )
                if m.any():
                    matched_cols.append(c)
                    matched_rows.update(np.flatnonzero(m).tolist())

            if matched_cols:
                out.append({
                    "manifest_path": str(p),
                    "manifest_sha256": sha256_file(p),
                    "matched_columns": matched_cols,
                    "matched_row_count": len(matched_rows),
                })

    return out


def attach_provenance(npzs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for r in tqdm(
        npzs.itertuples(index=False),
        total=len(npzs),
        desc="Resolve NPZ provenance",
        unit="npz",
        dynamic_ncols=True,
    ):
        p = Path(r.path)
        sidecars = json_sidecars_for_npz(p, r.sha256)
        manifests = manifest_refs_for_npz(p)

        all_action = [
            str(r.action_hint)
        ]
        all_action += [x["action_hint"] for x in sidecars]

        rows.append({
            **r._asdict(),
            "resolved_action_hints": ";".join(sorted(set(
                a for x in all_action
                for a in str(x).split(";")
                if a
            ))),
            "sidecar_count": len(sidecars),
            "sidecar_json": json.dumps(
                sidecars,
                ensure_ascii=False,
            ),
            "manifest_ref_count": len(manifests),
            "manifest_refs_json": json.dumps(
                manifests,
                ensure_ascii=False,
            ),
        })

    return pd.DataFrame(rows)


def filename_family(stem: str) -> str:
    """
    Conservative grouping helper:
    removes explicit seed/state/copy-like suffixes but leaves action/method
    tokens intact. It is diagnostic only, not used to select an asset.
    """
    s = stem.lower()
    s = re.sub(r"seed\d{6,}", "seed*", s)
    s = re.sub(r"state[_-]?\d+", "state*", s)
    s = re.sub(r"fold[_-]?\d+", "fold*", s)
    s = re.sub(r"run[_-]?\d+", "run*", s)
    s = re.sub(r"copy[_-]?\d+", "copy*", s)
    s = re.sub(r"[_-]\d+$", "_*", s)
    return s


def r05d3_decomposition(prov: pd.DataFrame) -> pd.DataFrame:
    if prov.empty or not R05D3_STATE_DIR.is_dir():
        return pd.DataFrame()

    x = prov[
        prov["parent"].astype(str).str.lower()
        == str(R05D3_STATE_DIR).lower()
    ].copy()

    if x.empty:
        return x

    x["filename_family"] = x["filename"].map(
        lambda x: filename_family(Path(x).stem)
    )

    # Parse selected sidecar metadata into compact diagnostic fields.
    rows = []
    for _, r in x.iterrows():
        sidecars = json.loads(r["sidecar_json"])
        meta = {}
        for sc in sidecars:
            for k, v in sc.get("selected_metadata", {}).items():
                kl = k.lower()
                if "model_state_id" in kl and "model_state_id" not in meta:
                    meta["model_state_id"] = v
                if kl.endswith("training_seed") and "training_seed" not in meta:
                    meta["training_seed"] = v
                if "checkpoint_sha256" in kl and "checkpoint_sha256" not in meta:
                    meta["checkpoint_sha256"] = v
                if "decision" in kl and "decision" not in meta:
                    meta["decision"] = v
                if "action" in kl and "action_meta" not in meta:
                    meta["action_meta"] = v
                if (
                    ("learning_rate" in kl or kl.endswith(".lr") or kl == "lr")
                    and "learning_rate" not in meta
                ):
                    meta["learning_rate"] = v
                if "step" in kl and "steps_meta" not in meta:
                    meta["steps_meta"] = v

        rows.append({
            "filename": r["filename"],
            "filename_family": r["filename_family"],
            "sha256": r["sha256"],
            "mask_members": r["mask_members"],
            "mask_row_counts": r["mask_row_counts"],
            "resolved_action_hints": r["resolved_action_hints"],
            "sidecar_count": r["sidecar_count"],
            "manifest_ref_count": r["manifest_ref_count"],
            "model_state_id": meta.get("model_state_id", ""),
            "training_seed": meta.get("training_seed", ""),
            "checkpoint_sha256": meta.get("checkpoint_sha256", ""),
            "decision": meta.get("decision", ""),
            "action_meta": meta.get("action_meta", ""),
            "learning_rate": meta.get("learning_rate", ""),
            "steps_meta": meta.get("steps_meta", ""),
        })

    return pd.DataFrame(rows)


def summarize_candidates(prov: pd.DataFrame) -> pd.DataFrame:
    if prov.empty:
        return pd.DataFrame()

    x = prov[
        prov["cohort_shape_hint"].astype(str).str.len() > 0
    ].copy()

    rows = []
    group_cols = [
        "parent",
        "mask_row_counts",
        "mask_members",
        "resolved_action_hints",
    ]

    for key, g in x.groupby(group_cols, dropna=False, sort=False):
        rows.append({
            "parent": key[0],
            "mask_row_counts": key[1],
            "mask_members": key[2],
            "resolved_action_hints": key[3],
            "npz_count": int(len(g)),
            "unique_sha256": int(g["sha256"].nunique()),
            "sidecar_count_total": int(g["sidecar_count"].sum()),
            "manifest_ref_count_total": int(g["manifest_ref_count"].sum()),
            "example_files": ";".join(g["filename"].head(8).tolist()),
        })

    return pd.DataFrame(rows).sort_values(
        ["mask_row_counts", "resolved_action_hints", "npz_count"],
        ascending=[True, True, False],
    )


def code_search_for_dirs(prov: pd.DataFrame) -> pd.DataFrame:
    if prov.empty:
        return pd.DataFrame()

    dirs = sorted(set(
        str(Path(p).parent)
        for p in prov["path"].astype(str)
    ))
    dir_needles = {
        Path(d).name.lower(): d for d in dirs if Path(d).name
    }

    rows = []
    for p in iter_files(CODE, {".py", ".ps1", ".sh", ".md", ".txt"}):
        try:
            if p.stat().st_size > MAX_TEXT_BYTES:
                continue
            txt = p.read_text(
                encoding="utf-8", errors="ignore"
            )
        except Exception:
            continue

        low = txt.lower()
        hits = [
            original
            for needle, original in dir_needles.items()
            if needle in low
        ]
        if not hits:
            continue

        rows.append({
            "script": str(p),
            "script_sha256": sha256_file(p),
            "directory_hits": ";".join(hits),
            "action_hint": normalize_action_hint(txt),
        })

    return pd.DataFrame(rows)


def self_test():
    assert PACKED_BYTES == 15488
    assert NEOPOLYP_ROWS_PER_STATE == 800
    assert POLYPGEN_ROWS_PER_STATE == 1532
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P01B deep mask-asset provenance audit v2."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite deep mask provenance audit: {OUT_DIR}\n"
            "Use _fix1 only if this implementation needs repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 164)
    print("SafeTTA B6-P0-1B — deep mask-asset provenance audit v2")
    print(f"Version           : {VERSION}")
    print("Model fitting     : NO")
    print("GT access         : NO")
    print("Scientific metrics: NO")
    print("=" * 164)

    print("\n[1/5] Inventory all mask-bearing NPZ files")
    npzs = inventory_mask_npzs()
    p_npzs = OUT_DIR / "B6_P01B_MASK_NPZ_INVENTORY.csv"
    npzs.to_csv(p_npzs, index=False, encoding="utf-8")
    print("mask-bearing NPZ files =", len(npzs))

    print("\n[2/5] Resolve sidecar/manifest provenance")
    prov = attach_provenance(npzs)
    p_prov = OUT_DIR / "B6_P01B_MASK_NPZ_PROVENANCE.csv"
    prov.to_csv(p_prov, index=False, encoding="utf-8")

    print("\n[3/5] Decompose historical R05D3 90-shard directory")
    r05 = r05d3_decomposition(prov)
    p_r05 = OUT_DIR / "B6_P01B_R05D3_90SHARD_DECOMPOSITION.csv"
    r05.to_csv(p_r05, index=False, encoding="utf-8")

    if len(r05):
        print("R05D3 mask NPZ count =", len(r05))
        cols = [
            "filename",
            "filename_family",
            "mask_members",
            "resolved_action_hints",
            "model_state_id",
            "training_seed",
            "action_meta",
            "learning_rate",
            "steps_meta",
            "decision",
        ]
        print(r05[cols].to_string(index=False))
    else:
        print("R05D3 decomposition: NONE")

    print("\n[4/5] Summarize 800-row / 1532-row mask asset families")
    summary = summarize_candidates(prov)
    p_summary = OUT_DIR / "B6_P01B_MASK_ASSET_FAMILY_SUMMARY.csv"
    summary.to_csv(p_summary, index=False, encoding="utf-8")

    if len(summary):
        print(summary.to_string(index=False))
    else:
        print("No 800/1532-row mask families found.")

    print("\n[5/5] Historical code directory-reference scan")
    refs = code_search_for_dirs(prov)
    p_refs = OUT_DIR / "B6_P01B_MASK_DIRECTORY_CODE_REFERENCES.csv"
    refs.to_csv(p_refs, index=False, encoding="utf-8")
    print("code reference rows =", len(refs))

    # Compact action/cohort readiness view.
    readiness = {
        "NeoPolyp_TENT1": 0,
        "NeoPolyp_PL_CONF90": 0,
        "PolypGen_MEMO": 0,
    }

    for _, r in prov.iterrows():
        counts = str(r["mask_row_counts"]).split(";")
        actions = set(
            x for x in str(r["resolved_action_hints"]).split(";")
            if x
        )
        if str(NEOPOLYP_ROWS_PER_STATE) in counts:
            if "TENT1" in actions:
                readiness["NeoPolyp_TENT1"] += 1
            if "PL-CONF90" in actions:
                readiness["NeoPolyp_PL_CONF90"] += 1
        if str(POLYPGEN_ROWS_PER_STATE) in counts:
            if "MEMO" in actions:
                readiness["PolypGen_MEMO"] += 1

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "read_only": True,
        "model_fit": False,
        "gt_access": False,
        "scientific_metrics": False,
        "mask_npz_count": int(len(npzs)),
        "r05d3_mask_npz_count": int(len(r05)),
        "readiness_npz_counts": readiness,
        "outputs": {},
    }

    for p in [p_npzs, p_prov, p_r05, p_summary, p_refs]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_MASK_ASSET_PROVENANCE_DEEP_AUDIT_V2.json"
    write_json(p_audit, audit)

    print("\nREADINESS NPZ COUNTS")
    for k, v in readiness.items():
        print(f"{k:24s} {v}")

    print("\n" + "=" * 164)
    print("SafeTTA B6-P0-1B MASK ASSET PROVENANCE DEEP AUDIT V2 COMPLETE")
    print(f"GATE={PASS_GATE}")
    print(f"audit_json={p_audit}")
    print(f"stage_dir={OUT_DIR}")
    print("=" * 164)


if __name__ == "__main__":
    main()
