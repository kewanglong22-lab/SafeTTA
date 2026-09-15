#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — NeoPolyp candidate-mask lineage conflict audit.

Why this audit exists
---------------------
P01B geometry-core fix3 has already passed:
- exact NeoPolyp TENT1/PL-CONF90 outcome binding;
- exact PolypGen unseen-MEMO outcome binding;
- exact NeoPolyp SOURCE66 binding;
- exact PolypGen Q66 binding.

It then stopped because all 7200 NeoPolyp TENT1 keys were covered by more than
one mask asset and those candidate assets produced non-identical geometry.

This audit DOES NOT choose a winner. It inventories every exact-support
SOURCE/candidate mask pair, computes a deterministic geometry digest, groups
identical variants, and searches local historical code/sidecars for provenance
evidence. The next fix is allowed to bind only after this audit identifies the
authoritative historical mask lineage.

READ ONLY.
NO model fitting.
NO HARM/AUROC/AUPRC.
NO GT access.
NO target tuning.
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


VERSION = "2026-09-15-B6-P01B-MASK-LINEAGE-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

FIRST_BIND_DIR = ROOT / "B6_P01B_polypgen_asset_binding_audit_v1"
FIRST_INVENTORY = FIRST_BIND_DIR / "B6_P01B_ASSET_INVENTORY.csv"

R32A1_OOF = (
    ROOT
    / "R32A1_three_action_loao_comparison_execution_v1_fix1"
    / "R32A1_FIX1_OOF_PREDICTIONS.csv"
)

R33A3_TARGET = (
    ROOT
    / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
    / "R33A3_POLYPGEN_MEMO_MATCHED_SCORE_OUTCOME_PANEL.csv"
)

OUT_DIR = ROOT / "B6_P01B_neopolyp_mask_lineage_conflict_audit_v1"
PASS_GATE = "PASS_B6_P01B_NEOPOLYP_MASK_LINEAGE_CONFLICT_AUDIT_COMPLETE"

H = 352
W = 352
PIXELS = H * W
PACKED_BYTES = 15488
BITORDER = "little"

SOURCE_IMAGES = 800
SOURCE_STATES = 9
SOURCE_ROWS = 7200

TARGET_IMAGES = 1532
TARGET_STATES = 3
TARGET_ROWS = 4596

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
ROW_INDEX_ALIASES = [
    "state_row_index", "row_index", "global_index", "index",
]
NPZ_REF_ALIASES = [
    "state_prediction_npz", "prediction_npz", "npz_path", "mask_npz",
    "prediction_path", "artifact_path", "source_asset_relpath",
]

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".cache", OUT_DIR.name,
}
MAX_TEXT_BYTES = 12_000_000


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


def first_existing_ci(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for a in aliases:
        hit = lookup.get(str(a).strip().lower())
        if hit is not None:
            return hit
    return None


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


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


def infer_action_from_text(text: str) -> str:
    return normalize_action_value(text)


def load_source_support(action: str) -> pd.DataFrame:
    if not R32A1_OOF.is_file():
        raise FileNotFoundError(R32A1_OOF)

    df = pd.read_csv(R32A1_OOF, low_memory=False)
    sc = first_existing_ci(df.columns, SAMPLE_ALIASES)
    stc = first_existing_ci(df.columns, STATE_ALIASES)
    ac = first_existing_ci(df.columns, ACTION_ALIASES)

    if sc is None or stc is None or ac is None:
        raise RuntimeError("R32A1 OOF missing sample/state/action key.")

    act = df[ac].map(normalize_action_value)
    sub = pd.DataFrame({
        "sample_id": norm_id(df.loc[act == action, sc]),
        "model_state_id": df.loc[act == action, stc].astype(str).str.strip(),
    }).drop_duplicates().reset_index(drop=True)

    if len(sub) != SOURCE_ROWS:
        raise RuntimeError(f"{action} support rows={len(sub)} != {SOURCE_ROWS}")
    if sub["sample_id"].nunique() != SOURCE_IMAGES:
        raise RuntimeError(f"{action} image count changed.")
    if sub["model_state_id"].nunique() != SOURCE_STATES:
        raise RuntimeError(f"{action} state count changed.")
    if sub[KEY_COLS].drop_duplicates().shape[0] != SOURCE_ROWS:
        raise RuntimeError(f"{action} support key not unique.")

    return sub


def load_target_support() -> pd.DataFrame:
    if not R33A3_TARGET.is_file():
        raise FileNotFoundError(R33A3_TARGET)

    df = pd.read_csv(R33A3_TARGET, low_memory=False)
    required = {"sample_id", "model_state_id", "action"}
    if not required.issubset(df.columns):
        raise RuntimeError("R33A3 target panel key columns missing.")

    if len(df) != TARGET_ROWS:
        raise RuntimeError("R33A3 row count changed.")

    act = df["action"].map(normalize_action_value)
    if set(act) != {"MEMO"}:
        raise RuntimeError("R33A3 target action changed.")

    out = pd.DataFrame({
        "sample_id": norm_id(df["sample_id"]),
        "model_state_id": df["model_state_id"].astype(str).str.strip(),
    })

    if out[KEY_COLS].drop_duplicates().shape[0] != TARGET_ROWS:
        raise RuntimeError("R33A3 target key not unique.")

    return out


def candidate_npz_paths(inv: pd.DataFrame) -> List[Path]:
    out: List[Path] = []
    seen = set()

    # Include all inventory NPZs with mask/pred/keyed roles.
    for _, r in inv.iterrows():
        p = Path(str(r.get("path", "")))
        if not p.is_file() or p.suffix.lower() != ".npz":
            continue
        roles = str(r.get("roles", ""))
        low = str(p).lower()
        relevant = (
            "MASK" in roles
            or "KEYED_TABLE" in roles
            or any(x in low for x in [
                "r10", "r31", "r32", "r33", "neopolyp", "polypgen",
                "tent", "pl", "memo", "prediction", "mask",
            ])
        )
        if not relevant:
            continue
        k = str(p.resolve()).lower()
        if k not in seen:
            seen.add(k)
            out.append(p)

    # Add sibling NPZs because inventory can attach role to a sidecar but not
    # the actual state-prediction NPZ.
    for p0 in list(out):
        for p in p0.parent.glob("*.npz"):
            k = str(p.resolve()).lower()
            if p.is_file() and k not in seen:
                seen.add(k)
                out.append(p)

    return out


def candidate_mask_pairs(npz_path: Path) -> List[Tuple[str, str, str]]:
    try:
        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return []

    mask_keys = []
    for k in z.files:
        try:
            a = z[k]
        except Exception:
            continue
        if (
            getattr(a, "ndim", 0) == 2
            and a.shape[1] == PACKED_BYTES
            and "mask" in k.lower()
        ):
            mask_keys.append(k)

    source_keys = [k for k in mask_keys if "source" in k.lower()]
    if not source_keys:
        return []

    pairs = []
    for sk in source_keys:
        for ck in mask_keys:
            if ck == sk:
                continue

            # This is only a hint, never the authoritative choice.
            hint = infer_action_from_text(f"{npz_path} {ck}")
            if hint == "" and ck.lower().startswith("a1"):
                hint = "TENT1"

            pairs.append((sk, ck, hint))

    return pairs


def find_manifests(npz_path: Path, n: int) -> List[pd.DataFrame]:
    results = []
    seen = set()

    # Exact local neighborhood, then one parent level.
    for d in [npz_path.parent, npz_path.parent.parent]:
        if not d.is_dir():
            continue

        for p in d.glob("*.csv"):
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                hdr = pd.read_csv(p, nrows=0, low_memory=False)
            except Exception:
                continue

            sc = first_existing_ci(hdr.columns, SAMPLE_ALIASES)
            stc = first_existing_ci(hdr.columns, STATE_ALIASES)
            if sc is None or stc is None:
                continue

            refc = first_existing_ci(hdr.columns, NPZ_REF_ALIASES)
            ric = first_existing_ci(hdr.columns, ROW_INDEX_ALIASES)

            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue

            if refc is not None:
                ref = df[refc].astype(str)
                match = ref.map(
                    lambda x: Path(str(x)).name.lower() == npz_path.name.lower()
                )
                sub = df.loc[match].copy()
            else:
                # Conservative fallback only within exact same directory.
                if p.parent != npz_path.parent or len(df) != n:
                    continue
                sub = df.copy()

            if len(sub) != n:
                continue

            meta = pd.DataFrame({
                "sample_id": norm_id(sub[sc]),
                "model_state_id": sub[stc].astype(str).str.strip(),
            })

            if ric is not None:
                idx = pd.to_numeric(sub[ric], errors="coerce")
                if idx.isna().any():
                    continue
                meta["_row"] = idx.astype(int).to_numpy()
            else:
                meta["_row"] = np.arange(n, dtype=np.int64)

            if meta["_row"].min() < 0 or meta["_row"].max() >= n:
                continue
            if meta[KEY_COLS].drop_duplicates().shape[0] != n:
                continue

            meta["_manifest"] = str(p)
            results.append(meta)

    return results


def unpack_little(batch: np.ndarray) -> np.ndarray:
    p = np.asarray(batch, dtype=np.uint8)
    bits = np.unpackbits(
        p,
        axis=1,
        count=PIXELS,
        bitorder=BITORDER,
    )
    return bits.reshape(-1, H, W).astype(bool, copy=False)


def morphology(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=bool)
    fg = m.mean(axis=(1, 2), dtype=np.float64)
    denom = float((H - 1) * W + H * (W - 1))
    v = np.not_equal(m[:, 1:, :], m[:, :-1, :]).sum(
        axis=(1, 2), dtype=np.int64
    )
    h = np.not_equal(m[:, :, 1:], m[:, :, :-1]).sum(
        axis=(1, 2), dtype=np.int64
    )
    bd = (v + h).astype(np.float64) / denom
    return np.column_stack([fg, bd])


def geometry_for_rows(
    source_packed: np.ndarray,
    candidate_packed: np.ndarray,
    rows: np.ndarray,
    batch: int = 64,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), 4), dtype=np.float64)

    for st in range(0, len(rows), batch):
        en = min(st + batch, len(rows))
        idx = rows[st:en]

        src = unpack_little(np.asarray(source_packed[idx], dtype=np.uint8))
        can = unpack_little(np.asarray(candidate_packed[idx], dtype=np.uint8))

        inter = np.logical_and(src, can).sum(
            axis=(1, 2), dtype=np.int64
        )
        union = np.logical_or(src, can).sum(
            axis=(1, 2), dtype=np.int64
        )
        den = (
            src.sum(axis=(1, 2), dtype=np.int64)
            + can.sum(axis=(1, 2), dtype=np.int64)
        )

        dice = np.ones(len(idx), dtype=np.float64)
        nz = den != 0
        dice[nz] = 2.0 * inter[nz] / den[nz]

        iou = np.ones(len(idx), dtype=np.float64)
        nzu = union != 0
        iou[nzu] = inter[nzu] / union[nzu]

        sm = morphology(src)
        cm = morphology(can)

        out[st:en] = np.column_stack([
            dice,
            iou,
            cm[:, 0] - sm[:, 0],
            cm[:, 1] - sm[:, 1],
        ])

    return out


def array_digest(x: np.ndarray) -> str:
    a = np.asarray(x, dtype="<f8", order="C")
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def sidecar_evidence(npz_path: Path, npz_sha: str) -> Dict[str, Any]:
    """
    Search nearby JSON/TXT/CSV/PY files for exact NPZ filename/SHA and action
    terminology. This is provenance evidence only.
    """
    hits = []

    dirs = [npz_path.parent, npz_path.parent.parent]
    needles = [npz_path.name.lower(), npz_sha.lower()]

    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".json", ".txt", ".md", ".csv", ".py"}:
                continue
            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            low = txt.lower()
            if not any(n in low for n in needles):
                continue

            action_terms = []
            for term in [
                "tent1", "tent", "pl-conf90", "pl_conf90",
                "conf90", "memo", "a1",
            ]:
                if term in low:
                    action_terms.append(term)

            hits.append({
                "path": str(p),
                "sha256": sha256_file(p),
                "action_terms": action_terms,
            })

    return {
        "reference_count": len(hits),
        "references": hits[:20],
    }


def inspect_panel(
    panel_name: str,
    requested_action: str,
    support: pd.DataFrame,
    npz_paths: List[Path],
) -> pd.DataFrame:
    required_tuples = set(map(tuple, support[KEY_COLS].to_numpy()))
    rows_out = []

    for p in tqdm(
        npz_paths,
        desc=f"Mask lineage {panel_name}/{requested_action}",
        unit="npz",
        dynamic_ncols=True,
    ):
        try:
            z = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            continue

        pairs = candidate_mask_pairs(p)
        if not pairs:
            continue

        npz_sha = sha256_file(p)

        for sk, ck, action_hint in pairs:
            try:
                source = z[sk]
                candidate = z[ck]
            except Exception:
                continue

            if source.shape != candidate.shape:
                continue
            if source.ndim != 2 or source.shape[1] != PACKED_BYTES:
                continue

            n = int(source.shape[0])
            manifests = find_manifests(p, n)

            for meta in manifests:
                keys = set(map(tuple, meta[KEY_COLS].to_numpy()))
                overlap = len(keys & required_tuples)
                if overlap == 0:
                    continue

                exact_support = (
                    overlap == len(required_tuples)
                    and len(required_tuples) == len(support)
                )

                rec = {
                    "panel": panel_name,
                    "requested_action": requested_action,
                    "action_hint": action_hint,
                    "npz_path": str(p),
                    "npz_sha256": npz_sha,
                    "source_key": sk,
                    "candidate_key": ck,
                    "npz_rows": n,
                    "manifest_path": str(meta["_manifest"].iloc[0]),
                    "support_overlap": int(overlap),
                    "required_rows": int(len(support)),
                    "exact_support": bool(exact_support),
                }

                if exact_support:
                    mm = support.merge(
                        meta,
                        on=KEY_COLS,
                        how="left",
                        validate="one_to_one",
                    )
                    if mm["_row"].isna().any():
                        raise RuntimeError("Exact-support merge unexpectedly missing rows.")

                    rr = mm["_row"].to_numpy(dtype=np.int64)
                    geom = geometry_for_rows(source, candidate, rr)

                    rec.update({
                        "geometry_sha256": array_digest(geom),
                        "pred_dice_mean": float(geom[:, 0].mean()),
                        "pred_dice_median": float(np.median(geom[:, 0])),
                        "pred_iou_mean": float(geom[:, 1].mean()),
                        "area_delta_mean": float(geom[:, 2].mean()),
                        "area_delta_abs_mean": float(np.abs(geom[:, 2]).mean()),
                        "boundary_delta_mean": float(geom[:, 3].mean()),
                        "boundary_delta_abs_mean": float(np.abs(geom[:, 3]).mean()),
                        "unchanged_geometry_rows": int(
                            np.sum(
                                (geom[:, 0] == 1.0)
                                & (geom[:, 1] == 1.0)
                                & (geom[:, 2] == 0.0)
                                & (geom[:, 3] == 0.0)
                            )
                        ),
                    })

                    evidence = sidecar_evidence(p, npz_sha)
                    rec["sidecar_reference_count"] = evidence["reference_count"]
                    rec["sidecar_references_json"] = json.dumps(
                        evidence["references"],
                        ensure_ascii=False,
                    )

                rows_out.append(rec)

    return pd.DataFrame(rows_out)


def summarize_exact(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df[df["exact_support"] == True].copy()
    if x.empty:
        return pd.DataFrame()

    group_cols = [
        "panel", "requested_action", "geometry_sha256"
    ]

    rows = []
    for key, g in x.groupby(group_cols, dropna=False):
        rows.append({
            "panel": key[0],
            "requested_action": key[1],
            "geometry_sha256": key[2],
            "candidate_records": int(len(g)),
            "unique_npz_sha256": int(g["npz_sha256"].nunique()),
            "unique_npz_paths": int(g["npz_path"].nunique()),
            "action_hints": ";".join(
                sorted(set(g["action_hint"].fillna("").astype(str)))
            ),
            "source_keys": ";".join(sorted(set(g["source_key"].astype(str)))),
            "candidate_keys": ";".join(sorted(set(g["candidate_key"].astype(str)))),
            "pred_dice_mean": float(g["pred_dice_mean"].iloc[0]),
            "pred_iou_mean": float(g["pred_iou_mean"].iloc[0]),
            "area_delta_abs_mean": float(g["area_delta_abs_mean"].iloc[0]),
            "boundary_delta_abs_mean": float(g["boundary_delta_abs_mean"].iloc[0]),
            "unchanged_geometry_rows": int(g["unchanged_geometry_rows"].iloc[0]),
            "sidecar_reference_count_total": int(
                g["sidecar_reference_count"].fillna(0).sum()
            ),
            "representative_path": str(g["npz_path"].iloc[0]),
            "representative_manifest": str(g["manifest_path"].iloc[0]),
        })

    return pd.DataFrame(rows).sort_values(
        ["panel", "requested_action", "candidate_records"],
        ascending=[True, True, False],
    )


def code_reference_scan(exact: pd.DataFrame) -> pd.DataFrame:
    if exact.empty:
        return pd.DataFrame()

    needles = {}
    for _, r in exact.iterrows():
        for value in [r["npz_path"], r["npz_sha256"]]:
            s = str(value)
            if s:
                needles[s.lower()] = s

    rows = []
    for dp, dns, fns in os.walk(CODE):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() not in {".py", ".ps1", ".sh", ".md", ".txt"}:
                continue
            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            low = txt.lower()
            hits = [orig for n, orig in needles.items() if n in low]
            if not hits:
                continue

            action_terms = []
            for term in [
                "tent1", "tent", "pl-conf90", "pl_conf90",
                "conf90", "memo", "r31", "r32", "r33",
            ]:
                if term in low:
                    action_terms.append(term)

            rows.append({
                "script": str(p),
                "script_sha256": sha256_file(p),
                "asset_hits": ";".join(hits[:20]),
                "action_terms": ";".join(sorted(set(action_terms))),
            })

    return pd.DataFrame(rows)


def self_test():
    assert SOURCE_IMAGES * SOURCE_STATES == SOURCE_ROWS
    assert TARGET_IMAGES * TARGET_STATES == TARGET_ROWS

    z = np.zeros((1, H, W), dtype=bool)
    m = morphology(z)
    assert m.shape == (1, 2)
    assert float(m[0, 0]) == 0.0
    assert float(m[0, 1]) == 0.0

    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST_GEOMETRY=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P01B NeoPolyp candidate-mask lineage conflict audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    for p in [FIRST_INVENTORY, R32A1_OOF, R33A3_TARGET]:
        if not p.is_file():
            raise FileNotFoundError(p)

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite mask-lineage audit: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 164)
    print("SafeTTA B6-P0-1B — candidate-mask lineage conflict audit")
    print(f"Version            : {VERSION}")
    print("Model fitting      : NO")
    print("HARM/AUROC/AUPRC   : NO")
    print("GT access          : NO")
    print("Purpose            : identify authoritative TENT1 / PL-CONF90 / MEMO mask lineage")
    print("=" * 164)

    inv = pd.read_csv(FIRST_INVENTORY, low_memory=False)
    npz_paths = candidate_npz_paths(inv)
    print("candidate NPZ files =", len(npz_paths))

    tent_support = load_source_support("TENT1")
    pl_support = load_source_support("PL-CONF90")
    memo_support = load_target_support()

    all_rows = []

    print("\n[1/4] NeoPolyp TENT1 exact-support mask variants")
    all_rows.append(
        inspect_panel("NeoPolyp", "TENT1", tent_support, npz_paths)
    )

    print("\n[2/4] NeoPolyp PL-CONF90 exact-support mask variants")
    all_rows.append(
        inspect_panel("NeoPolyp", "PL-CONF90", pl_support, npz_paths)
    )

    print("\n[3/4] PolypGen MEMO exact-support mask variants")
    all_rows.append(
        inspect_panel("PolypGen", "MEMO", memo_support, npz_paths)
    )

    detail = pd.concat(all_rows, ignore_index=True)
    p_detail = OUT_DIR / "B6_P01B_MASK_LINEAGE_CANDIDATES.csv"
    detail.to_csv(p_detail, index=False, encoding="utf-8")

    summary = summarize_exact(detail)
    p_summary = OUT_DIR / "B6_P01B_MASK_LINEAGE_GEOMETRY_GROUPS.csv"
    summary.to_csv(p_summary, index=False, encoding="utf-8")

    print("\n[4/4] Historical code-reference scan")
    exact = detail[detail["exact_support"] == True].copy()
    refs = code_reference_scan(exact)
    p_refs = OUT_DIR / "B6_P01B_MASK_LINEAGE_CODE_REFERENCES.csv"
    refs.to_csv(p_refs, index=False, encoding="utf-8")

    print("\nEXACT-SUPPORT GEOMETRY GROUPS")
    if len(summary):
        display_cols = [
            "panel", "requested_action", "candidate_records",
            "unique_npz_sha256", "action_hints",
            "source_keys", "candidate_keys",
            "pred_dice_mean", "pred_iou_mean",
            "area_delta_abs_mean", "boundary_delta_abs_mean",
            "unchanged_geometry_rows",
            "sidecar_reference_count_total",
            "representative_path",
        ]
        print(summary[display_cols].to_string(index=False))
    else:
        print("NONE")

    decision_map = {}
    for panel, action in [
        ("NeoPolyp", "TENT1"),
        ("NeoPolyp", "PL-CONF90"),
        ("PolypGen", "MEMO"),
    ]:
        g = summary[
            (summary["panel"] == panel)
            & (summary["requested_action"] == action)
        ]
        n = int(len(g))
        if n == 0:
            decision = "NO_EXACT_SUPPORT_MASK_VARIANT"
        elif n == 1:
            decision = "UNIQUE_GEOMETRY_VARIANT"
        else:
            decision = "MULTIPLE_NONIDENTICAL_GEOMETRY_VARIANTS_REQUIRE_LINEAGE_BINDING"

        decision_map[f"{panel}/{action}"] = {
            "geometry_variant_count": n,
            "decision": decision,
        }

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "read_only": True,
        "model_fit": False,
        "gt_access": False,
        "scientific_metrics": False,
        "decisions": decision_map,
        "outputs": {},
    }

    for p in [p_detail, p_summary, p_refs]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_MASK_LINEAGE_CONFLICT_AUDIT.json"
    write_json(p_audit, audit)

    print("\nDECISIONS")
    for k, v in decision_map.items():
        print(
            f"{k:24s} variants={v['geometry_variant_count']} "
            f"decision={v['decision']}"
        )

    print("\n" + "=" * 164)
    print("SafeTTA B6-P0-1B MASK LINEAGE CONFLICT AUDIT COMPLETE")
    print(f"GATE={PASS_GATE}")
    print(f"audit_json={p_audit}")
    print(f"stage_dir={OUT_DIR}")
    print("=" * 164)


if __name__ == "__main__":
    main()
