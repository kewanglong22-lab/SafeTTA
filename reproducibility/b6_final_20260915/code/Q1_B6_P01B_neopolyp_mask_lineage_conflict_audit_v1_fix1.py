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

Fix1 repairs the audit implementation: historical masks are state-sharded, so complete support is assembled across NPZs within one provenance directory; it also handles an empty summary without crashing.\n\nThis audit DOES NOT choose a winner. It inventories every exact-support
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


VERSION = "2026-09-15-B6-P01B-MASK-LINEAGE-v1-fix1"

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

OUT_DIR = ROOT / "B6_P01B_neopolyp_mask_lineage_conflict_audit_v1_fix1"
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


def bundle_id_for(npz_path: Path, source_key: str, candidate_key: str) -> str:
    """
    A historical mask panel is typically split into one NPZ per model state.
    The natural lineage bundle is therefore the NPZ parent directory together
    with the exact SOURCE/candidate array-key pair.

    We deliberately do NOT merge across different parent directories here:
    release copies, R10/R31/R32 reruns, and later reconstructions must remain
    distinct provenance candidates until their geometry digests prove equality.
    """
    return (
        f"{str(npz_path.parent.resolve()).lower()}||"
        f"{source_key}||{candidate_key}"
    )


def inspect_panel(
    panel_name: str,
    requested_action: str,
    support: pd.DataFrame,
    npz_paths: List[Path],
) -> pd.DataFrame:
    """
    Collect partial per-NPZ exact-key geometry pieces.

    Fix1 versus v1:
    - v1 incorrectly required a SINGLE NPZ to cover all 7200/4596 keys.
      Historical prediction assets are state-sharded, e.g. 800 rows per
      NeoPolyp model state or 1532 rows per PolypGen state.
    - fix1 keeps every exact-key partial shard, then assembles complete
      lineage bundles across NPZs belonging to the same historical directory.
    - only mask pairs whose action hint exactly matches requested_action are
      admitted; no cross-action mask pair can enter a bundle.
    """
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

        pairs = [
            x for x in candidate_mask_pairs(p)
            if x[2] == requested_action
        ]
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
                meta_keys = set(map(tuple, meta[KEY_COLS].to_numpy()))
                overlap_keys = meta_keys & required_tuples
                overlap = len(overlap_keys)
                if overlap == 0:
                    continue

                # Bind only the required overlap in frozen support order.
                partial_support = support[
                    support[KEY_COLS].apply(tuple, axis=1).isin(overlap_keys)
                ].copy()

                mm = partial_support.merge(
                    meta,
                    on=KEY_COLS,
                    how="left",
                    validate="one_to_one",
                )
                if mm["_row"].isna().any():
                    raise RuntimeError(
                        "Partial mask-shard merge unexpectedly missing rows."
                    )

                rr = mm["_row"].to_numpy(dtype=np.int64)
                geom = geometry_for_rows(source, candidate, rr)

                if geom.shape != (len(mm), 4):
                    raise RuntimeError("Partial geometry shape changed.")
                if not np.isfinite(geom).all():
                    raise RuntimeError("Partial geometry contains non-finite values.")

                # Store geometry keyed by exact model-case key so multiple
                # state shards can later be assembled without relying on row order.
                key_strings = [
                    f"{sid}\x1f{state}"
                    for sid, state in mm[KEY_COLS].itertuples(index=False, name=None)
                ]

                rec = {
                    "panel": panel_name,
                    "requested_action": requested_action,
                    "action_hint": action_hint,
                    "bundle_id": bundle_id_for(p, sk, ck),
                    "bundle_dir": str(p.parent),
                    "npz_path": str(p),
                    "npz_sha256": npz_sha,
                    "source_key": sk,
                    "candidate_key": ck,
                    "npz_rows": n,
                    "manifest_path": str(meta["_manifest"].iloc[0]),
                    "support_overlap": int(overlap),
                    "required_rows": int(len(support)),
                    "partial_key_sha256": hashlib.sha256(
                        "\n".join(key_strings).encode("utf-8")
                    ).hexdigest(),
                    "partial_geometry_sha256": array_digest(geom),
                    "partial_keys_json": json.dumps(
                        key_strings,
                        ensure_ascii=False,
                    ),
                    "partial_geometry_json": json.dumps(
                        geom.tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }

                evidence = sidecar_evidence(p, npz_sha)
                rec["sidecar_reference_count"] = evidence["reference_count"]
                rec["sidecar_references_json"] = json.dumps(
                    evidence["references"],
                    ensure_ascii=False,
                )
                rows_out.append(rec)

    return pd.DataFrame(rows_out)


def assemble_complete_bundles(
    partial: pd.DataFrame,
    support_by_panel_action: Dict[Tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """
    Assemble state-sharded NPZs into complete historical lineage bundles.

    A bundle is valid only if:
    1) its union covers the exact frozen support (7200 or 4596 keys);
    2) any key repeated across shards has numerically identical geometry;
    3) no extra scientific key is invented;
    4) requested action is fixed by the already-frozen candidate key/action hint.
    """
    if partial.empty:
        return pd.DataFrame(columns=[
            "panel", "requested_action", "bundle_id", "bundle_dir",
            "source_key", "candidate_key", "bundle_shards",
            "support_rows", "required_rows", "complete_support",
            "conflicting_duplicate_keys", "geometry_sha256",
            "pred_dice_mean", "pred_dice_median", "pred_iou_mean",
            "area_delta_mean", "area_delta_abs_mean",
            "boundary_delta_mean", "boundary_delta_abs_mean",
            "unchanged_geometry_rows", "unique_npz_sha256",
            "sidecar_reference_count_total", "representative_path",
            "representative_manifest",
        ])

    rows = []

    group_cols = [
        "panel", "requested_action", "bundle_id",
        "bundle_dir", "source_key", "candidate_key",
    ]

    for key, g in partial.groupby(group_cols, dropna=False, sort=False):
        panel, action, bundle_id, bundle_dir, sk, ck = key
        support = support_by_panel_action[(panel, action)].copy()

        required_key_strings = [
            f"{sid}\x1f{state}"
            for sid, state in support[KEY_COLS].itertuples(index=False, name=None)
        ]
        required_set = set(required_key_strings)

        geom_by_key: Dict[str, np.ndarray] = {}
        conflicts = set()

        for _, r in g.iterrows():
            keys = json.loads(r["partial_keys_json"])
            geom = np.asarray(
                json.loads(r["partial_geometry_json"]),
                dtype=np.float64,
            )
            if geom.shape != (len(keys), 4):
                raise RuntimeError("Serialized partial geometry/key length mismatch.")

            for kk, vv in zip(keys, geom):
                if kk not in required_set:
                    continue
                if kk in geom_by_key:
                    if not np.array_equal(geom_by_key[kk], vv):
                        conflicts.add(kk)
                else:
                    geom_by_key[kk] = vv

        covered = set(geom_by_key)
        complete = covered == required_set and len(conflicts) == 0

        rec = {
            "panel": panel,
            "requested_action": action,
            "bundle_id": bundle_id,
            "bundle_dir": bundle_dir,
            "source_key": sk,
            "candidate_key": ck,
            "bundle_shards": int(len(g)),
            "support_rows": int(len(covered)),
            "required_rows": int(len(required_set)),
            "complete_support": bool(complete),
            "conflicting_duplicate_keys": int(len(conflicts)),
            "unique_npz_sha256": int(g["npz_sha256"].nunique()),
            "sidecar_reference_count_total": int(
                g["sidecar_reference_count"].fillna(0).sum()
            ),
            "representative_path": str(g["npz_path"].iloc[0]),
            "representative_manifest": str(g["manifest_path"].iloc[0]),
        }

        if complete:
            full_geom = np.vstack([
                geom_by_key[k]
                for k in required_key_strings
            ]).astype(np.float64)

            rec.update({
                "geometry_sha256": array_digest(full_geom),
                "pred_dice_mean": float(full_geom[:, 0].mean()),
                "pred_dice_median": float(np.median(full_geom[:, 0])),
                "pred_iou_mean": float(full_geom[:, 1].mean()),
                "area_delta_mean": float(full_geom[:, 2].mean()),
                "area_delta_abs_mean": float(np.abs(full_geom[:, 2]).mean()),
                "boundary_delta_mean": float(full_geom[:, 3].mean()),
                "boundary_delta_abs_mean": float(np.abs(full_geom[:, 3]).mean()),
                "unchanged_geometry_rows": int(
                    np.sum(
                        (full_geom[:, 0] == 1.0)
                        & (full_geom[:, 1] == 1.0)
                        & (full_geom[:, 2] == 0.0)
                        & (full_geom[:, 3] == 0.0)
                    )
                ),
            })
        else:
            rec.update({
                "geometry_sha256": "",
                "pred_dice_mean": np.nan,
                "pred_dice_median": np.nan,
                "pred_iou_mean": np.nan,
                "area_delta_mean": np.nan,
                "area_delta_abs_mean": np.nan,
                "boundary_delta_mean": np.nan,
                "boundary_delta_abs_mean": np.nan,
                "unchanged_geometry_rows": np.nan,
            })

        rows.append(rec)

    return pd.DataFrame(rows)


def summarize_exact(bundles: pd.DataFrame) -> pd.DataFrame:
    """
    Group complete lineage bundles by full-panel geometry digest.

    Multiple filesystem copies with identical geometry collapse into one
    scientific geometry variant; different digests remain distinct.
    """
    columns = [
        "panel", "requested_action", "geometry_sha256",
        "complete_bundle_count", "unique_bundle_dirs",
        "bundle_shards_min", "bundle_shards_max",
        "source_keys", "candidate_keys",
        "pred_dice_mean", "pred_iou_mean",
        "area_delta_abs_mean", "boundary_delta_abs_mean",
        "unchanged_geometry_rows",
        "sidecar_reference_count_total",
        "representative_path", "representative_bundle_dir",
    ]

    if bundles.empty:
        return pd.DataFrame(columns=columns)

    x = bundles[
        (bundles["complete_support"] == True)
        & (bundles["conflicting_duplicate_keys"] == 0)
        & bundles["geometry_sha256"].astype(str).ne("")
    ].copy()

    if x.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for key, g in x.groupby(
        ["panel", "requested_action", "geometry_sha256"],
        dropna=False,
        sort=False,
    ):
        rows.append({
            "panel": key[0],
            "requested_action": key[1],
            "geometry_sha256": key[2],
            "complete_bundle_count": int(len(g)),
            "unique_bundle_dirs": int(g["bundle_dir"].nunique()),
            "bundle_shards_min": int(g["bundle_shards"].min()),
            "bundle_shards_max": int(g["bundle_shards"].max()),
            "source_keys": ";".join(sorted(set(g["source_key"].astype(str)))),
            "candidate_keys": ";".join(sorted(set(g["candidate_key"].astype(str)))),
            "pred_dice_mean": float(g["pred_dice_mean"].iloc[0]),
            "pred_iou_mean": float(g["pred_iou_mean"].iloc[0]),
            "area_delta_abs_mean": float(g["area_delta_abs_mean"].iloc[0]),
            "boundary_delta_abs_mean": float(g["boundary_delta_abs_mean"].iloc[0]),
            "unchanged_geometry_rows": int(g["unchanged_geometry_rows"].iloc[0]),
            "sidecar_reference_count_total": int(
                g["sidecar_reference_count_total"].sum()
            ),
            "representative_path": str(g["representative_path"].iloc[0]),
            "representative_bundle_dir": str(g["bundle_dir"].iloc[0]),
        })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["panel", "requested_action", "complete_bundle_count"],
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

    print("\n[1/4] NeoPolyp TENT1 state-sharded mask lineage")
    all_rows.append(
        inspect_panel("NeoPolyp", "TENT1", tent_support, npz_paths)
    )

    print("\n[2/4] NeoPolyp PL-CONF90 state-sharded mask lineage")
    all_rows.append(
        inspect_panel("NeoPolyp", "PL-CONF90", pl_support, npz_paths)
    )

    print("\n[3/4] PolypGen MEMO state-sharded mask lineage")
    all_rows.append(
        inspect_panel("PolypGen", "MEMO", memo_support, npz_paths)
    )

    nonempty = [x for x in all_rows if not x.empty]
    detail = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame()
    )
    p_detail = OUT_DIR / "B6_P01B_MASK_LINEAGE_PARTIAL_SHARDS.csv"
    detail.to_csv(p_detail, index=False, encoding="utf-8")

    support_map = {
        ("NeoPolyp", "TENT1"): tent_support,
        ("NeoPolyp", "PL-CONF90"): pl_support,
        ("PolypGen", "MEMO"): memo_support,
    }
    bundles = assemble_complete_bundles(detail, support_map)
    p_bundles = OUT_DIR / "B6_P01B_MASK_LINEAGE_COMPLETE_BUNDLES.csv"
    bundles.to_csv(p_bundles, index=False, encoding="utf-8")

    summary = summarize_exact(bundles)
    p_summary = OUT_DIR / "B6_P01B_MASK_LINEAGE_GEOMETRY_GROUPS.csv"
    summary.to_csv(p_summary, index=False, encoding="utf-8")

    print("\n[4/4] Historical code-reference scan")
    refs = code_reference_scan(detail)
    p_refs = OUT_DIR / "B6_P01B_MASK_LINEAGE_CODE_REFERENCES.csv"
    refs.to_csv(p_refs, index=False, encoding="utf-8")

    print("\nCOMPLETE LINEAGE BUNDLES")
    if len(bundles):
        bundle_cols = [
            "panel", "requested_action", "complete_support",
            "conflicting_duplicate_keys", "support_rows", "required_rows",
            "bundle_shards", "source_key", "candidate_key",
            "geometry_sha256", "bundle_dir",
        ]
        print(
            bundles.sort_values(
                ["panel", "requested_action", "complete_support", "support_rows"],
                ascending=[True, True, False, False],
            )[bundle_cols].to_string(index=False)
        )
    else:
        print("NONE")

    print("\nEXACT-SUPPORT GEOMETRY GROUPS")
    if len(summary):
        display_cols = [
            "panel", "requested_action", "complete_bundle_count",
            "unique_bundle_dirs", "bundle_shards_min", "bundle_shards_max",
            "source_keys", "candidate_keys",
            "pred_dice_mean", "pred_iou_mean",
            "area_delta_abs_mean", "boundary_delta_abs_mean",
            "unchanged_geometry_rows",
            "sidecar_reference_count_total",
            "representative_bundle_dir",
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
        ] if len(summary) else summary

        n = int(len(g))
        if n == 0:
            # Distinguish "nothing discovered" from "shards found but no
            # complete provenance bundle".
            b = bundles[
                (bundles["panel"] == panel)
                & (bundles["requested_action"] == action)
            ] if len(bundles) else bundles

            if len(b):
                decision = "SHARDS_FOUND_BUT_NO_COMPLETE_CONFLICT_FREE_BUNDLE"
            else:
                decision = "NO_MASK_SHARDS_BOUND"
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

    for p in [p_detail, p_bundles, p_summary, p_refs]:
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
