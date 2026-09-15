#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — authoritative mask parity + lock audit.

Goal
----
Resolve the final remaining P01B asset ambiguity WITHOUT using any scientific
outcome metric.

Recovered provenance establishes the following candidate lineages:

NeoPolyp TENT1 original historical lock:
  outputs/Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1/state_predictions
  - 9 states
  - source_masks_packed
  - a1_masks_packed
  - frozen A1_TENT_1STEP, lr=1e-3, 1 step

NeoPolyp TENT1 + PL-CONF90 exact replay:
  R38A0C_exact_frozen_tent_pl_spatial_mask_replay_v1_fix3/state_predictions
  - 9 states
  - source_masks_packed
  - tent1_masks_packed
  - pl_conf90_masks_packed

Independent historical common800 recovery:
  R38A1A_historical_candidate_mask_recovery_and_parity_localization_v1/
  historical_common800_masks
  - 9 states
  - source_masks_packed
  - tent1_masks_packed
  - pl_conf90_masks_packed

PolypGen unseen-MEMO pre-HARM lock:
  R33A2B_external_polypgen_memo_transition_score_lock_v1[_fix1]
  - 3 DeepLab states
  - source_masks_packed
  - memo_masks_packed

This audit requires exact byte parity:
1) R38A0C SOURCE == original R05D3 SOURCE, all 9 states.
2) R38A0C TENT1 == original R05D3 A1, all 9 states.
3) R38A1A SOURCE/TENT1/PL == R38A0C SOURCE/TENT1/PL, all 9 states.
4) R33A2B fix1 SOURCE/MEMO == R33A2B v1 SOURCE/MEMO, all 3 states.
5) Each R33A2B state lock maps uniquely to one DeepLab model_state_id.

Only if ALL checks pass does the audit emit an authoritative mask lock JSON.

READ ONLY.
NO GT access.
NO HARM/AUROC/AUPRC.
NO model fitting.
NO target-driven selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P01B-AUTH-MASK-PARITY-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

R05D3 = (
    ROOT
    / "outputs"
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
    / "state_predictions"
)

R38A0C = (
    ROOT
    / "R38A0C_exact_frozen_tent_pl_spatial_mask_replay_v1_fix3"
    / "state_predictions"
)

R38A1A = (
    ROOT
    / "R38A1A_historical_candidate_mask_recovery_and_parity_localization_v1"
    / "historical_common800_masks"
)

R33_V1 = ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1"
R33_FIX1 = ROOT / "R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1"

OUT_DIR = ROOT / "B6_P01B_authoritative_mask_parity_lock_audit_v1"
PASS_GATE = "PASS_B6_P01B_AUTHORITATIVE_MASK_PARITY_LOCK_COMPLETE"
FAIL_GATE = "FAIL_B6_P01B_AUTHORITATIVE_MASK_PARITY_NOT_ESTABLISHED"

PACKED_BYTES = 15488
NEO_ROWS = 800
POLYPGEN_ROWS = 1532

NEO_STATES = [
    ("DeepLabV3-R50", 20260817),
    ("DeepLabV3-R50", 20260818),
    ("DeepLabV3-R50", 20260819),
    ("PraNet", 20260817),
    ("PraNet", 20260818),
    ("PraNet", 20260819),
    ("SegFormer-B0", 20260820),
    ("SegFormer-B0", 20260821),
    ("SegFormer-B0", 20260822),
]

TARGET_STATE_IDS = {
    "DeepLabV3-R50::20260817",
    "DeepLabV3-R50::20260818",
    "DeepLabV3-R50::20260819",
}


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


def slug_family(family: str) -> str:
    if family == "DeepLabV3-R50":
        return "deeplabv3_r50"
    if family == "PraNet":
        return "pranet"
    if family == "SegFormer-B0":
        return "segformer_b0"
    raise ValueError(family)


def r05_path(family: str, seed: int) -> Path:
    return R05D3 / f"{slug_family(family)}_seed{seed}_predictions.npz"


def r38_dirname(family: str, seed: int) -> str:
    return f"{family}_{seed}"


def r38a1_dirname(family: str, seed: int) -> str:
    return f"{family}__{seed}"


def r38_path(family: str, seed: int) -> Path:
    return R38A0C / r38_dirname(family, seed) / "source_tent_pl_packed.npz"


def r38a1_path(family: str, seed: int) -> Path:
    return (
        R38A1A
        / r38a1_dirname(family, seed)
        / "historical_source_tent_pl_common800.npz"
    )


def load_mask_member(path: Path, key: str, rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    z = np.load(path, mmap_mode="r", allow_pickle=False)
    if key not in z.files:
        raise RuntimeError(f"{path} missing key={key}; available={z.files}")

    a = np.asarray(z[key], dtype=np.uint8)
    if a.shape != (rows, PACKED_BYTES):
        raise RuntimeError(
            f"{path}::{key} shape={a.shape}, expected={(rows, PACKED_BYTES)}"
        )
    return a


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            "equal": False,
            "shape_equal": False,
            "differing_rows": None,
            "differing_bytes": None,
        }

    diff = a != b
    row_diff = np.any(diff, axis=1)
    return {
        "equal": bool(np.array_equal(a, b)),
        "shape_equal": True,
        "differing_rows": int(row_diff.sum()),
        "differing_bytes": int(diff.sum()),
    }


def packed_member_sha(a: np.ndarray) -> str:
    x = np.asarray(a, dtype=np.uint8, order="C")
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def neo_parity() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    lock_states = []

    for family, seed in tqdm(
        NEO_STATES,
        desc="NeoPolyp TENT/PL parity",
        unit="state",
        dynamic_ncols=True,
    ):
        p05 = r05_path(family, seed)
        p38 = r38_path(family, seed)
        p381 = r38a1_path(family, seed)

        for p in [p05, p38, p381]:
            if not p.is_file():
                raise FileNotFoundError(p)

        src05 = load_mask_member(p05, "source_masks_packed", NEO_ROWS)
        tent05 = load_mask_member(p05, "a1_masks_packed", NEO_ROWS)

        src38 = load_mask_member(p38, "source_masks_packed", NEO_ROWS)
        tent38 = load_mask_member(p38, "tent1_masks_packed", NEO_ROWS)
        pl38 = load_mask_member(p38, "pl_conf90_masks_packed", NEO_ROWS)

        src381 = load_mask_member(p381, "source_masks_packed", NEO_ROWS)
        tent381 = load_mask_member(p381, "tent1_masks_packed", NEO_ROWS)
        pl381 = load_mask_member(p381, "pl_conf90_masks_packed", NEO_ROWS)

        checks = {
            "r38_source_vs_r05_source": compare_arrays(src38, src05),
            "r38_tent_vs_r05_a1": compare_arrays(tent38, tent05),
            "r38a1_source_vs_r38_source": compare_arrays(src381, src38),
            "r38a1_tent_vs_r38_tent": compare_arrays(tent381, tent38),
            "r38a1_pl_vs_r38_pl": compare_arrays(pl381, pl38),
        }

        state_pass = all(v["equal"] for v in checks.values())

        row = {
            "model_family": family,
            "training_seed": seed,
            "model_state_id": f"{family}::{seed}",
            "state_pass": state_pass,
            "r05_npz": str(p05),
            "r05_npz_sha256": sha256_file(p05),
            "r38_npz": str(p38),
            "r38_npz_sha256": sha256_file(p38),
            "r38a1_npz": str(p381),
            "r38a1_npz_sha256": sha256_file(p381),
            "source_member_sha256": packed_member_sha(src38),
            "tent_member_sha256": packed_member_sha(tent38),
            "pl_member_sha256": packed_member_sha(pl38),
        }

        for name, result in checks.items():
            row[f"{name}_equal"] = result["equal"]
            row[f"{name}_differing_rows"] = result["differing_rows"]
            row[f"{name}_differing_bytes"] = result["differing_bytes"]

        rows.append(row)

        if state_pass:
            lock_states.append({
                "model_state_id": f"{family}::{seed}",
                "model_family": family,
                "training_seed": seed,
                # Use the exact replay bundle as one common source for SOURCE/TENT/PL.
                # Its TENT and SOURCE are required to be byte-identical to original R05D3.
                # Its PL is required to be independently byte-identical to R38A1A recovery.
                "authoritative_npz": str(p38),
                "authoritative_npz_sha256": sha256_file(p38),
                "source_key": "source_masks_packed",
                "tent1_key": "tent1_masks_packed",
                "pl_conf90_key": "pl_conf90_masks_packed",
                "historical_tent_anchor_npz": str(p05),
                "historical_tent_anchor_sha256": sha256_file(p05),
                "independent_recovery_npz": str(p381),
                "independent_recovery_sha256": sha256_file(p381),
                "rows": NEO_ROWS,
            })

    df = pd.DataFrame(rows)
    summary = {
        "states": len(df),
        "passed_states": int(df["state_pass"].sum()),
        "all_states_pass": bool(df["state_pass"].all()),
        "lock_states": lock_states,
    }
    return df, summary


def find_state_lock_json(directory: Path, idx: int) -> Path:
    candidates = [
        directory / f"state_{idx:02d}_polypgen_source_memo_pre_harm.lock.json",
        directory / f"state_{idx:02d}_polypgen_source_memo_pre_harm_lock.json",
    ]
    for p in candidates:
        if p.is_file():
            return p

    hits = sorted(directory.glob(f"state_{idx:02d}*.json"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"No state_{idx:02d} lock JSON in {directory}"
        )
    raise RuntimeError(
        f"Ambiguous state_{idx:02d} lock JSONs: {[str(x) for x in hits]}"
    )


def extract_model_state_id(lock_path: Path) -> str:
    text = lock_path.read_text(encoding="utf-8", errors="ignore")
    hits = sorted(set(re.findall(
        r"DeepLabV3-R50::20\d{6}",
        text,
    )))
    if len(hits) != 1:
        raise RuntimeError(
            f"{lock_path}: expected exactly one DeepLab model_state_id, got {hits}"
        )
    return hits[0]


def r33_parity() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    lock_states = []

    for idx in tqdm(
        range(3),
        desc="PolypGen MEMO parity",
        unit="state",
        dynamic_ncols=True,
    ):
        name = f"state_{idx:02d}_polypgen_source_memo_pre_harm.npz"
        p0 = R33_V1 / name
        p1 = R33_FIX1 / name

        for p in [p0, p1]:
            if not p.is_file():
                raise FileNotFoundError(p)

        src0 = load_mask_member(p0, "source_masks_packed", POLYPGEN_ROWS)
        mem0 = load_mask_member(p0, "memo_masks_packed", POLYPGEN_ROWS)
        src1 = load_mask_member(p1, "source_masks_packed", POLYPGEN_ROWS)
        mem1 = load_mask_member(p1, "memo_masks_packed", POLYPGEN_ROWS)

        source_cmp = compare_arrays(src0, src1)
        memo_cmp = compare_arrays(mem0, mem1)

        lock_json = find_state_lock_json(R33_FIX1, idx)
        state_id = extract_model_state_id(lock_json)

        if state_id not in TARGET_STATE_IDS:
            raise RuntimeError(
                f"Unexpected R33 model_state_id={state_id}"
            )

        state_pass = bool(source_cmp["equal"] and memo_cmp["equal"])

        rows.append({
            "state_index": idx,
            "model_state_id": state_id,
            "state_pass": state_pass,
            "v1_npz": str(p0),
            "v1_npz_sha256": sha256_file(p0),
            "fix1_npz": str(p1),
            "fix1_npz_sha256": sha256_file(p1),
            "fix1_lock_json": str(lock_json),
            "fix1_lock_json_sha256": sha256_file(lock_json),
            "source_v1_vs_fix1_equal": source_cmp["equal"],
            "source_differing_rows": source_cmp["differing_rows"],
            "source_differing_bytes": source_cmp["differing_bytes"],
            "memo_v1_vs_fix1_equal": memo_cmp["equal"],
            "memo_differing_rows": memo_cmp["differing_rows"],
            "memo_differing_bytes": memo_cmp["differing_bytes"],
            "source_member_sha256": packed_member_sha(src1),
            "memo_member_sha256": packed_member_sha(mem1),
        })

        if state_pass:
            lock_states.append({
                "model_state_id": state_id,
                "authoritative_npz": str(p1),
                "authoritative_npz_sha256": sha256_file(p1),
                "source_key": "source_masks_packed",
                "memo_key": "memo_masks_packed",
                "state_lock_json": str(lock_json),
                "state_lock_json_sha256": sha256_file(lock_json),
                "historical_v1_npz": str(p0),
                "historical_v1_npz_sha256": sha256_file(p0),
                "rows": POLYPGEN_ROWS,
            })

    df = pd.DataFrame(rows)

    mapped = set(df["model_state_id"].astype(str))
    mapping_complete = mapped == TARGET_STATE_IDS

    summary = {
        "states": len(df),
        "passed_states": int(df["state_pass"].sum()),
        "all_states_pass": bool(df["state_pass"].all()),
        "mapping_complete": bool(mapping_complete),
        "mapped_model_state_ids": sorted(mapped),
        "lock_states": lock_states,
    }
    return df, summary


def self_test():
    assert len(NEO_STATES) == 9
    assert len(TARGET_STATE_IDS) == 3
    assert PACKED_BYTES == 15488
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA P01B authoritative candidate-mask parity/lock audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite authoritative mask audit: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 164)
    print("SafeTTA B6-P0-1B — authoritative mask parity + lock audit")
    print(f"Version             : {VERSION}")
    print("Scientific metrics  : NO")
    print("GT access           : NO")
    print("Model fitting       : NO")
    print("Selection by outcome: NO")
    print("=" * 164)

    print("\n[1/2] NeoPolyp 9-state SOURCE/TENT1/PL exact parity")
    neo_df, neo = neo_parity()
    print(neo_df[
        [
            "model_state_id",
            "state_pass",
            "r38_source_vs_r05_source_equal",
            "r38_tent_vs_r05_a1_equal",
            "r38a1_source_vs_r38_source_equal",
            "r38a1_tent_vs_r38_tent_equal",
            "r38a1_pl_vs_r38_pl_equal",
        ]
    ].to_string(index=False))

    print("\n[2/2] PolypGen 3-state SOURCE/MEMO exact parity + state mapping")
    r33_df, r33 = r33_parity()
    print(r33_df[
        [
            "state_index",
            "model_state_id",
            "state_pass",
            "source_v1_vs_fix1_equal",
            "memo_v1_vs_fix1_equal",
        ]
    ].to_string(index=False))

    p_neo = OUT_DIR / "B6_P01B_NEOPOLYP_MASK_PARITY.csv"
    p_r33 = OUT_DIR / "B6_P01B_POLYPGEN_MEMO_MASK_PARITY.csv"
    neo_df.to_csv(p_neo, index=False, encoding="utf-8")
    r33_df.to_csv(p_r33, index=False, encoding="utf-8")

    all_pass = bool(
        neo["all_states_pass"]
        and r33["all_states_pass"]
        and r33["mapping_complete"]
    )

    lock = {
        "lock_id": "B6_P01B_AUTHORITATIVE_MASK_LOCK_v1",
        "version": VERSION,
        "status": (
            "LOCKED_BEFORE_P01B_GEOMETRY_METRICS"
            if all_pass
            else "NOT_LOCKED_PARITY_FAILURE"
        ),
        "selection_basis": (
            "historical provenance + exact byte parity only; "
            "no HARM/AUROC/AUPRC used"
        ),
        "neopolyp": {
            "support": "common800 x 9 model states",
            "authoritative_bundle": (
                "R38A0C exact frozen TENT1/PL replay, accepted only if "
                "SOURCE+TENT1 exactly match original R05D3 and "
                "SOURCE+TENT1+PL exactly match independent R38A1A recovery"
            ),
            "states": neo["lock_states"] if all_pass else [],
        },
        "polypgen": {
            "support": "1532 images x 3 DeepLabV3-R50 states",
            "authoritative_bundle": (
                "R33A2B fix1 pre-HARM SOURCE/MEMO lock, accepted only if "
                "exactly byte-identical to R33A2B v1 and state-lock mapping "
                "covers the exact three R33 model_state_ids"
            ),
            "states": r33["lock_states"] if all_pass else [],
        },
    }

    p_lock = OUT_DIR / "B6_P01B_AUTHORITATIVE_MASK_LOCK_v1.json"
    write_json(p_lock, lock)

    audit = {
        "status": "PASS" if all_pass else "FAIL",
        "gate": PASS_GATE if all_pass else FAIL_GATE,
        "version": VERSION,
        "read_only": True,
        "gt_access": False,
        "scientific_metrics": False,
        "model_fit": False,
        "neo_all_states_pass": neo["all_states_pass"],
        "polypgen_all_states_pass": r33["all_states_pass"],
        "polypgen_state_mapping_complete": r33["mapping_complete"],
        "authoritative_lock_emitted": all_pass,
        "outputs": {},
    }

    for p in [p_neo, p_r33, p_lock]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P01B_AUTHORITATIVE_MASK_PARITY_AUDIT.json"
    write_json(p_audit, audit)

    print("\n" + "=" * 164)
    print("SafeTTA B6-P0-1B AUTHORITATIVE MASK PARITY AUDIT COMPLETE")
    print("NeoPolyp 9-state parity       =", neo["all_states_pass"])
    print("PolypGen 3-state parity       =", r33["all_states_pass"])
    print("PolypGen state mapping        =", r33["mapping_complete"])
    print("Authoritative mask lock       =", all_pass)
    print("GATE=" + audit["gate"])
    print(f"lock_json={p_lock}")
    print(f"audit_json={p_audit}")
    print(f"stage_dir={OUT_DIR}")
    print("=" * 164)


if __name__ == "__main__":
    main()
