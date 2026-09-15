#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-6B1 — FinalB3 MRI outcome-lineage audit.

READ-ONLY / NO NEW SCIENTIFIC ENDPOINTS.

Why this exists
---------------
P06B v1 could not reproduce FinalB3 SOURCE Dice by directly combining:
  FinalB1 current packed masks + CM6 frozen packed GT.

P06B0 proved:
- FinalB1 uses BITORDER="big";
- CM6 uses bitorder="little";
- but per-byte bit reversal alone does NOT globally make FinalB1 SOURCE/TENT1
  equal to the older CM5A masks, especially for SegFormer TENT1.

Therefore this audit determines the *actual FinalB3 outcome lineage* before any
patient-utility calculation is allowed.

It performs only:
1) static producer-code inspection for FinalB1 / FinalB3 / CM6;
2) lock/JSON path tracing;
3) inventory of already-frozen GT/mask-like packed arrays;
4) serialization-equivalence checks between packed GT candidates and CM6 GT.

It does NOT:
- fit models;
- run inference/TTA;
- compute new AUROC/AUPRC;
- compute deployment utility;
- choose score direction/coverage;
- modify any frozen asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Iterable

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P06B1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

FINALB1_SCRIPT = CODE / "Q1_FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3.py"
FINALB3_SCRIPT = CODE / "Q1_FinalB3_mri_action_specific_outcome_and_future_harm_v1.py"
CM6_SCRIPT = CODE / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1.py"

FINALB1_DIR = ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3"
FINALB3_DIR = ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1"
CM6_DIR = OUT / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"

CM6_GT = CM6_DIR / "PROMISE12_GT_MASKS_PACKBITS.npy"

OUT_DIR = ROOT / "B6_P06B1_mri_finalb3_outcome_lineage_audit_v1"

N_SLICES = 1377
PACKED_BYTES = 15488

BIT_REVERSE = np.asarray(
    [int(f"{i:08b}"[::-1], 2) for i in range(256)],
    dtype=np.uint8,
)

PASS_GATE = "PASS_B6_P06B1_MRI_FINALB3_OUTCOME_LINEAGE_AUDIT"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_context(path: Path, patterns: List[str], radius: int = 3) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "matches": []}

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pats = [p.lower() for p in patterns]
    matched_lines = []
    seen = set()

    for i, line in enumerate(lines):
        low = line.lower()
        if any(p in low for p in pats):
            lo = max(0, i - radius)
            hi = min(len(lines), i + radius + 1)
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            matched_lines.append({
                "start_line": lo + 1,
                "end_line": hi,
                "text": "\n".join(
                    f"L{j+1}: {lines[j]}" for j in range(lo, hi)
                ),
            })

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "matches": matched_lines[:120],
    }


def collect_json_refs(root: Path) -> List[Dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows

    for p in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"json": str(p), "error": repr(e)})
            continue

        refs = []
        def walk(x, prefix=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{prefix}[{i}]")
            elif isinstance(x, str):
                low = x.lower()
                if any(tok in low for tok in [
                    "gt", "mask", "promise", "finalb1", "cm6",
                    ".npy", ".npz", ".csv"
                ]):
                    refs.append({"key": prefix, "value": x})

        walk(obj)
        if refs:
            rows.append({
                "json": str(p),
                "sha256": sha256_file(p),
                "status": obj.get("status") if isinstance(obj, dict) else None,
                "gate": obj.get("gate") if isinstance(obj, dict) else None,
                "decision": obj.get("decision") if isinstance(obj, dict) else None,
                "refs": refs[:300],
            })
    return rows


def array_inventory(roots: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".npy", ".npz"}:
                continue
            k = str(p.resolve()).lower()
            if k in seen:
                continue
            seen.add(k)

            # We only care about MRI-sized packed arrays or explicitly GT/mask names.
            try:
                if p.suffix.lower() == ".npy":
                    a = np.load(p, mmap_mode="r", allow_pickle=False)
                    shape = list(a.shape)
                    dtype = str(a.dtype)
                    if (
                        tuple(a.shape) != (N_SLICES, PACKED_BYTES)
                        and not any(tok in p.name.lower() for tok in ["gt", "mask"])
                    ):
                        continue
                    rows.append({
                        "path": str(p),
                        "suffix": ".npy",
                        "shape": shape,
                        "dtype": dtype,
                        "sha256": sha256_file(p),
                    })
                else:
                    arrays = {}
                    with np.load(p, allow_pickle=False) as z:
                        for name in z.files:
                            arr = z[name]
                            if (
                                tuple(arr.shape) == (N_SLICES, PACKED_BYTES)
                                or any(tok in name.lower() for tok in ["gt", "mask"])
                            ):
                                arrays[name] = {
                                    "shape": list(arr.shape),
                                    "dtype": str(arr.dtype),
                                }
                    if arrays:
                        rows.append({
                            "path": str(p),
                            "suffix": ".npz",
                            "arrays": arrays,
                            "sha256": sha256_file(p),
                        })
            except Exception as e:
                rows.append({
                    "path": str(p),
                    "error": repr(e),
                })
    return rows


def load_exact_packed_npy(path: Path):
    try:
        a = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if a.shape != (N_SLICES, PACKED_BYTES) or a.dtype != np.uint8:
        return None
    return np.asarray(a)


def compare_to_cm6_gt(inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CM6_GT.is_file():
        raise FileNotFoundError(CM6_GT)
    cm6 = load_exact_packed_npy(CM6_GT)
    if cm6 is None:
        raise RuntimeError("CM6 GT packed array schema drift.")

    rows = []
    for item in inventory:
        p = Path(item.get("path", ""))
        if item.get("suffix") != ".npy":
            continue
        arr = load_exact_packed_npy(p)
        if arr is None:
            continue

        ident = np.array_equal(arr, cm6)
        rev = np.array_equal(BIT_REVERSE[arr], cm6)

        # Only report likely GT candidates or exact serialization equivalences.
        name_low = str(p).lower()
        gt_like = any(tok in name_low for tok in [
            "gt", "ground", "label", "segmentation"
        ])
        if not (gt_like or ident or rev):
            continue

        ident_rows = int(np.all(arr == cm6, axis=1).sum())
        rev_rows = int(np.all(BIT_REVERSE[arr] == cm6, axis=1).sum())

        rows.append({
            "path": str(p),
            "sha256": item.get("sha256"),
            "identity_exact": bool(ident),
            "bitreverse8_exact": bool(rev),
            "identity_exact_rows": ident_rows,
            "bitreverse8_exact_rows": rev_rows,
            "identity_mismatch_bytes": int(np.count_nonzero(arr != cm6)),
            "bitreverse8_mismatch_bytes": int(
                np.count_nonzero(BIT_REVERSE[arr] != cm6)
            ),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA FinalB3 MRI outcome-lineage audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        x = np.asarray([1, 128, 176], dtype=np.uint8)
        assert np.array_equal(BIT_REVERSE[BIT_REVERSE[x]], x)
        print("SELF_TEST_BIT_REVERSE=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 172)
    print("SafeTTA B6-P0-6B1 — FinalB3 MRI outcome-lineage audit")
    print("Version         :", VERSION)
    print("GT decode       : NO")
    print("Science metrics : NO")
    print("Inference/TTA   : NO")
    print("Asset writes    : NO (except audit outputs)")
    print("=" * 172)

    patterns = [
        "BITORDER",
        "packbits",
        "unpackbits",
        "PROMISE",
        "GT",
        "ground",
        "segmentation",
        "source_current_masks",
        "tent1_current_masks",
        "pl_conf90_masks",
        "memo_seg4",
        "FINALB1",
        "CM6",
        "ACTION_OUTCOMES",
        "source_dice",
        "action_dice",
        "delta_dice",
    ]

    print("\n[1/4] Producer-code lineage contexts")
    code = {
        "FinalB1": code_context(FINALB1_SCRIPT, patterns, radius=2),
        "FinalB3": code_context(FINALB3_SCRIPT, patterns, radius=2),
        "CM6": code_context(CM6_SCRIPT, patterns, radius=2),
    }

    for label, obj in code.items():
        print(f"\n===== {label} =====")
        print("path =", obj["path"])
        print("exists =", obj.get("exists"))
        print("sha256 =", obj.get("sha256"))
        for m in obj.get("matches", [])[:50]:
            print(f"\n[{m['start_line']}-{m['end_line']}]")
            print(m["text"])

    print("\n[2/4] Lock/JSON path tracing")
    json_rows = []
    for root in [FINALB1_DIR, FINALB3_DIR]:
        json_rows.extend(collect_json_refs(root))
    for r in json_rows:
        print("\nJSON:", r.get("json"))
        print(" status/gate/decision:", r.get("status"), r.get("gate"), r.get("decision"))
        for ref in r.get("refs", [])[:80]:
            print("  ", ref["key"], "=", ref["value"])

    print("\n[3/4] Frozen packed-array inventory")
    roots = [
        ROOT / "FinalB0_promise12_transition_centric_asset_audit_v1",
        ROOT / "FinalB0_promise12_transition_centric_asset_audit_v1_fix2",
        ROOT / "FinalB0A_mri_action_complete_common_support_lock_v1",
        FINALB1_DIR,
        ROOT / "FinalB2_mri_current_runtime_candidate_conditioned_s64_lock_v1",
        FINALB3_DIR,
        CM6_DIR,
    ]
    inv = array_inventory(roots)

    for r in inv:
        print(r)

    print("\n[4/4] GT serialization-equivalence against CM6 frozen GT")
    gt_cmp = compare_to_cm6_gt(inv)
    for r in gt_cmp:
        print(
            f"{r['path']}\n"
            f"  identity={r['identity_exact']} "
            f"rows={r['identity_exact_rows']}/{N_SLICES} "
            f"mismatch_bytes={r['identity_mismatch_bytes']}\n"
            f"  bitreverse8={r['bitreverse8_exact']} "
            f"rows={r['bitreverse8_exact_rows']}/{N_SLICES} "
            f"mismatch_bytes={r['bitreverse8_mismatch_bytes']}"
        )

    # Outcome-lineage decision is intentionally conservative.
    finalb3_exists = code["FinalB3"].get("exists", False)
    if not finalb3_exists:
        decision = "STOP_FINALB3_PRODUCER_SCRIPT_NOT_FOUND"
    else:
        exact_alt_gt = [
            r for r in gt_cmp
            if Path(r["path"]).resolve() != CM6_GT.resolve()
            and (r["identity_exact"] or r["bitreverse8_exact"])
        ]
        if exact_alt_gt:
            decision = "ALT_FROZEN_GT_SERIALIZATION_EQUIVALENT_TO_CM6_FOUND"
        else:
            decision = "NO_ALT_EXACT_GT_CACHE_FOUND_USE_FINALB3_CODE_CONTEXT_TO_BIND_GT_TRANSFORM"

    print("\nDECISION =", decision)

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite B6-P06B1 output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    p_code = OUT_DIR / "B6_P06B1_CODE_CONTEXT.json"
    p_json = OUT_DIR / "B6_P06B1_JSON_REF_TRACE.json"
    p_inv = OUT_DIR / "B6_P06B1_PACKED_ARRAY_INVENTORY.json"
    p_cmp = OUT_DIR / "B6_P06B1_GT_SERIALIZATION_COMPARISON.csv"
    p_dec = OUT_DIR / "B6_P06B1_OUTCOME_LINEAGE_DECISION.json"

    p_code.write_text(json.dumps(code, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p_json.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p_inv.write_text(json.dumps(inv, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(gt_cmp).to_csv(p_cmp, index=False)

    decision_obj = {
        "status": "PASS_AUDIT",
        "gate": PASS_GATE,
        "version": VERSION,
        "decision": decision,
        "finalb3_script": code["FinalB3"],
        "gt_comparisons": gt_cmp,
        "scientific_metrics_computed": False,
        "utility_computed": False,
        "next": (
            "Bind P06B recount to the exact FinalB3 GT/mask transform only after "
            "this audit identifies that transform."
        ),
    }
    p_dec.write_text(
        json.dumps(decision_obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("code_context =", p_code)
    print("json_trace   =", p_json)
    print("inventory    =", p_inv)
    print("gt_compare   =", p_cmp)
    print("decision     =", p_dec)
    print("GATE=" + PASS_GATE)
    print("=" * 172)


if __name__ == "__main__":
    main()
