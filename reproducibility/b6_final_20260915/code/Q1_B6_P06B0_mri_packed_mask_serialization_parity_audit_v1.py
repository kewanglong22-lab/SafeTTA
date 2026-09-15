#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-6B0 — MRI packed-mask serialization parity audit.

READ-ONLY implementation audit.

Purpose:
Diagnose why direct bytewise TP/FP/FN recount from FinalB1 masks against the
CM6 packed GT does not reproduce FinalB3 slice Dice.

This audit does NOT compute new scientific endpoints. It compares exact frozen
mask assets representing the same SOURCE/TENT1 predictions:

  FinalB1 current masks
vs
  historical CM5A masks already used by CM6

The CM6 producer explicitly stores masks using np.packbits(..., bitorder="little").
If FinalB1 used NumPy's default packbits bitorder ("big"), exact physical masks
will differ only by per-byte bit reversal.

The audit checks:
1) producer-code serialization hints;
2) exact byte parity under identity;
3) exact byte parity after deterministic per-byte bit reversal;
4) SOURCE and TENT1 across both MRI families.

No GT is read.
No Dice/HARM/utility is computed.
No fitting/inference/TTA is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P06B0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

FINALB1_DIR = ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3"
CM5A_DIR = OUT / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1"

FINALB1_SCRIPT = CODE / "Q1_FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3.py"
CM6_SCRIPT = CODE / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1.py"

OUT_DIR = ROOT / "B6_P06B0_mri_packed_mask_serialization_parity_audit_v1"

FAMILIES = ["DeepLabV3_R50", "SegFormer_B0"]
N_SLICES = 1377
PACKED_BYTES = 15488

# byte -> byte whose 8 bits are reversed.
BIT_REVERSE = np.asarray(
    [int(f"{i:08b}"[::-1], 2) for i in range(256)],
    dtype=np.uint8,
)

PASS_GATE = "PASS_B6_P06B0_MRI_PACKED_MASK_SERIALIZATION_PARITY_AUDIT"


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_code_hints(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    selected = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if (
            "packbits" in low
            or "unpackbits" in low
            or "bitorder" in low
            or "packed" in low and "mask" in low
        ):
            selected.append({"line": i, "text": line.strip()})

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "serialization_lines": selected[:160],
        "contains_explicit_little": 'bitorder="little"' in text or "bitorder='little'" in text,
        "contains_explicit_big": 'bitorder="big"' in text or "bitorder='big'" in text,
        "contains_packbits_without_bitorder": bool(
            re.search(r"np\.packbits\s*\([^)]*\)", text, flags=re.S)
        ),
    }


def load_u8(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    if a.shape != (N_SLICES, PACKED_BYTES):
        raise RuntimeError(f"{path}: shape={a.shape}")
    if a.dtype != np.uint8:
        raise RuntimeError(f"{path}: dtype={a.dtype}")
    return np.asarray(a)


def compare_pair(new_path: Path, old_path: Path, family: str, role: str) -> Dict[str, Any]:
    new = load_u8(new_path)
    old = load_u8(old_path)

    identity_equal = np.array_equal(new, old)
    rev = BIT_REVERSE[new]
    bitreverse_equal = np.array_equal(rev, old)

    byte_mismatch_identity = int(np.count_nonzero(new != old))
    byte_mismatch_bitreverse = int(np.count_nonzero(rev != old))

    # Row-wise exact parity is useful if only some cases differ.
    row_identity = np.all(new == old, axis=1)
    row_bitreverse = np.all(rev == old, axis=1)

    return {
        "family": family,
        "role": role,
        "finalb1_path": str(new_path),
        "finalb1_sha256": sha256_file(new_path),
        "cm5a_path": str(old_path),
        "cm5a_sha256": sha256_file(old_path),
        "identity_exact": bool(identity_equal),
        "bitreverse8_exact": bool(bitreverse_equal),
        "identity_mismatch_bytes": byte_mismatch_identity,
        "bitreverse8_mismatch_bytes": byte_mismatch_bitreverse,
        "identity_exact_rows": int(row_identity.sum()),
        "bitreverse8_exact_rows": int(row_bitreverse.sum()),
        "rows": N_SLICES,
    }


def self_test():
    x = np.asarray([0b00000001, 0b10000000, 0b10110000], dtype=np.uint8)
    y = BIT_REVERSE[x]
    assert int(y[0]) == 0b10000000
    assert int(y[1]) == 0b00000001
    assert int(y[2]) == 0b00001101
    assert np.array_equal(BIT_REVERSE[y], x)
    print("SELF_TEST_BIT_REVERSE=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description="SafeTTA MRI packed-mask serialization parity audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 164)
    print("SafeTTA B6-P0-6B0 — MRI packed-mask serialization parity audit")
    print("Version           :", VERSION)
    print("GT read           : NO")
    print("Science metrics   : NO")
    print("Inference/TTA     : NO")
    print("Asset modification: NO")
    print("=" * 164)

    print("\n[1/3] Producer serialization hints")
    hints = {
        "FinalB1": read_code_hints(FINALB1_SCRIPT),
        "CM6": read_code_hints(CM6_SCRIPT),
    }
    for name, h in hints.items():
        print(f"\n{name} script: {h['path']}")
        print("exists =", h.get("exists"))
        print("explicit little =", h.get("contains_explicit_little"))
        print("explicit big    =", h.get("contains_explicit_big"))
        for row in h.get("serialization_lines", [])[:30]:
            print(f"  L{row['line']}: {row['text']}")

    print("\n[2/3] Exact frozen-mask parity")
    rows = []
    for fam in FAMILIES:
        final_base = FINALB1_DIR / "target_promise12" / fam
        old_base = CM5A_DIR / "predictions" / fam

        pairs = [
            (
                "SOURCE",
                final_base / "source_current_masks_packbits.npy",
                old_base / "source_masks_packbits.npy",
            ),
            (
                "TENT1",
                final_base / "tent1_current_masks_packbits.npy",
                old_base / "tent1_masks_packbits.npy",
            ),
        ]
        for role, newp, oldp in pairs:
            r = compare_pair(newp, oldp, fam, role)
            rows.append(r)
            print(
                f"{fam:16s} {role:6s} | "
                f"identity_exact={r['identity_exact']} "
                f"identity_rows={r['identity_exact_rows']}/{N_SLICES} | "
                f"bitreverse8_exact={r['bitreverse8_exact']} "
                f"bitreverse8_rows={r['bitreverse8_exact_rows']}/{N_SLICES} | "
                f"mismatch bytes id/rev="
                f"{r['identity_mismatch_bytes']}/{r['bitreverse8_mismatch_bytes']}"
            )

    df = pd.DataFrame(rows)

    print("\n[3/3] Decision")
    all_identity = bool(df["identity_exact"].all())
    all_bitreverse = bool(df["bitreverse8_exact"].all())

    if all_identity:
        decision = "FINALB1_SERIALIZATION_MATCHES_CM5A_IDENTITY"
        recommended = "Do not change packed-byte orientation; investigate row/preprocessing lineage."
    elif all_bitreverse:
        decision = "FINALB1_IS_EXACT_PER_BYTE_BIT_REVERSE_OF_CM5A"
        recommended = (
            "For CM6 little-bitorder GT recount, apply deterministic BIT_REVERSE "
            "to FinalB1 packed masks before TP/FP/FN bitwise counting. "
            "This is a serialization repair, not a scientific change."
        )
    else:
        decision = "NO_GLOBAL_EXACT_SERIALIZATION_EQUIVALENCE"
        recommended = (
            "Do not proceed to utility. Inspect FinalB1 generation/row ordering/preprocessing "
            "before changing P06B."
        )

    print("DECISION =", decision)
    print("RECOMMENDATION =", recommended)

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite audit output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    p_csv = OUT_DIR / "B6_P06B0_MASK_SERIALIZATION_PARITY.csv"
    p_json = OUT_DIR / "B6_P06B0_MASK_SERIALIZATION_AUDIT.json"
    df.to_csv(p_csv, index=False)

    payload = {
        "status": "PASS_AUDIT",
        "gate": PASS_GATE,
        "version": VERSION,
        "decision": decision,
        "recommendation": recommended,
        "producer_hints": hints,
        "pairs": rows,
        "scientific_change": False,
        "gt_read": False,
        "utility_computed": False,
    }
    p_json.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("audit_csv  =", p_csv)
    print("audit_json =", p_json)
    print("GATE=" + PASS_GATE)
    print("=" * 164)


if __name__ == "__main__":
    main()
