#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S01-A: Freeze the source train/validation split before any model training.

Input:
    F:\MEDSEG_SAFETTA\data\splits\S00_polyp_locked_manifest_v1.csv

Output:
    F:\MEDSEG_SAFETTA\data\splits\S01_locked_protocol_manifest_v1.csv
    F:\MEDSEG_SAFETTA\outputs\S01_A_freeze_source_train_val_split_v1\

Frozen roles:
    source_train   : 1305
    source_val     : 145
    seen_sanity    : 162
    unseen_locked  : 636

No target image or target mask is opened.
No model is trained.
No segmentation metric is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


VERSION = "2026-08-17-S01-A-v1"
BUILD = "S01_A_SOURCE_SPLIT_FREEZE_BEFORE_TRAINING"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_INPUT = ROOT / "data" / "splits" / "S00_polyp_locked_manifest_v1.csv"
DEFAULT_OUTPUT_MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "S01_A_freeze_source_train_val_split_v1"

SOURCE_POOL_N = 1450
SOURCE_TRAIN_N = 1305
SOURCE_VAL_N = 145
SEEN_SANITY_N = 162
UNSEEN_LOCKED_N = 636
TOTAL_N = 2248

SOURCE_SPLIT_SEED = 20260817
TRAINING_SEEDS = [20260817, 20260818, 20260819]

EXPECTED_BASE_SPLITS = {
    "source_train": 1450,
    "seen_test": 162,
    "unseen_test": 636,
}

EXPECTED_DATASET_COUNTS = {
    ("source_train", "SOURCE_COMBINED"): 1450,
    ("seen_test", "Kvasir-SEG"): 100,
    ("seen_test", "CVC-ClinicDB"): 62,
    ("unseen_test", "CVC-ColonDB"): 380,
    ("unseen_test", "CVC-300"): 60,
    ("unseen_test", "ETIS-LaribPolypDB"): 196,
}


def stable_rank(seed: int, sample_id: str) -> str:
    """
    Stable deterministic hash key independent of CSV row order,
    Python hash randomization, NumPy version, or filesystem order.
    """
    payload = f"{seed}::{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_input(rows, fields):
    required = {
        "sample_id",
        "split",
        "dataset",
        "pair_key",
        "image_relpath",
        "mask_relpath",
        "mask_encoding",
        "mask_semantically_valid",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Input manifest missing required columns: {missing}")

    if len(rows) != TOTAL_N:
        raise RuntimeError(
            f"Total manifest count mismatch: expected={TOTAL_N}, actual={len(rows)}"
        )

    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate sample_id values detected.")

    base_counts = Counter(r["split"] for r in rows)
    for split, expected in EXPECTED_BASE_SPLITS.items():
        actual = base_counts[split]
        if actual != expected:
            raise RuntimeError(
                f"Base split count mismatch for {split}: "
                f"expected={expected}, actual={actual}"
            )

    pair_counts = Counter((r["split"], r["dataset"]) for r in rows)
    for key, expected in EXPECTED_DATASET_COUNTS.items():
        actual = pair_counts[key]
        if actual != expected:
            raise RuntimeError(
                f"Dataset count mismatch for {key}: "
                f"expected={expected}, actual={actual}"
            )

    invalid_masks = [
        r["sample_id"] for r in rows
        if str(r["mask_semantically_valid"]).strip().lower() not in {"true", "1", "yes"}
    ]
    if invalid_masks:
        raise RuntimeError(
            f"Input manifest contains semantically invalid masks: "
            f"{invalid_masks[:10]}"
        )

    # A1 established that the actual source pool is fully hard 0/255.
    source_rows = [r for r in rows if r["split"] == "source_train"]
    source_encodings = Counter(r["mask_encoding"] for r in source_rows)
    if source_encodings != Counter({"hard_0_255": SOURCE_POOL_N}):
        raise RuntimeError(
            "Unexpected source-mask encoding distribution. "
            f"Expected all {SOURCE_POOL_N} hard_0_255; got {dict(source_encodings)}"
        )


def assign_roles(rows):
    source_rows = [r for r in rows if r["split"] == "source_train"]

    ranked = sorted(
        source_rows,
        key=lambda r: (stable_rank(SOURCE_SPLIT_SEED, r["sample_id"]), r["sample_id"]),
    )

    val_ids = {r["sample_id"] for r in ranked[:SOURCE_VAL_N]}
    train_ids = {r["sample_id"] for r in ranked[SOURCE_VAL_N:]}

    if len(val_ids) != SOURCE_VAL_N:
        raise RuntimeError("Source validation ID count mismatch.")
    if len(train_ids) != SOURCE_TRAIN_N:
        raise RuntimeError("Source training ID count mismatch.")
    if train_ids & val_ids:
        raise RuntimeError("Source train/val overlap detected.")

    out_rows = []
    for row in rows:
        new = dict(row)
        sid = row["sample_id"]

        if row["split"] == "source_train":
            if sid in val_ids:
                role = "source_val"
            elif sid in train_ids:
                role = "source_train"
            else:
                raise RuntimeError(f"Source sample not assigned: {sid}")

        elif row["split"] == "seen_test":
            role = "seen_sanity"

        elif row["split"] == "unseen_test":
            role = "unseen_locked"

        else:
            raise RuntimeError(f"Unexpected base split: {row['split']}")

        new["s01_role"] = role
        if row["split"] == "source_train":
            new["source_split_rank_sha256"] = stable_rank(
                SOURCE_SPLIT_SEED, row["sample_id"]
            )
        else:
            new["source_split_rank_sha256"] = ""

        out_rows.append(new)

    return out_rows


def validate_output(rows):
    role_counts = Counter(r["s01_role"] for r in rows)

    expected_roles = {
        "source_train": SOURCE_TRAIN_N,
        "source_val": SOURCE_VAL_N,
        "seen_sanity": SEEN_SANITY_N,
        "unseen_locked": UNSEEN_LOCKED_N,
    }

    if dict(role_counts) != expected_roles:
        # Counter dict ordering is irrelevant; compare exactly by key/value.
        if role_counts != Counter(expected_roles):
            raise RuntimeError(
                f"Role counts mismatch: expected={expected_roles}, "
                f"actual={dict(role_counts)}"
            )

    train_ids = {r["sample_id"] for r in rows if r["s01_role"] == "source_train"}
    val_ids = {r["sample_id"] for r in rows if r["s01_role"] == "source_val"}
    seen_ids = {r["sample_id"] for r in rows if r["s01_role"] == "seen_sanity"}
    unseen_ids = {r["sample_id"] for r in rows if r["s01_role"] == "unseen_locked"}

    sets = [train_ids, val_ids, seen_ids, unseen_ids]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise RuntimeError(
                    f"Role overlap detected between groups {i} and {j}"
                )

    if len(set().union(*sets)) != TOTAL_N:
        raise RuntimeError("Roles do not cover all samples exactly once.")

    # Unseen target remains completely locked for training/model selection.
    forbidden = [
        r["sample_id"]
        for r in rows
        if r["split"] == "unseen_test" and r["s01_role"] != "unseen_locked"
    ]
    if forbidden:
        raise RuntimeError(
            f"Unseen target leakage in S01 role assignment: {forbidden[:10]}"
        )

    return expected_roles


def freeze(
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    overwrite: bool,
):
    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Input manifest : {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"Output dir     : {output_dir}")
    print()

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    if output_manifest.exists() or output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "S01-A output already exists. Refusing to overwrite a frozen split. "
                "Use --overwrite only for a purely technical rerun before training."
            )
        if output_manifest.exists():
            output_manifest.unlink()
        if output_dir.exists():
            shutil.rmtree(output_dir)

    rows, fields = read_csv(input_manifest)
    validate_input(rows, fields)

    out_rows = assign_roles(rows)
    expected_roles = validate_output(out_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    out_fields = list(fields)
    for col in ["s01_role", "source_split_rank_sha256"]:
        if col not in out_fields:
            out_fields.append(col)

    write_csv(output_manifest, out_rows, out_fields)

    # Also save a byte-identical copy in the audit directory.
    shutil.copy2(output_manifest, output_dir / "S01_locked_protocol_manifest_v1.csv")

    manifest_sha = hashlib.sha256(output_manifest.read_bytes()).hexdigest()
    input_sha = hashlib.sha256(input_manifest.read_bytes()).hexdigest()

    source_val_rows = [
        {
            "sample_id": r["sample_id"],
            "dataset": r["dataset"],
            "pair_key": r["pair_key"],
            "image_relpath": r["image_relpath"],
            "mask_relpath": r["mask_relpath"],
            "source_split_rank_sha256": r["source_split_rank_sha256"],
        }
        for r in out_rows
        if r["s01_role"] == "source_val"
    ]
    write_csv(
        output_dir / "source_val_members.csv",
        source_val_rows,
        [
            "sample_id",
            "dataset",
            "pair_key",
            "image_relpath",
            "mask_relpath",
            "source_split_rank_sha256",
        ],
    )

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_sha,
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": manifest_sha,
        "source_split_seed": SOURCE_SPLIT_SEED,
        "source_split_method": (
            "stable SHA256 rank of '<seed>::<sample_id>'; "
            "first 145 samples = source_val, remaining 1305 = source_train"
        ),
        "training_seeds_frozen_for_s01": TRAINING_SEEDS,
        "role_counts": expected_roles,
        "target_images_opened": False,
        "target_masks_opened": False,
        "target_metrics_computed": False,
        "unseen_target_used_for_model_selection": False,
        "decision": "S01_A_SPLIT_FROZEN_READY_FOR_S01_B_SOURCE_ONLY_TRAINING",
    }

    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    protocol_text = f"""S01 SOURCE SPLIT FROZEN

Version: {VERSION}
Build: {BUILD}

Source pool: {SOURCE_POOL_N}
Source train: {SOURCE_TRAIN_N}
Source validation: {SOURCE_VAL_N}

Seen sanity: {SEEN_SANITY_N}
Unseen locked: {UNSEEN_LOCKED_N}

Source split seed: {SOURCE_SPLIT_SEED}
Source split method:
  SHA256("<seed>::<sample_id>") deterministic ranking.
  First {SOURCE_VAL_N} source samples -> source_val.
  Remaining {SOURCE_TRAIN_N} source samples -> source_train.

Training seeds:
  {TRAINING_SEEDS}

Rules:
- source_val may be used for checkpoint/epoch selection.
- seen_sanity is evaluation-only and may not select checkpoints or hyperparameters.
- unseen_locked is inaccessible to training/model-selection code.
- no target image or mask was opened by S01-A.
- this split must not be changed after model training starts.

Decision:
S01_A_SPLIT_FROZEN_READY_FOR_S01_B_SOURCE_ONLY_TRAINING
"""
    (output_dir / "S01_PROTOCOL_FROZEN.txt").write_text(
        protocol_text, encoding="utf-8"
    )

    summary = f"""===== S01-A SOURCE TRAIN/VAL SPLIT FREEZE =====
Script version: {VERSION}
Build: {BUILD}

Input manifest: {input_manifest}
Input SHA256: {input_sha}

Frozen roles:
  source_train : {SOURCE_TRAIN_N}
  source_val   : {SOURCE_VAL_N}
  seen_sanity  : {SEEN_SANITY_N}
  unseen_locked: {UNSEEN_LOCKED_N}
  TOTAL        : {TOTAL_N}

Source split seed: {SOURCE_SPLIT_SEED}
Training seeds: {TRAINING_SEEDS}
Split method: stable SHA256 rank by sample_id

Source-mask encoding check:
  hard_0_255 = {SOURCE_POOL_N}/{SOURCE_POOL_N}

Leakage controls:
  Target images opened=NO
  Target masks opened=NO
  Target metrics computed=NO
  unseen_locked used for model selection=NO

Output manifest SHA256: {manifest_sha}

Decision: S01_A_SPLIT_FROZEN_READY_FOR_S01_B_SOURCE_ONLY_TRAINING
"""
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"[OK] Frozen manifest: {output_manifest}")
    print(f"[OK] Audit outputs  : {output_dir}")


def self_test():
    # Determinism independent of row order.
    ids = [f"sample_{i:04d}" for i in range(200)]
    ranks_a = {
        sid: stable_rank(SOURCE_SPLIT_SEED, sid)
        for sid in ids
    }
    ranks_b = {
        sid: stable_rank(SOURCE_SPLIT_SEED, sid)
        for sid in reversed(ids)
    }
    assert ranks_a == ranks_b

    ordered_a = sorted(ids, key=lambda x: (ranks_a[x], x))
    ordered_b = sorted(reversed(ids), key=lambda x: (ranks_b[x], x))
    assert ordered_a == ordered_b

    assert stable_rank(SOURCE_SPLIT_SEED, "abc") == stable_rank(
        SOURCE_SPLIT_SEED, "abc"
    )
    assert stable_rank(SOURCE_SPLIT_SEED, "abc") != stable_rank(
        SOURCE_SPLIT_SEED + 1, "abc"
    )

    assert SOURCE_TRAIN_N + SOURCE_VAL_N == SOURCE_POOL_N
    assert SOURCE_POOL_N + SEEN_SANITY_N + UNSEEN_LOCKED_N == TOTAL_N

    print("STABLE_HASH_SPLIT_TEST_PASS")
    print("COUNT_INVARIANTS_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze S01 source train/validation split before training."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Technical rerun only, before any model training.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    freeze(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
