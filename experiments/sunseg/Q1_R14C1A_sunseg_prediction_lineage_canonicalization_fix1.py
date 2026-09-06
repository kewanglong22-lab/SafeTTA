#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C1A_sunseg_prediction_lineage_canonicalization_fix1.py

Purpose
-------
Canonicalize the already-completed R14C1 Fix2 SUN-SEG prediction lock without
rerunning any model inference.

Why this is needed
------------------
The successful terminal result is scientifically complete:
- 980 frames
- 49 physical cases
- 9 model states
- 8,820 model-frame rows
- SOURCE/TENT1/PL predictions locked before GT reveal
- no GT/Dice/HARM/safety-score access

However, the successful lock path is the Fix2 output directory. Fix2 migrated
the first six completed Fix1 states by copying their state CSV/lock files.
Those migrated CSV rows can retain artifact-relative-path metadata originating
from Fix1. Prediction NPZ bytes are valid, but the artifact-reference lineage
should be canonicalized before R14C2.

This script:
1. validates the complete Fix2 top lock and all nine state artifacts;
2. copies every prediction NPZ BYTE-FOR-BYTE unchanged;
3. rewrites only the three artifact-reference columns in each state CSV;
4. rebuilds each state lock with updated CSV SHA and explicit provenance;
5. rebuilds the global 8,820-row prediction manifest;
6. verifies every non-path scientific/data field is unchanged from Fix2;
7. writes a clean canonical Fix3-style R14C1 output lock.

Forbidden
---------
- SUN RGB decoding
- GT pixel decoding
- model loading/inference
- TTA execution
- Dice/DeltaDice computation
- HARM/BENEFIT access
- frozen safety-score access
- target tuning
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

SOURCE_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix2_v1"
)
DEST_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1"
)

SOURCE_TOP_LOCK = (
    SOURCE_DIR
    / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
)
SOURCE_GLOBAL_CSV = (
    SOURCE_DIR
    / "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"
)

EXPECTED_DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"
)
EXPECTED_FRAMES = 980
EXPECTED_CASES = 49
EXPECTED_STATES = 9
EXPECTED_ROWS = 8820
PACKED_BYTES = 15488

EXPECTED_FINAL_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)

GLOBAL_FIELDS = [
    "row_index",
    "sample_id",
    "external_case_id",
    "cluster_id",
    "clip_id",
    "source_split_label",
    "frame_member",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "source_foreground_pixels",
    "tent1_foreground_pixels",
    "source_tent1_changed_pixels",
    "pl_foreground_pixels",
    "source_pl_changed_pixels",
    "confident_pixels",
    "total_pixels",
    "confident_fraction",
    "pl_loss",
    "pl_parameter_max_abs_delta",
    "pl_buffer_max_abs_delta_after_adaptation",
    "pl_updated",
    "state_prediction_npz_relpath",
    "state_index_csv_relpath",
    "state_lock_json_relpath",
]

ARTIFACT_PATH_FIELDS = {
    "state_prediction_npz_relpath",
    "state_index_csv_relpath",
    "state_lock_json_relpath",
}

EXPECTED_PANEL = [
    ("PraNet", "20260817"),
    ("PraNet", "20260818"),
    ("PraNet", "20260819"),
    ("DeepLabV3-R50", "20260817"),
    ("DeepLabV3-R50", "20260818"),
    ("DeepLabV3-R50", "20260819"),
    ("SegFormer-B0", "20260820"),
    ("SegFormer-B0", "20260821"),
    ("SegFormer-B0", "20260822"),
]

DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_CANONICALIZED_"
    "BEFORE_GT_REVEAL"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def state_token(family: str, seed: str) -> str:
    return (
        family.lower().replace("-", "_").replace(" ", "_")
        + f"_seed{seed}"
    )


def state_paths(base: Path, family: str, seed: str):
    token = state_token(family, seed)
    d = base / "state_predictions"
    return (
        d / f"{token}_source_tent1_pl_predictions.npz",
        d / f"{token}_source_tent1_pl_index.csv",
        d / f"{token}_source_tent1_pl_lock.json",
    )


def relpath(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def validate_top_lock():
    if not SOURCE_TOP_LOCK.is_file():
        raise FileNotFoundError(SOURCE_TOP_LOCK)
    if not SOURCE_GLOBAL_CSV.is_file():
        raise FileNotFoundError(SOURCE_GLOBAL_CSV)

    lock = json.loads(
        SOURCE_TOP_LOCK.read_text(encoding="utf-8")
    )

    if lock.get("status") != "PASS":
        raise RuntimeError("Fix2 top lock is not PASS.")
    if lock.get("decision") != EXPECTED_DECISION:
        raise RuntimeError(
            f"Unexpected Fix2 decision: {lock.get('decision')}"
        )

    cohort = lock.get("cohort", {})
    if cohort.get("final_manifest_sha256") != EXPECTED_FINAL_MANIFEST_SHA:
        raise RuntimeError("Fix2 cohort manifest SHA mismatch.")
    if int(cohort.get("frames", -1)) != EXPECTED_FRAMES:
        raise RuntimeError("Fix2 frame count mismatch.")
    if int(cohort.get("physical_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("Fix2 physical-case count mismatch.")

    panel = lock.get("model_panel", {})
    if int(panel.get("states", -1)) != EXPECTED_STATES:
        raise RuntimeError("Fix2 state count mismatch.")
    if int(panel.get("model_frame_rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("Fix2 model-frame row count mismatch.")

    info = lock.get("information_boundary", {})
    required_false = (
        "sun_gt_pixels_decoded",
        "dice_computed",
        "delta_dice_computed",
        "harm_benefit_accessed",
        "frozen_safety_scores_accessed",
        "target_outcome_based_selection",
        "target_tuning",
    )
    for k in required_false:
        if bool(info.get(k, True)):
            raise RuntimeError(
                f"Fix2 information boundary violated or missing: {k}"
            )

    g = lock.get("artifacts", {}).get("global_manifest", {})
    if g.get("path") != str(SOURCE_GLOBAL_CSV):
        raise RuntimeError("Fix2 global-manifest path mismatch.")
    if int(g.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("Fix2 global-manifest lock rows mismatch.")

    actual_global_sha = sha256_file(SOURCE_GLOBAL_CSV)
    if actual_global_sha != g.get("sha256"):
        raise RuntimeError("Fix2 global-manifest SHA mismatch.")

    print("Fix2 top lock: PASS")
    print("Fix2 top lock SHA256:", sha256_file(SOURCE_TOP_LOCK))
    print("Fix2 global CSV SHA256:", actual_global_sha)

    return lock


def validate_global_csv():
    fields, rows = read_csv(SOURCE_GLOBAL_CSV)

    if fields != GLOBAL_FIELDS:
        raise RuntimeError("Fix2 global CSV schema mismatch.")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Fix2 global CSV row count mismatch.")

    keys = {
        (r["sample_id"], r["model_state_id"])
        for r in rows
    }
    if len(keys) != EXPECTED_ROWS:
        raise RuntimeError("Fix2 global keys are not unique.")

    per_sample = Counter(r["sample_id"] for r in rows)
    if len(per_sample) != EXPECTED_FRAMES:
        raise RuntimeError("Fix2 unique sample count mismatch.")
    if set(per_sample.values()) != {EXPECTED_STATES}:
        raise RuntimeError("Every frame must have exactly 9 states.")

    per_state = Counter(r["model_state_id"] for r in rows)
    if len(per_state) != EXPECTED_STATES:
        raise RuntimeError("Fix2 state-id count mismatch.")
    if set(per_state.values()) != {EXPECTED_FRAMES}:
        raise RuntimeError("Every state must have exactly 980 rows.")

    physical_cases = {r["cluster_id"] for r in rows}
    if len(physical_cases) != EXPECTED_CASES:
        raise RuntimeError("Fix2 global physical-case count mismatch.")

    forbidden = (
        "dice",
        "delta_dice",
        "harm",
        "benefit",
        "gt_path",
        "gt_pixels",
        "safety_score",
    )
    low_fields = [x.lower() for x in fields]
    for token in forbidden:
        if any(token in f for f in low_fields):
            raise RuntimeError(
                f"Forbidden pre-GT field found in global CSV: {token}"
            )

    print("Fix2 global manifest structure: PASS")
    return rows


def validate_and_canonicalize_state(
    family: str,
    seed: str,
    source_global_map,
):
    src_npz, src_idx, src_lock = state_paths(
        SOURCE_DIR,
        family,
        seed,
    )
    dst_npz, dst_idx, dst_lock = state_paths(
        DEST_DIR,
        family,
        seed,
    )

    for p in (src_npz, src_idx, src_lock):
        if not p.is_file():
            raise FileNotFoundError(p)

    lk = json.loads(src_lock.read_text(encoding="utf-8"))

    expected_state_id = f"{family}__seed{seed}"

    if lk.get("status") != "PASS":
        raise RuntimeError(f"{expected_state_id}: state lock not PASS.")
    if lk.get("model_state_id") != expected_state_id:
        raise RuntimeError(f"{expected_state_id}: lock state id mismatch.")
    if int(lk.get("target_cases", -1)) != EXPECTED_FRAMES:
        raise RuntimeError(f"{expected_state_id}: target count mismatch.")
    if int(lk.get("physical_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError(f"{expected_state_id}: case count mismatch.")

    npz_sha = sha256_file(src_npz)
    idx_sha = sha256_file(src_idx)

    if npz_sha != lk.get("prediction_npz_sha256"):
        raise RuntimeError(f"{expected_state_id}: NPZ SHA mismatch.")
    if idx_sha != lk.get("index_csv_sha256"):
        raise RuntimeError(f"{expected_state_id}: CSV SHA mismatch.")

    with np.load(src_npz, allow_pickle=False) as z:
        expected_arrays = (
            "source_masks_packed",
            "tent1_masks_packed",
            "pl_masks_packed",
        )
        if set(z.files) != set(expected_arrays):
            raise RuntimeError(
                f"{expected_state_id}: NPZ keys={z.files}"
            )
        for key in expected_arrays:
            if z[key].dtype != np.uint8:
                raise RuntimeError(
                    f"{expected_state_id}: {key} dtype={z[key].dtype}"
                )
            if z[key].shape != (EXPECTED_FRAMES, PACKED_BYTES):
                raise RuntimeError(
                    f"{expected_state_id}: {key} shape={z[key].shape}"
                )

    fields, rows = read_csv(src_idx)
    if fields != GLOBAL_FIELDS:
        raise RuntimeError(
            f"{expected_state_id}: state CSV schema mismatch."
        )
    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"{expected_state_id}: state CSV rows={len(rows)}"
        )

    for r in rows:
        if r["model_state_id"] != expected_state_id:
            raise RuntimeError(
                f"{expected_state_id}: row state id mismatch."
            )
        key = (r["sample_id"], r["model_state_id"])
        gr = source_global_map.get(key)
        if gr is None:
            raise RuntimeError(
                f"{expected_state_id}: missing global row {key}"
            )

        # Every scientific/data field must be byte-for-byte text-equal.
        for field in GLOBAL_FIELDS:
            if field in ARTIFACT_PATH_FIELDS:
                continue
            if str(r[field]) != str(gr[field]):
                raise RuntimeError(
                    f"{expected_state_id}: non-path field changed "
                    f"for {key}, field={field}: "
                    f"{r[field]!r} != {gr[field]!r}"
                )

    dst_npz.parent.mkdir(parents=True, exist_ok=True)

    # Prediction bytes are the evidence artifact: copy unchanged.
    shutil.copy2(src_npz, dst_npz)
    if sha256_file(dst_npz) != npz_sha:
        raise RuntimeError(
            f"{expected_state_id}: copied NPZ SHA changed."
        )

    pred_rel = relpath(dst_npz, DEST_DIR)
    idx_rel = relpath(dst_idx, DEST_DIR)
    lock_rel = relpath(dst_lock, DEST_DIR)

    corrected_rows = []
    for r in rows:
        rr = dict(r)
        rr["state_prediction_npz_relpath"] = pred_rel
        rr["state_index_csv_relpath"] = idx_rel
        rr["state_lock_json_relpath"] = lock_rel
        corrected_rows.append(rr)

    write_csv(dst_idx, corrected_rows, GLOBAL_FIELDS)
    dst_idx_sha = sha256_file(dst_idx)

    new_lk = dict(lk)
    new_lk["index_csv_sha256"] = dst_idx_sha
    new_lk["prediction_npz_sha256"] = npz_sha
    new_lk["canonicalized_from"] = {
        "source_directory": str(SOURCE_DIR),
        "source_state_lock_path": str(src_lock),
        "source_state_lock_sha256": sha256_file(src_lock),
        "source_prediction_npz_sha256": npz_sha,
        "source_index_csv_sha256": idx_sha,
    }
    new_lk["canonicalization"] = {
        "prediction_npz_byte_changed": False,
        "scientific_data_fields_changed": False,
        "changed_fields": sorted(ARTIFACT_PATH_FIELDS),
        "reason": (
            "Rewrite artifact-relative paths to the canonical Fix3 "
            "R14C1 output directory."
        ),
    }
    write_json(dst_lock, new_lk)

    # Revalidate destination.
    dst_lock_loaded = json.loads(
        dst_lock.read_text(encoding="utf-8")
    )
    if sha256_file(dst_npz) != dst_lock_loaded["prediction_npz_sha256"]:
        raise RuntimeError(
            f"{expected_state_id}: destination NPZ lock mismatch."
        )
    if sha256_file(dst_idx) != dst_lock_loaded["index_csv_sha256"]:
        raise RuntimeError(
            f"{expected_state_id}: destination CSV lock mismatch."
        )

    _, check_rows = read_csv(dst_idx)
    if len(check_rows) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"{expected_state_id}: destination rows mismatch."
        )

    print(
        f"{expected_state_id}: PASS | "
        f"NPZ unchanged={npz_sha[:12]}... | "
        f"metadata rows={len(corrected_rows)}"
    )

    audit = {
        "model_family": family,
        "training_seed": seed,
        "model_state_id": expected_state_id,
        "source_npz_sha256": npz_sha,
        "destination_npz_sha256": sha256_file(dst_npz),
        "npz_bytes_unchanged": 1,
        "source_index_csv_sha256": idx_sha,
        "destination_index_csv_sha256": dst_idx_sha,
        "rows": len(corrected_rows),
        "artifact_path_fields_rewritten": 3,
        "non_path_scientific_fields_changed": 0,
        "destination_state_lock_sha256": sha256_file(dst_lock),
    }

    return corrected_rows, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEST_DIR)
    args = ap.parse_args()

    # Frozen implementation intentionally supports only the preregistered paths.
    if args.source_dir != SOURCE_DIR:
        raise RuntimeError(
            f"Unexpected source directory: {args.source_dir}"
        )
    if args.output_dir != DEST_DIR:
        raise RuntimeError(
            f"Unexpected output directory: {args.output_dir}"
        )

    print(
        "===== Q1 R14C1A SUN-SEG PREDICTION LINEAGE "
        "CANONICALIZATION ====="
    )
    print("SOURCE=R14C1_FIX2_COMPLETED_PASS")
    print("DESTINATION=R14C1_FIX3_CANONICAL")
    print("MODEL_LOADING=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("SUN_RGB_DECODE=NO")
    print("SUN_GT_PIXEL_DECODE=NO")
    print("DICE_DELTADICE_COMPUTATION=NO")
    print("HARM_BENEFIT_ACCESS=NO")
    print("FROZEN_SAFETY_SCORE_ACCESS=NO")
    print("TARGET_TUNING=NO")
    print("PREDICTION_NPZ_BYTES_MUST_REMAIN_IDENTICAL=YES")

    if DEST_DIR.exists():
        raise FileExistsError(DEST_DIR)

    source_lock = validate_top_lock()
    source_global_rows = validate_global_csv()

    source_global_map = {
        (r["sample_id"], r["model_state_id"]): r
        for r in source_global_rows
    }
    if len(source_global_map) != EXPECTED_ROWS:
        raise RuntimeError("Source global map cardinality mismatch.")

    DEST_DIR.mkdir(parents=True, exist_ok=False)

    all_rows = []
    audit_rows = []

    try:
        print("\n===== PER-STATE CANONICALIZATION =====")
        for family, seed in EXPECTED_PANEL:
            rows, audit = validate_and_canonicalize_state(
                family,
                seed,
                source_global_map,
            )
            all_rows.extend(rows)
            audit_rows.append(audit)

        if len(all_rows) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Canonical global rows={len(all_rows)} "
                f"expected={EXPECTED_ROWS}"
            )

        # Verify all non-path scientific fields against the completed Fix2
        # global manifest before writing the canonical global manifest.
        new_map = {
            (r["sample_id"], r["model_state_id"]): r
            for r in all_rows
        }
        if set(new_map) != set(source_global_map):
            raise RuntimeError(
                "Canonical and Fix2 global key sets differ."
            )

        changed_nonpath = 0
        for key, old in source_global_map.items():
            new = new_map[key]
            for field in GLOBAL_FIELDS:
                if field in ARTIFACT_PATH_FIELDS:
                    continue
                if str(old[field]) != str(new[field]):
                    changed_nonpath += 1

        if changed_nonpath != 0:
            raise RuntimeError(
                f"Non-path scientific fields changed: {changed_nonpath}"
            )

        all_rows.sort(
            key=lambda r: (
                int(r["row_index"]),
                r["model_family"],
                int(r["training_seed"]),
            )
        )

        global_csv = (
            DEST_DIR
            / "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"
        )
        write_csv(global_csv, all_rows, GLOBAL_FIELDS)

        # Cardinality gates after canonicalization.
        per_sample = Counter(r["sample_id"] for r in all_rows)
        per_state = Counter(r["model_state_id"] for r in all_rows)
        physical_cases = {r["cluster_id"] for r in all_rows}

        if len(per_sample) != EXPECTED_FRAMES:
            raise RuntimeError("Canonical unique frame count mismatch.")
        if set(per_sample.values()) != {EXPECTED_STATES}:
            raise RuntimeError("Canonical frame state multiplicity mismatch.")
        if len(per_state) != EXPECTED_STATES:
            raise RuntimeError("Canonical state count mismatch.")
        if set(per_state.values()) != {EXPECTED_FRAMES}:
            raise RuntimeError("Canonical state row count mismatch.")
        if len(physical_cases) != EXPECTED_CASES:
            raise RuntimeError("Canonical physical-case count mismatch.")

        audit_csv = (
            DEST_DIR
            / "R14C1A_CANONICALIZATION_STATE_AUDIT.csv"
        )
        write_csv(
            audit_csv,
            audit_rows,
            [
                "model_family",
                "training_seed",
                "model_state_id",
                "source_npz_sha256",
                "destination_npz_sha256",
                "npz_bytes_unchanged",
                "source_index_csv_sha256",
                "destination_index_csv_sha256",
                "rows",
                "artifact_path_fields_rewritten",
                "non_path_scientific_fields_changed",
                "destination_state_lock_sha256",
            ],
        )

        # Preserve the completed scientific result, while explicitly recording
        # that this output is a metadata/lineage canonicalization.
        canonical_lock = dict(source_lock)
        canonical_lock["status"] = "PASS"
        canonical_lock["decision"] = EXPECTED_DECISION
        canonical_lock["canonicalization_decision"] = DECISION
        canonical_lock["canonicalized_from"] = {
            "source_output_directory": str(SOURCE_DIR),
            "source_top_lock_path": str(SOURCE_TOP_LOCK),
            "source_top_lock_sha256": sha256_file(SOURCE_TOP_LOCK),
            "source_global_manifest_path": str(SOURCE_GLOBAL_CSV),
            "source_global_manifest_sha256": sha256_file(
                SOURCE_GLOBAL_CSV
            ),
        }
        canonical_lock["canonicalization"] = {
            "prediction_npz_bytes_changed": False,
            "model_inference_rerun": False,
            "tta_rerun": False,
            "gt_accessed": False,
            "safety_score_accessed": False,
            "non_path_scientific_fields_changed": 0,
            "artifact_reference_fields_rewritten": sorted(
                ARTIFACT_PATH_FIELDS
            ),
            "states": EXPECTED_STATES,
            "rows": EXPECTED_ROWS,
        }

        canonical_lock["artifacts"] = {
            "global_manifest": {
                "path": str(global_csv),
                "sha256": sha256_file(global_csv),
                "rows": EXPECTED_ROWS,
            },
            "state_prediction_directory": str(
                DEST_DIR / "state_predictions"
            ),
            "canonicalization_state_audit": {
                "path": str(audit_csv),
                "sha256": sha256_file(audit_csv),
                "rows": EXPECTED_STATES,
            },
            "rgb_cache": {
                "copied": False,
                "technical_source_cache_only": str(
                    SOURCE_DIR / "rgb_cache"
                ),
            },
        }
        canonical_lock["next_stage"] = (
            "R14C2_SUNSEG_FROZEN_SOURCE_SAFETY_SCORE_LOCK_PRE_GT"
        )

        lock_path = (
            DEST_DIR
            / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
        )
        write_json(lock_path, canonical_lock)

        print("\n===== R14C1A FINAL =====")
        print("frames:", len(per_sample))
        print("physical cases:", len(physical_cases))
        print("model states:", len(per_state))
        print("model-frame rows:", len(all_rows))
        print("prediction NPZ bytes changed: NO")
        print("non-path scientific/data fields changed: 0")
        print("model inference rerun: NO")
        print("TTA rerun: NO")
        print("GT pixels decoded: NO")
        print("frozen safety score accessed: NO")
        print("Decision=", DECISION)
        print("LOCK=", lock_path)
        print("PASS")

    except Exception:
        # Never leave a partially canonicalized directory looking final.
        # The source Fix2 evidence remains untouched.
        if DEST_DIR.exists():
            shutil.rmtree(DEST_DIR)
        raise


if __name__ == "__main__":
    main()
