#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1.py

Pre-contamination, pre-inference SUN-SEG confirmatory candidate cohort lock.

Scientific decisions frozen here:
- Eligible domain = official TestEasyDataset/Unseen UNION TestHardDataset/Unseen.
- Physical cluster unit = base caseNN.
- Easy/Hard are NOT treated as independent cohorts when they share caseNN.
- Each eligible physical case contributes exactly 20 frames.
- Frame selection is deterministic, GT-blind, outcome-blind SHA256 ranking
  over pair_key with a fixed salt.
- No replacement after future contamination audit: if a selected frame is
  contaminated under the frozen R14B4 rule, the entire physical case will be
  excluded from confirmatory analysis.

This stage does NOT:
- extract archives;
- decode RGB or GT pixels;
- run SOURCE/TTA inference;
- read safety scores or outcomes;
- claim contamination-free independence.

It consumes the R14B2 Fix2 sanitized Frame<->GT manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"F:\MEDSEG_SAFETTA")

R14B2_DIR = (
    ROOT / "outputs"
    / "Q1_R14B2_sunseg_deep_schema_pairing_identity_audit_fix2_v1"
)

DEFAULT_PAIRS = R14B2_DIR / "R14B2_fix2_real_frame_gt_pairs.csv"
DEFAULT_LOCK = R14B2_DIR / "R14B2_FIX2_REAL_IMAGE_PAIRING_CASE_IDENTITY_LOCK.json"

DEFAULT_OUT = (
    ROOT / "outputs"
    / "Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1_v1"
)

EXPECTED_ARCHIVE_RAW_SHA = (
    "d9a00fada04782937a144e9d1bc3c2a3b7f8e321e37ba1cf83bebdd490563016"
)
EXPECTED_ARCHIVE_ANN_SHA = (
    "4ce0324c38743aeb47188dc18eded33258b2da8c07b7392a457dc55e77a1c7e7"
)

EXPECTED_ALL_REAL_PAIRS = 49136
EXPECTED_UNSEEN_FRAMES = 20991
EXPECTED_UNSEEN_CASES = 49

FRAMES_PER_CASE = 20
SAMPLING_SALT = "R14B3_SUNSEG_CONFIRMATORY_CASE_BALANCED_V1_20260903"

DECISION = (
    "SUNSEG_UNSEEN_CASE_BALANCED_CANDIDATE_COHORT_LOCKED_"
    "PENDING_CONTAMINATION_AUDIT"
)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def selection_hash(pair_key: str):
    payload = f"{SAMPLING_SALT}|{pair_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_label(row):
    if row["dataset"] == "TestEasyDataset":
        return "Easy-Unseen"
    if row["dataset"] == "TestHardDataset":
        return "Hard-Unseen"
    return f"{row['dataset']}-{row['visibility']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--r14b2-lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("===== Q1 R14B3 SUN-SEG UNSEEN CASE-BALANCED CANDIDATE LOCK =====")
    print("ELIGIBLE=Easy-Unseen UNION Hard-Unseen")
    print("PHYSICAL_CLUSTER_UNIT=caseNN")
    print("FRAMES_PER_CASE=", FRAMES_PER_CASE)
    print("SAMPLING=deterministic SHA256 rank")
    print("GT_PIXEL_DECODE=NO")
    print("RGB_PIXEL_DECODE=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("SAFETY_SCORE_ACCESS=NO")
    print("OUTCOME_ACCESS=NO")
    print("CONTAMINATION_CLAIM=NO")

    for p in (args.pairs, args.r14b2_lock):
        if not p.exists():
            raise FileNotFoundError(p)

    lock = json.loads(args.r14b2_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS":
        raise RuntimeError("R14B2 Fix2 lock is not PASS.")
    if lock.get("decision") != (
        "SUNSEG_REAL_IMAGE_PAIRING_AND_CASE_IDENTITY_AUDITED_PRE_EXTRACTION"
    ):
        raise RuntimeError("Unexpected R14B2 Fix2 decision.")
    shas = lock.get("input_sha256", {})
    if shas.get("raw_tar_gz") != EXPECTED_ARCHIVE_RAW_SHA:
        raise RuntimeError("R14B2 raw archive SHA mismatch.")
    if shas.get("annotation_zip") != EXPECTED_ARCHIVE_ANN_SHA:
        raise RuntimeError("R14B2 annotation archive SHA mismatch.")

    rows = read_csv(args.pairs)
    if len(rows) != EXPECTED_ALL_REAL_PAIRS:
        raise RuntimeError(
            f"R14B2 sanitized pair row count {len(rows)} "
            f"!= {EXPECTED_ALL_REAL_PAIRS}"
        )

    # Validate physical-case semantics against the full official split structure.
    train_cases = {
        r["physical_case_id"] for r in rows
        if r["dataset"] == "TrainDataset"
    }
    easy_seen_cases = {
        r["physical_case_id"] for r in rows
        if r["dataset"] == "TestEasyDataset" and r["visibility"] == "Seen"
    }
    hard_seen_cases = {
        r["physical_case_id"] for r in rows
        if r["dataset"] == "TestHardDataset" and r["visibility"] == "Seen"
    }
    easy_unseen_cases = {
        r["physical_case_id"] for r in rows
        if r["dataset"] == "TestEasyDataset" and r["visibility"] == "Unseen"
    }
    hard_unseen_cases = {
        r["physical_case_id"] for r in rows
        if r["dataset"] == "TestHardDataset" and r["visibility"] == "Unseen"
    }

    print("\n===== SPLIT SEMANTICS GATE =====")
    print("Train cases:", len(train_cases))
    print("Easy-Seen cases:", len(easy_seen_cases))
    print("Hard-Seen cases:", len(hard_seen_cases))
    print("Easy-Unseen cases:", len(easy_unseen_cases))
    print("Hard-Unseen cases:", len(hard_unseen_cases))
    print("Easy-Seen outside Train:", len(easy_seen_cases - train_cases))
    print("Hard-Seen outside Train:", len(hard_seen_cases - train_cases))
    print("Easy-Unseen intersect Train:", len(easy_unseen_cases & train_cases))
    print("Hard-Unseen intersect Train:", len(hard_unseen_cases & train_cases))
    print(
        "Easy-Unseen intersect Hard-Unseen:",
        len(easy_unseen_cases & hard_unseen_cases)
    )

    if easy_seen_cases - train_cases:
        raise RuntimeError("Easy-Seen contains physical cases outside Train.")
    if hard_seen_cases - train_cases:
        raise RuntimeError("Hard-Seen contains physical cases outside Train.")
    if easy_unseen_cases & train_cases:
        raise RuntimeError("Easy-Unseen overlaps Train by physical case.")
    if hard_unseen_cases & train_cases:
        raise RuntimeError("Hard-Unseen overlaps Train by physical case.")

    eligible = [
        r for r in rows
        if r["visibility"] == "Unseen"
        and r["dataset"] in {"TestEasyDataset", "TestHardDataset"}
    ]

    eligible_cases = sorted({r["physical_case_id"] for r in eligible})
    if len(eligible) != EXPECTED_UNSEEN_FRAMES:
        raise RuntimeError(
            f"Eligible unseen frame count {len(eligible)} "
            f"!= {EXPECTED_UNSEEN_FRAMES}"
        )
    if len(eligible_cases) != EXPECTED_UNSEEN_CASES:
        raise RuntimeError(
            f"Eligible unseen case count {len(eligible_cases)} "
            f"!= {EXPECTED_UNSEEN_CASES}"
        )

    by_case = defaultdict(list)
    for r in eligible:
        by_case[r["physical_case_id"]].append(r)

    min_frames = min(len(v) for v in by_case.values())
    max_frames = max(len(v) for v in by_case.values())
    print("\n===== ELIGIBLE UNSEEN CANDIDATE POOL =====")
    print("frames:", len(eligible))
    print("physical cases:", len(eligible_cases))
    print("min frames/case:", min_frames)
    print("max frames/case:", max_frames)

    if min_frames < FRAMES_PER_CASE:
        raise RuntimeError(
            f"At least one physical case has fewer than "
            f"{FRAMES_PER_CASE} eligible frames."
        )

    selected = []
    case_summary = []

    for case_id in eligible_cases:
        candidates = by_case[case_id]
        ranked = []
        for r in candidates:
            h = selection_hash(r["pair_key"])
            ranked.append((h, r["pair_key"], r))
        ranked.sort(key=lambda x: (x[0], x[1]))
        chosen = ranked[:FRAMES_PER_CASE]

        split_counts = Counter()
        clip_ids = set()

        for rank, (h, _, r) in enumerate(chosen, start=1):
            out = dict(r)
            out["external_case_id"] = f"SUNSEG_{case_id}"
            out["cluster_id"] = case_id
            out["sampling_rank_within_case"] = rank
            out["sampling_hash"] = h
            out["source_split_label"] = split_label(r)
            out["precontamination_status"] = "SELECTED_PENDING_CONTAMINATION"
            selected.append(out)
            split_counts[split_label(r)] += 1
            clip_ids.add(r["clip_id"])

        case_summary.append({
            "physical_case_id": case_id,
            "eligible_frame_count": len(candidates),
            "selected_frame_count": FRAMES_PER_CASE,
            "eligible_clip_count": len({r["clip_id"] for r in candidates}),
            "selected_clip_count": len(clip_ids),
            "selected_easy_unseen": split_counts["Easy-Unseen"],
            "selected_hard_unseen": split_counts["Hard-Unseen"],
        })

    if len(selected) != EXPECTED_UNSEEN_CASES * FRAMES_PER_CASE:
        raise RuntimeError("Selected row count is not exactly 49*20.")

    # Determinism re-check.
    for case_id in eligible_cases:
        rows_case = [r for r in selected if r["cluster_id"] == case_id]
        if len(rows_case) != FRAMES_PER_CASE:
            raise RuntimeError(f"Case balance failed for {case_id}.")
        if sorted(int(r["sampling_rank_within_case"]) for r in rows_case) != list(
            range(1, FRAMES_PER_CASE + 1)
        ):
            raise RuntimeError(f"Sampling ranks invalid for {case_id}.")

    selected_split_counts = Counter(r["source_split_label"] for r in selected)
    selected_clip_count = len({(r["cluster_id"], r["clip_id"]) for r in selected})

    print("\n===== FROZEN PRE-CONTAMINATION SAMPLE =====")
    print("selected frames:", len(selected))
    print("selected physical cases:", len({r["cluster_id"] for r in selected}))
    print("selected case-clip pairs:", selected_clip_count)
    for k, v in selected_split_counts.items():
        print(f"{k}: {v}")
    print("all cases exactly 20 frames: YES")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    selected_fields = list(selected[0].keys())
    manifest_path = (
        args.output_dir
        / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_MANIFEST.csv"
    )
    write_csv(manifest_path, selected, selected_fields)

    case_summary_path = (
        args.output_dir
        / "R14B3_SUNSEG_CASE_BALANCED_SAMPLING_SUMMARY.csv"
    )
    write_csv(
        case_summary_path,
        case_summary,
        [
            "physical_case_id",
            "eligible_frame_count",
            "selected_frame_count",
            "eligible_clip_count",
            "selected_clip_count",
            "selected_easy_unseen",
            "selected_hard_unseen",
        ],
    )

    selection_sha = sha256_file(manifest_path)

    lock_out = {
        "status": "PASS",
        "decision": DECISION,
        "r14b2_lock_path": str(args.r14b2_lock),
        "r14b2_lock_sha256": sha256_file(args.r14b2_lock),
        "r14b2_pair_manifest_sha256": sha256_file(args.pairs),
        "archive_sha256": {
            "raw_tar_gz": EXPECTED_ARCHIVE_RAW_SHA,
            "annotation_zip": EXPECTED_ARCHIVE_ANN_SHA,
        },
        "eligibility": {
            "included": [
                "TestEasyDataset/Unseen",
                "TestHardDataset/Unseen",
            ],
            "excluded": [
                "TrainDataset",
                "TestEasyDataset/Seen",
                "TestHardDataset/Seen",
            ],
            "reason": (
                "Both official Unseen partitions have zero physical-case "
                "overlap with Train; Easy-Unseen and Hard-Unseen share 16 "
                "physical cases and are therefore unioned at case level "
                "rather than treated as independent cohorts."
            ),
            "eligible_frame_count": len(eligible),
            "eligible_physical_case_count": len(eligible_cases),
        },
        "sampling": {
            "physical_cluster_unit": "caseNN",
            "frames_per_case": FRAMES_PER_CASE,
            "method": (
                "Sort eligible pair_key values within each physical case by "
                "SHA256(SAMPLING_SALT|pair_key), then select first 20."
            ),
            "sampling_salt": SAMPLING_SALT,
            "selected_frame_count": len(selected),
            "selected_physical_case_count": len(
                {r["cluster_id"] for r in selected}
            ),
            "selected_manifest_sha256": selection_sha,
        },
        "future_contamination_rule": {
            "unit_of_exclusion": "entire_physical_case",
            "replacement_after_case_exclusion": False,
            "note": (
                "R14B4 will audit selected RGB frames against S01, NeoPolyp "
                "and PolypGen. Any frozen contamination criterion triggered "
                "by any selected frame excludes that whole physical case; "
                "no replacement frame/case will be sampled."
            ),
        },
        "primary_future_inference_unit": (
            "selected_frame x 9 frozen model states"
        ),
        "primary_future_statistics_cluster": "physical_case_id",
        "information_boundary": {
            "archive_extracted": False,
            "rgb_pixels_decoded": False,
            "gt_pixels_decoded": False,
            "model_inference": False,
            "tta_execution": False,
            "safety_scores_accessed": False,
            "target_outcomes_accessed": False,
            "contamination_claim": False,
            "final_confirmatory_cohort": False,
        },
        "next_stage": (
            "R14B4_SELECTED_RGB_CONTAMINATION_AUDIT_AGAINST_"
            "S01_NEOPOLYP_POLYPGEN_BEFORE_INFERENCE"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(lock_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nmanifest:", manifest_path)
    print("manifest SHA256:", selection_sha)
    print("Decision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
