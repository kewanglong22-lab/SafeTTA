#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1.py

Purpose
-------
Resolve the R10L2A "lineage incomplete" result using the project's
authoritative frozen S01 training manifest rather than generic path-column
heuristics.

Why R10L2A missed it
--------------------
The authoritative manifest uses:
    image_relpath
    mask_relpath
    s01_role
rather than generic `image_path` / `split` naming.

All three frozen model families are tied to the SAME frozen manifest SHA256:
    afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7

This audit verifies:
1) the exact 9 frozen checkpoint states;
2) the family training-code lineage to the same S01 manifest;
3) the current S01 manifest SHA and frozen role counts;
4) all source_train + source_val image files against manifest SHA256 metadata;
5) exact RGB SHA256 overlap against the locked 1532-case PolypGen cohort.

source_val is included because it selected checkpoints and therefore belongs
to the development lineage.

NO MODEL INFERENCE.
NO POLYPGEN PERFORMANCE.
NO TTA.
NO SAFETY SCORING.
NO THRESHOLD TUNING.
NO METHOD TUNING.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import pandas as pd
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")

EXPECTED_S01_SHA256 = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)

EXPECTED_ROLE_COUNTS = {
    "source_train": 1305,
    "source_val": 145,
    "seen_sanity": 162,
    "unseen_locked": 636,
}

EXPECTED_STATE_PANEL = {
    ("DeepLabV3-R50", 20260817),
    ("DeepLabV3-R50", 20260818),
    ("DeepLabV3-R50", 20260819),
    ("PraNet", 20260817),
    ("PraNet", 20260818),
    ("PraNet", 20260819),
    ("SegFormer-B0", 20260820),
    ("SegFormer-B0", 20260821),
    ("SegFormer-B0", 20260822),
}

DEFAULT_S01_MANIFEST = (
    "F:/MEDSEG_SAFETTA/data/splits/S01_locked_protocol_manifest_v1.csv"
)

DEFAULT_DATA_ROOT = (
    "F:/MEDSEG_SAFETTA/data/processed/S00_polyp_locked_v1"
)

DEFAULT_CHECKPOINT_RECOVERY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K1_higher_level_representation_asset_audit_fix1_v1/"
    "R10K1_checkpoint_recovery.csv"
)

DEFAULT_POLYPGEN_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1/"
    "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
)

DEFAULT_POLYPGEN_LOCK = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1/"
    "R10L1B_MANIFEST_LOCK.json"
)

DEFAULT_R10L2A_SUMMARY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L2A_source_training_lineage_discovery_audit_fix1_v1/"
    "R10L2A_summary.json"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L2B_authoritative_s01_source_training_overlap_lock_fix1_v1"
)

# Family-level source-training implementation evidence.
# We do not infer membership from these files; they establish that each family
# consumes the exact frozen S01 manifest and its source_train/source_val roles.
TRAINING_CODE = {
    "PraNet": ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py",
    "DeepLabV3-R50": (
        ROOT / "code" / "S05_B_train_deeplabv3_resnet50_source_only_frozen_seeds_v1.py"
    ),
    "SegFormer-B0": (
        ROOT / "code" / "Q1_R02_segformer_b0_three_seed_source_training_v1_fix7.py"
    ),
}


def sha256_file(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_lineage_audit(path, family):
    if not path.exists():
        return {
            "model_family": family,
            "training_code": str(path),
            "exists": False,
            "contains_s01_manifest_name": False,
            "contains_expected_manifest_sha": False,
            "contains_source_train": False,
            "contains_source_val": False,
            "lineage_pass": False,
        }

    text = path.read_text(encoding="utf-8", errors="replace")

    manifest_name = "S01_locked_protocol_manifest_v1.csv"
    checks = {
        "model_family": family,
        "training_code": str(path),
        "exists": True,
        "contains_s01_manifest_name": manifest_name in text,
        "contains_expected_manifest_sha": EXPECTED_S01_SHA256 in text,
        "contains_source_train": "source_train" in text,
        "contains_source_val": "source_val" in text,
    }
    checks["lineage_pass"] = all([
        checks["contains_s01_manifest_name"],
        checks["contains_expected_manifest_sha"],
        checks["contains_source_train"],
        checks["contains_source_val"],
    ])
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s01_manifest", default=DEFAULT_S01_MANIFEST)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--checkpoint_recovery",
        default=DEFAULT_CHECKPOINT_RECOVERY,
    )
    ap.add_argument(
        "--polypgen_manifest",
        default=DEFAULT_POLYPGEN_MANIFEST,
    )
    ap.add_argument(
        "--polypgen_lock",
        default=DEFAULT_POLYPGEN_LOCK,
    )
    ap.add_argument(
        "--r10l2a_summary",
        default=DEFAULT_R10L2A_SUMMARY,
    )
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    s01_path = Path(args.s01_manifest)
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint_recovery)
    pg_path = Path(args.polypgen_manifest)
    pg_lock_path = Path(args.polypgen_lock)
    l2a_path = Path(args.r10l2a_summary)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10L2B AUTHORITATIVE S01 SOURCE-TRAINING OVERLAP LOCK FIX1 =====")
    print("STATUS: TRAINING-LINEAGE / INDEPENDENCE GATE")
    print("AUTHORITATIVE TRAIN MEMBERSHIP: S01 s01_role")
    print("SOURCE_TRAIN INCLUDED: YES")
    print("SOURCE_VAL INCLUDED (checkpoint selection): YES")
    print("MODEL INFERENCE: NONE")
    print("POLYPGEN PERFORMANCE: NONE")
    print("TTA: NONE")
    print("SAFETY SCORING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("METHOD TUNING: NONE")
    print()

    for name, p in {
        "S01 manifest": s01_path,
        "S01 data root": data_root,
        "checkpoint recovery": checkpoint_path,
        "PolypGen manifest": pg_path,
        "PolypGen lock": pg_lock_path,
        "R10L2A summary": l2a_path,
    }.items():
        print(name, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    # --------------------------------------------------------------
    # Upstream locks.
    # --------------------------------------------------------------
    with open(pg_lock_path, "r", encoding="utf-8") as f:
        pg_lock = json.load(f)

    if pg_lock.get("decision") != (
        "POLYPGEN_STATIC_UNIQUE_MANIFEST_LOCKED_PENDING_SOURCE_TRAINING_AUDIT"
    ):
        raise AssertionError(
            f"Unexpected PolypGen lock: {pg_lock.get('decision')}"
        )

    with open(l2a_path, "r", encoding="utf-8") as f:
        l2a = json.load(f)

    print("\nR10L2A prior decision:", l2a.get("decision"))
    print(
        "R10L2A exact PolypGen training overlaps:",
        l2a.get("exact_polypgen_training_overlaps"),
    )

    # --------------------------------------------------------------
    # Exact frozen checkpoint panel.
    # --------------------------------------------------------------
    ck = pd.read_csv(checkpoint_path, low_memory=False)
    required_ck = [
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "recovered",
        "exact_sha256_match_count",
    ]
    missing = [c for c in required_ck if c not in ck.columns]
    if missing:
        raise AssertionError(f"Checkpoint table missing: {missing}")

    ck["training_seed"] = pd.to_numeric(
        ck["training_seed"], errors="raise"
    ).astype(int)

    observed_panel = set(
        zip(
            ck["model_family"].astype(str),
            ck["training_seed"].astype(int),
        )
    )

    print("\n===== FROZEN CHECKPOINT PANEL =====")
    print(
        ck[
            [
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
                "exact_sha256_match_count",
            ]
        ].to_string(index=False)
    )

    if len(ck) != 9:
        raise AssertionError(f"Expected 9 frozen states, got {len(ck)}.")
    if observed_panel != EXPECTED_STATE_PANEL:
        raise AssertionError(
            f"Frozen state panel mismatch: {sorted(observed_panel)}"
        )
    if not ck["recovered"].astype(bool).all():
        raise AssertionError("One or more frozen checkpoints not recovered.")
    if not (ck["exact_sha256_match_count"].astype(int) == 1).all():
        raise AssertionError(
            "Each frozen state must have exactly one SHA256 checkpoint match."
        )

    # --------------------------------------------------------------
    # Family training-code -> authoritative S01 bridge.
    # --------------------------------------------------------------
    family_rows = []
    for family, code_path in TRAINING_CODE.items():
        family_rows.append(code_lineage_audit(code_path, family))

    family_audit = pd.DataFrame(family_rows)

    print("\n===== FAMILY TRAINING-CODE -> S01 LINEAGE =====")
    print(family_audit.to_string(index=False))

    if not family_audit["lineage_pass"].all():
        raise AssertionError(
            "At least one model family is not explicitly tied to the "
            "frozen S01 manifest in its training code."
        )

    # --------------------------------------------------------------
    # Authoritative S01 manifest.
    # --------------------------------------------------------------
    manifest_sha = sha256_file(s01_path).lower()

    print("\n===== AUTHORITATIVE S01 MANIFEST =====")
    print("Manifest SHA256:", manifest_sha)
    print("Expected SHA256:", EXPECTED_S01_SHA256)

    if manifest_sha != EXPECTED_S01_SHA256:
        raise AssertionError(
            "S01 manifest SHA256 mismatch; refusing lineage reconstruction."
        )

    s01 = pd.read_csv(s01_path, low_memory=False)

    required_s01 = [
        "sample_id",
        "split",
        "dataset",
        "image_relpath",
        "mask_relpath",
        "image_file_sha256",
        "s01_role",
    ]
    missing = [c for c in required_s01 if c not in s01.columns]
    if missing:
        raise AssertionError(f"S01 manifest missing: {missing}")

    actual_counts = (
        s01["s01_role"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    print("Rows:", len(s01))
    print("Role counts:", actual_counts)

    if len(s01) != 2248:
        raise AssertionError(f"Expected 2248 S01 rows, got {len(s01)}.")
    if actual_counts != EXPECTED_ROLE_COUNTS:
        raise AssertionError(
            f"S01 role counts mismatch: {actual_counts}"
        )
    if s01["sample_id"].astype(str).nunique() != 2248:
        raise AssertionError("S01 sample_id is not unique.")

    dev = s01[
        s01["s01_role"].astype(str).isin(
            ["source_train", "source_val"]
        )
    ].copy()

    train = dev[
        dev["s01_role"].astype(str) == "source_train"
    ].copy()
    val = dev[
        dev["s01_role"].astype(str) == "source_val"
    ].copy()

    print("\nDevelopment rows used by frozen source models:")
    print("source_train:", len(train))
    print("source_val:", len(val))
    print("total train+val:", len(dev))
    print(
        "datasets:",
        dev["dataset"].astype(str).value_counts().to_dict(),
    )

    if len(train) != 1305 or len(val) != 145 or len(dev) != 1450:
        raise AssertionError("Frozen source development counts mismatch.")
    if set(dev["dataset"].astype(str)) != {"SOURCE_COMBINED"}:
        raise AssertionError(
            "Source development rows are not exclusively SOURCE_COMBINED."
        )
    if not (
        dev["split"].astype(str) == "source_train"
    ).all():
        raise AssertionError(
            "Frozen source train/val rows must originate from base split source_train."
        )

    # --------------------------------------------------------------
    # Verify all current source development files against frozen hashes.
    # --------------------------------------------------------------
    file_audit_rows = []

    for r in tqdm(
        dev.itertuples(index=False),
        total=len(dev),
        desc="Verifying S01 source train+val RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = data_root / str(r.image_relpath)
        exists = p.exists() and p.is_file()
        actual_sha = sha256_file(p).lower() if exists else ""
        expected_sha = str(r.image_file_sha256).strip().lower()

        file_audit_rows.append({
            "sample_id": r.sample_id,
            "s01_role": r.s01_role,
            "dataset": r.dataset,
            "image_relpath": r.image_relpath,
            "image_path": str(p),
            "exists": exists,
            "expected_image_sha256": expected_sha,
            "actual_image_sha256": actual_sha,
            "sha256_agreement": exists and actual_sha == expected_sha,
        })

    file_audit = pd.DataFrame(file_audit_rows)

    print("\nExisting source development images:",
          int(file_audit["exists"].sum()), "/", len(file_audit))
    print("Manifest SHA agreement:",
          int(file_audit["sha256_agreement"].sum()), "/", len(file_audit))

    if not file_audit["exists"].all():
        raise AssertionError("One or more frozen source train/val images missing.")
    if not file_audit["sha256_agreement"].all():
        raise AssertionError(
            "One or more current source images differ from the frozen S01 hash."
        )

    # --------------------------------------------------------------
    # Exact overlap with frozen PolypGen cohort.
    # --------------------------------------------------------------
    pg = pd.read_csv(pg_path, low_memory=False)

    if len(pg) != 1532:
        raise AssertionError(
            f"Expected 1532 frozen PolypGen rows, got {len(pg)}."
        )
    if "image_sha256" not in pg.columns:
        raise AssertionError("PolypGen manifest missing image_sha256.")

    pg_hashes = set(
        pg["image_sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )
    if len(pg_hashes) != 1532:
        raise AssertionError("PolypGen RGB hash set is not unique.")

    file_audit["exact_polypgen_overlap"] = (
        file_audit["actual_image_sha256"].isin(pg_hashes)
    )

    train_overlap = int(
        file_audit.loc[
            file_audit["s01_role"] == "source_train",
            "exact_polypgen_overlap",
        ].sum()
    )
    val_overlap = int(
        file_audit.loc[
            file_audit["s01_role"] == "source_val",
            "exact_polypgen_overlap",
        ].sum()
    )
    total_overlap = int(file_audit["exact_polypgen_overlap"].sum())

    print("\n===== POLYPGEN EXACT DEVELOPMENT-OVERLAP AUDIT =====")
    print("Frozen PolypGen cases:", len(pg))
    print("Source train exact overlaps:", train_overlap)
    print("Source val exact overlaps:", val_overlap)
    print("Total source train+val exact overlaps:", total_overlap)

    if total_overlap:
        print(
            file_audit[
                file_audit["exact_polypgen_overlap"]
            ].to_string(index=False)
        )

    # --------------------------------------------------------------
    # Per-state lineage result:
    # all nine states use the same S01 train/val development pool.
    # --------------------------------------------------------------
    family_map = {
        r["model_family"]: bool(r["lineage_pass"])
        for _, r in family_audit.iterrows()
    }

    state_rows = []
    for r in ck.itertuples(index=False):
        fam_ok = family_map[str(r.model_family)]

        if fam_ok and total_overlap == 0:
            status = (
                "FROZEN_S01_TRAIN_VAL_LINEAGE_CONFIRMED_"
                "ZERO_EXACT_POLYPGEN_OVERLAP"
            )
        elif total_overlap > 0:
            status = "POLYPGEN_EXACT_SOURCE_DEVELOPMENT_OVERLAP_DETECTED"
        else:
            status = "SOURCE_TRAINING_LINEAGE_NOT_CONFIRMED"

        state_rows.append({
            "model_state_id": r.model_state_id,
            "model_family": r.model_family,
            "training_seed": int(r.training_seed),
            "checkpoint_sha256": str(r.checkpoint_sha256),
            "s01_manifest_sha256": manifest_sha,
            "source_train_rows": 1305,
            "source_val_rows": 145,
            "source_train_exact_polypgen_overlap": train_overlap,
            "source_val_exact_polypgen_overlap": val_overlap,
            "lineage_status": status,
        })

    state_audit = pd.DataFrame(state_rows)

    print("\n===== PER-STATE FINAL LINEAGE =====")
    print(state_audit.to_string(index=False))

    if total_overlap > 0:
        decision = (
            "POLYPGEN_EXACT_SOURCE_DEVELOPMENT_OVERLAP_DETECTED_STOP"
        )
    elif (
        family_audit["lineage_pass"].all()
        and len(state_audit) == 9
        and state_audit["lineage_status"].str.startswith(
            "FROZEN_S01_TRAIN_VAL_LINEAGE_CONFIRMED"
        ).all()
    ):
        decision = (
            "POLYPGEN_INDEPENDENCE_GATE_PASS_"
            "ALL_9_STATES_S01_TRAIN_VAL_ZERO_EXACT_OVERLAP"
        )
    else:
        decision = (
            "POLYPGEN_INDEPENDENCE_GATE_NOT_RESOLVED"
        )

    # --------------------------------------------------------------
    # Outputs.
    # --------------------------------------------------------------
    family_audit.to_csv(
        out / "R10L2B_family_training_code_s01_lineage.csv",
        index=False,
    )
    file_audit.to_csv(
        out / "R10L2B_s01_train_val_image_hash_audit.csv",
        index=False,
    )
    state_audit.to_csv(
        out / "R10L2B_per_state_independence_audit.csv",
        index=False,
    )

    source_manifest = dev[
        [
            "sample_id",
            "s01_role",
            "dataset",
            "image_relpath",
            "image_file_sha256",
        ]
    ].copy()
    source_manifest.to_csv(
        out / "R10L2B_frozen_source_train_val_manifest.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "s01_manifest": str(s01_path),
        "s01_manifest_sha256": manifest_sha,
        "source_train_rows": len(train),
        "source_val_rows": len(val),
        "source_development_rows": len(dev),
        "source_files_verified_against_manifest_sha": int(
            file_audit["sha256_agreement"].sum()
        ),
        "polypgen_locked_cases": len(pg),
        "source_train_exact_polypgen_overlaps": train_overlap,
        "source_val_exact_polypgen_overlaps": val_overlap,
        "total_source_development_exact_polypgen_overlaps": total_overlap,
        "frozen_states": len(state_audit),
        "families_explicitly_tied_to_same_s01_manifest": int(
            family_audit["lineage_pass"].sum()
        ),
        "external_performance_evaluated": False,
        "model_inference_run": False,
    }

    with open(
        out / "R10L2B_INDEPENDENCE_GATE_LOCK.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)
    print("MODEL INFERENCE RUN: NO")
    print("EXTERNAL PERFORMANCE EVALUATED: NO")
    print("SOURCE TRAIN EXACT POLYPGEN OVERLAP:", train_overlap)
    print("SOURCE VAL EXACT POLYPGEN OVERLAP:", val_overlap)
    print("ALL 9 FROZEN STATES AUDITED:", len(state_audit) == 9)
    print("\nOutputs:")
    print(out / "R10L2B_family_training_code_s01_lineage.csv")
    print(out / "R10L2B_s01_train_val_image_hash_audit.csv")
    print(out / "R10L2B_per_state_independence_audit.csv")
    print(out / "R10L2B_INDEPENDENCE_GATE_LOCK.json")
    print("PASS")


if __name__ == "__main__":
    main()
