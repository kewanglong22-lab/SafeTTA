#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05C0 — Freeze PolypDB as untouched backup external confirmation cohort.

No PolypDB download.
No target image/mask decoding.
No model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import traceback
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05C0-v1-fix1"
BUILD = "Q1_R05C0_POLYPDB_TARGET_IDENTITY_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05C0_polypdb_backup_confirmatory_target_selection_lock_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "ccfb033adc08a467737633b7ba39840d8c2d338a27662f964fe684192d89813d"

R05B2_LOCK = (
    ROOT / "outputs"
    / "Q1_R05B2_source_only_paot_predictor_lock_v1"
    / "Q1_R05B2_SOURCE_PAOT_PREDICTOR_LOCK.json"
)
EXPECTED_R05B2_LOCK_SHA256 = (
    "bd182ac600afbb24b3a62ebd326931ab23c7cf6b873a2178e33500395c70f211"
)
EXPECTED_R05B2_DECISION = "SOURCE_ONLY_PAOT_PREDICTORS_LOCKED"
EXPECTED_PAOT_PACKAGES = 12

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05C0_polypdb_backup_confirmatory_target_selection_lock_v1_fix1"
)

DISQUALIFY_ROOTS = (
    ROOT / "data",
    ROOT / "outputs",
    ROOT / "cache",
)

REFERENCE_ROOTS = (
    ROOT / "code",
    ROOT / "docs",
)

TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".txt",
    ".log", ".md", ".yaml", ".yml",
}

MAX_TEXT_BYTES = 128 * 1024 * 1024

STRONG_TARGET_TOKENS = (
    "pr7ms",
    "debeshjha/polypdb",
)

GENERIC_TARGET_TOKEN = "polypdb"
PUBLICATION_ID = "2409.00045"

# Frozen, previously audited non-target identity:
# ETIS-LaribPolypDB is the ETIS dataset (196 images + 196 masks), not
# the 3934-image multi-center PolypDB selected in Q1-R05C0.
KNOWN_NON_TARGET_PATTERNS = (
    "etis-laribpolypdb",
    "etis_laribpolypdb",
    "etis laribpolypdb",
    "etis-larib polypdb",
    "etis larib polypdb",
)

SELF_TOKEN = "Q1_R05C0_polypdb_backup_confirmatory_target_selection_lock"

DECISION_LOCKED = "POLYPDB_BACKUP_CONFIRMATORY_TARGET_LOCKED"
DECISION_UNRESOLVED = "POLYPDB_BACKUP_CONFIRMATORY_TARGET_UNRESOLVED"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_self_artifact(path: Path) -> bool:
    return SELF_TOKEN.lower() in str(path).lower()


def strip_known_non_target_context(text: str):
    """
    Remove the already-audited ETIS-LaribPolypDB dataset name before
    classifying PolypDB identity.

    This is a dataset-identity correction, not a relaxation of the
    historical-exposure criterion.
    """
    low = text.lower()
    removed = 0

    for pattern in KNOWN_NON_TARGET_PATTERNS:
        count = low.count(pattern)
        if count:
            removed += count
            low = low.replace(pattern, " ")

    return low, removed


def evidence_tokens(text: str):
    """
    Return evidence for the 3934-image multi-center PolypDB target only.

    Strong identifiers:
      - OSF project token pr7ms
      - DebeshJha/PolypDB repository identity

    Generic "polypdb" counts only after removing the known ETIS-LaribPolypDB
    dataset identity.

    arXiv identifier 2409.00045 counts only in publication context.
    """
    low, _ = strip_known_non_target_context(text)
    hits = set()

    for token in STRONG_TARGET_TOKENS:
        if token in low:
            hits.add(token)

    if GENERIC_TARGET_TOKEN in low:
        hits.add(GENERIC_TARGET_TOKEN)

    for m in re.finditer(re.escape(PUBLICATION_ID), low):
        start = max(0, m.start() - 100)
        end = min(len(low), m.end() + 100)
        context = low[start:end]
        if "arxiv" in context or GENERIC_TARGET_TOKEN in context:
            hits.add(PUBLICATION_ID)
            break

    return sorted(hits)


def known_non_target_hit_count(text: str) -> int:
    _, removed = strip_known_non_target_context(text)
    return int(removed)


def validate_upstream():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05C0 protocol",
    )
    validate_sha(
        R05B2_LOCK,
        EXPECTED_R05B2_LOCK_SHA256,
        "R05B2 lock",
    )

    lock = json.loads(R05B2_LOCK.read_text(encoding="utf-8"))

    if lock.get("decision") != EXPECTED_R05B2_DECISION:
        raise RuntimeError(
            f"Unexpected R05B2 decision: {lock.get('decision')}"
        )

    if int(lock.get("predictor_packages", -1)) != EXPECTED_PAOT_PACKAGES:
        raise RuntimeError(
            f"Unexpected PAOT package count: "
            f"{lock.get('predictor_packages')}"
        )

    if bool(lock.get("target_data_used", True)):
        raise RuntimeError("R05B2 target-data boundary violated.")

    if bool(lock.get("target_image_pixels_opened", True)):
        raise RuntimeError("R05B2 target image boundary violated.")

    if bool(lock.get("target_mask_pixels_opened", True)):
        raise RuntimeError("R05B2 target mask boundary violated.")

    return {
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r05b2_decision": lock.get("decision"),
        "predictor_packages": lock.get("predictor_packages"),
        "primary_representation": lock.get("primary_representation"),
        "tasks": lock.get("tasks"),
        "target_data_used": lock.get("target_data_used"),
        "target_image_pixels_opened":
            lock.get("target_image_pixels_opened"),
        "target_mask_pixels_opened":
            lock.get("target_mask_pixels_opened"),
    }


def list_files(root: Path):
    files = []
    if not root.exists():
        return files

    for path in root.rglob("*"):
        try:
            if path.is_file() and not is_self_artifact(path):
                files.append(path)
        except OSError:
            continue

    return sorted(files)


def scan_disqualifying_assets():
    candidates = []
    for root in DISQUALIFY_ROOTS:
        for path in list_files(root):
            candidates.append((root, path))

    rows = []
    excluded_known_non_target_files = 0
    excluded_known_non_target_occurrences = 0

    for root, path in tqdm(
        candidates,
        desc="Q1-R05C0 historical PolypDB asset scan",
        unit="file",
        dynamic_ncols=True,
    ):
        path_text = str(path)
        path_hits = evidence_tokens(path_text)
        excluded_occurrences = known_non_target_hit_count(path_text)
        content_hits = []

        try:
            if (
                path.suffix.lower() in TEXT_SUFFIXES
                and path.stat().st_size <= MAX_TEXT_BYTES
            ):
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                content_hits = evidence_tokens(text)
                excluded_occurrences += known_non_target_hit_count(text)
        except Exception:
            content_hits = []

        if excluded_occurrences > 0:
            excluded_known_non_target_files += 1
            excluded_known_non_target_occurrences += excluded_occurrences

        hits = sorted(set(path_hits + content_hits))
        if not hits:
            continue

        rows.append({
            "root": str(root),
            "file": str(path),
            "file_size": path.stat().st_size,
            "path_hits": ";".join(path_hits),
            "content_hits": ";".join(content_hits),
            "all_hits": ";".join(hits),
            "disqualifying_historical_polypdb_asset": 1,
        })

    return (
        rows,
        len(candidates),
        excluded_known_non_target_files,
        excluded_known_non_target_occurrences,
    )


def scan_code_docs_references():
    rows = []
    candidates = []

    for root in REFERENCE_ROOTS:
        for path in list_files(root):
            try:
                if (
                    path.suffix.lower() in TEXT_SUFFIXES
                    and path.stat().st_size <= MAX_TEXT_BYTES
                ):
                    candidates.append((root, path))
            except OSError:
                continue

    for root, path in tqdm(
        candidates,
        desc="Q1-R05C0 code/docs PolypDB references",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        hits = evidence_tokens(str(path) + "\n" + text)
        if hits:
            rows.append({
                "root": str(root),
                "file": str(path),
                "tokens": ";".join(hits),
                "reference_only_not_data_exposure": 1,
            })

    return rows


def preflight():
    upstream = validate_upstream()

    print("===== Q1-R05C0 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05b2_lock_sha256={EXPECTED_R05B2_LOCK_SHA256}")
    print(f"r05b2_decision={upstream['r05b2_decision']}")
    print(f"frozen_paot_predictor_packages={upstream['predictor_packages']}")
    print("backup_confirmatory_dataset=PolypDB")
    print("frozen_external_target_use=ALL_3934")
    print("PolypDB_downloaded=NO")
    print("PolypDB_image_pixels_opened=NO")
    print("PolypDB_mask_pixels_opened=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    upstream = validate_upstream()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "R05C0 protocol copy",
    )

    (
        asset_rows,
        scanned_files,
        excluded_etis_identity_files,
        excluded_etis_identity_occurrences,
    ) = scan_disqualifying_assets()
    asset_path = build_dir / "historical_polypdb_asset_scan.csv"
    write_csv(
        asset_path,
        asset_rows,
        [
            "root",
            "file",
            "file_size",
            "path_hits",
            "content_hits",
            "all_hits",
            "disqualifying_historical_polypdb_asset",
        ],
    )

    reference_rows = scan_code_docs_references()
    reference_path = build_dir / "code_docs_reference_scan.csv"
    write_csv(
        reference_path,
        reference_rows,
        [
            "root",
            "file",
            "tokens",
            "reference_only_not_data_exposure",
        ],
    )

    upstream.update({
        "backup_confirmatory_dataset": "PolypDB",
        "frozen_external_target_use": "ALL_3934",
        "historical_data_output_cache_files_scanned": scanned_files,
        "historical_polypdb_disqualifying_assets": len(asset_rows),
        "known_non_target_identity": "ETIS-LaribPolypDB",
        "known_non_target_identity_files_excluded":
            excluded_etis_identity_files,
        "known_non_target_identity_occurrences_excluded":
            excluded_etis_identity_occurrences,
        "code_docs_reference_rows": len(reference_rows),
        "polypdb_downloaded_by_r05c0": False,
        "polypdb_image_pixels_opened_by_r05c0": False,
        "polypdb_mask_pixels_opened_by_r05c0": False,
        "target_model_inference_run": False,
        "target_tta_run": False,
        "target_predictor_fit": False,
    })

    upstream_path = build_dir / "upstream_r05b2_audit.json"
    write_json(upstream_path, upstream)

    checks = {
        "r05b2_lock_verified": True,
        "paot_predictor_packages_12":
            int(upstream["predictor_packages"]) == EXPECTED_PAOT_PACKAGES,
        "r05b2_target_data_closed":
            upstream["target_data_used"] is False,
        "historical_polypdb_assets_zero":
            len(asset_rows) == 0,
        "polypdb_not_downloaded_by_r05c0": True,
        "polypdb_pixels_not_opened_by_r05c0": True,
        "target_model_inference_not_run": True,
        "target_tta_not_run": True,
        "target_predictor_not_fit": True,
    }

    decision = (
        DECISION_LOCKED
        if all(checks.values())
        else DECISION_UNRESOLVED
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05C0 POLYPDB BACKUP CONFIRMATORY TARGET SELECTION LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen source predictor:",
        f"  R05B2 lock={EXPECTED_R05B2_LOCK_SHA256}",
        f"  PAOT predictor packages={upstream['predictor_packages']}",
        "",
        "Backup confirmatory target:",
        "  dataset=PolypDB",
        "  frozen use=ALL 3934 target-only images",
        "  PolypDB training use=NO",
        "  PolypDB validation/tuning use=NO",
        "",
        "Historical local exposure:",
        f"  data/outputs/cache files scanned={scanned_files}",
        "  known non-target identity=ETIS-LaribPolypDB",
        (
            "  excluded ETIS-LaribPolypDB files="
            f"{excluded_etis_identity_files}"
        ),
        (
            "  excluded ETIS-LaribPolypDB occurrences="
            f"{excluded_etis_identity_occurrences}"
        ),
        f"  TRUE disqualifying PolypDB assets={len(asset_rows)}",
        f"  code/docs reference rows={len(reference_rows)}",
        "",
        "Information boundary:",
        "  PolypDB downloaded=NO",
        "  PolypDB image pixels opened=NO",
        "  PolypDB mask pixels opened=NO",
        "  target model inference=NO",
        "  target TTA=NO",
        "  target predictor fit=NO",
        "",
        "Checks:",
    ]

    for key, passed in checks.items():
        lines.append(f"  {key}={'PASS' if passed else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    if decision == DECISION_LOCKED:
        lines += [
            "",
            "If locked:",
            "  next = Q1-R05C1 PolypDB public acquisition + manifest/pairing/overlap QC",
        ]
    else:
        lines += [
            "",
            "If unresolved:",
            "  inspect historical_polypdb_asset_scan.csv",
            "  do not download PolypDB yet",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "historical_polypdb_asset_scan": asset_path,
        "code_docs_reference_scan": reference_path,
        "upstream_r05b2_audit": upstream_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "frozen_source_paot_predictor_packages": EXPECTED_PAOT_PACKAGES,
        "backup_confirmatory_dataset": "PolypDB",
        "frozen_external_target_use": "ALL_3934",
        "polypdb_training_allowed": False,
        "polypdb_validation_tuning_allowed": False,
        "polypdb_downloaded": False,
        "polypdb_image_pixels_opened": False,
        "polypdb_mask_pixels_opened": False,
        "target_model_inference_run": False,
        "target_tta_run": False,
        "target_predictor_fit": False,
        "known_non_target_identity": "ETIS-LaribPolypDB",
        "known_non_target_identity_files_excluded":
            excluded_etis_identity_files,
        "known_non_target_identity_occurrences_excluded":
            excluded_etis_identity_occurrences,
        "historical_polypdb_disqualifying_assets": len(asset_rows),
        "decision": decision,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = build_dir / "Q1_R05C0_POLYPDB_SELECTION_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        path = build_dir / meta["relative_path"]
        if sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R05C0 LOCK:",
        args.output_dir / "Q1_R05C0_POLYPDB_SELECTION_LOCK.json",
    )
    print("Q1-R05C0 LOCK SHA256:", lock_sha)


def self_test():
    # True target identities.
    assert evidence_tokens("PolypDB") == ["polypdb"]
    assert evidence_tokens("https://osf.io/pr7ms/") == ["pr7ms"]
    assert evidence_tokens(
        "https://github.com/DebeshJha/PolypDB"
    ) == ["debeshjha/polypdb", "polypdb"]

    # Previously audited ETIS dataset identity must NEVER classify as the
    # new 3934-image PolypDB.
    etis_examples = [
        r"F:\MEDSEG_SAFETTA\data\processed\S00_polyp_locked_v1\unseen_test\ETIS-LaribPolypDB\images\1.png",
        r"ETIS-LaribPolypDB\masks\196.png",
        "ETIS_LaribPolypDB/images/10.png",
        "ETIS LaribPolypDB masks",
        "ETIS-Larib PolypDB/images/a.png",
    ]
    for value in etis_examples:
        assert evidence_tokens(value) == [], value
        assert known_non_target_hit_count(value) >= 1, value

    # A true target reference remains detectable even if the same text also
    # contains an ETIS path.
    mixed = (
        "old=ETIS-LaribPolypDB/images/1.png "
        "new=https://osf.io/pr7ms/ PolypDB"
    )
    assert evidence_tokens(mixed) == ["polypdb", "pr7ms"]

    # Publication context guard.
    assert evidence_tokens("2409.00045") == []
    assert evidence_tokens("arXiv:2409.00045") == ["2409.00045"]

    fake_self = Path(
        r"F:\MEDSEG_SAFETTA\outputs\Q1_R05C0_polypdb_backup_confirmatory_target_selection_lock_v1_fix1\run_log.txt"
    )
    assert is_self_artifact(fake_self)

    print("TRUE_POLYPDB_IDENTITY_TEST_PASS")
    print("ETIS_LARIBPOLYPDB_EXCLUSION_TEST_PASS")
    print("MIXED_CONTEXT_TARGET_PRESERVATION_TEST_PASS")
    print("PUBLICATION_CONTEXT_TEST_PASS")
    print("SELF_ARTIFACT_EXCLUSION_TEST_PASS")
    print("SELF_TEST_PASS")

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05C0: freeze public PolypDB as an untouched backup "
            "external PAOT confirmation cohort before downloading it."
        )
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.preflight_only:
        preflight()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
