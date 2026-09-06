#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05D0 — Freeze BKAI-IGH NeoPolyp-Small as backup external confirmation.

One-shot identity/exposure lock:
- no NeoPolyp download
- no image/GT access
- no repeated manual rescue audit
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


VERSION = "2026-08-19-Q1-R05D0-v1"
BUILD = "Q1_R05D0_NEOPOLYP_BACKUP_CONFIRMATORY_TARGET_SELECTION_LOCK"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05D0_neopolyp_backup_confirmatory_target_selection_lock_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "d19cc826032f534d92152b642cea1c5a59ac80988a46452bc981f4e624b218fd"

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
    / "Q1_R05D0_neopolyp_backup_confirmatory_target_selection_lock_v1"
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

STRONG_ID_PATTERNS = (
    "bkai-igh-neopolyp",
    "bkai_igh_neopolyp",
    "bkai-igh_neopolyp",
    "bkai_igh-neopolyp",
    "bkai-igh neopolyp",
    "bkai igh neopolyp",
    "neopolyp-small",
    "neopolyp_small",
)

OUTCOME_USE_TOKENS = (
    "dice",
    "delta",
    "prediction",
    "probability",
    "score",
    "loss",
    "harm",
    "benefit",
    "inference",
    "tta",
    "checkpoint",
    "train_gt",
)

SELF_TOKEN = "Q1_R05D0_neopolyp_backup_confirmatory_target_selection_lock"

DECISION_LOCKED = "NEOPOLYP_BACKUP_CONFIRMATORY_TARGET_LOCKED"
DECISION_RETIRED = "NEOPOLYP_BACKUP_CONFIRMATORY_TARGET_RETIRED"


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
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_self_artifact(path: Path) -> bool:
    return SELF_TOKEN.lower() in str(path).lower()


def strong_identity_hits(text: str):
    low = text.lower()
    return sorted({
        p for p in STRONG_ID_PATTERNS
        if p in low
    })


def local_outcome_use_hits(text: str):
    low = text.lower()
    return sorted({
        token for token in OUTCOME_USE_TOKENS
        if token in low
    })


def validate_upstream():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05D0 protocol",
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
            f"R05B2 predictor packages={lock.get('predictor_packages')}"
        )

    if bool(lock.get("target_data_used", True)):
        raise RuntimeError("R05B2 target data boundary violated.")
    if bool(lock.get("target_image_pixels_opened", True)):
        raise RuntimeError("R05B2 target image boundary violated.")
    if bool(lock.get("target_mask_pixels_opened", True)):
        raise RuntimeError("R05B2 target mask boundary violated.")

    return {
        "r05b2_lock_sha256": EXPECTED_R05B2_LOCK_SHA256,
        "r05b2_decision": lock.get("decision"),
        "predictor_packages": int(lock.get("predictor_packages")),
        "target_data_used": lock.get("target_data_used"),
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


def text_local_exposure(path: Path):
    """
    Text-only content is disqualifying only when strong NeoPolyp identity and
    model-use/outcome evidence occur locally (same line +/- one line).
    """
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception:
        return []

    hits = []

    for i, line in enumerate(lines):
        ids = strong_identity_hits(line)
        if not ids:
            continue

        start = max(0, i - 1)
        end = min(len(lines), i + 2)
        context = "\n".join(lines[start:end])
        uses = local_outcome_use_hits(context)

        if uses:
            hits.append({
                "location": f"line:{i+1}",
                "identity_tokens": ";".join(ids),
                "outcome_use_tokens": ";".join(uses),
            })

    return hits


def scan_historical_exposure():
    candidates = []
    for root in DISQUALIFY_ROOTS:
        for path in list_files(root):
            candidates.append((root, path))

    rows = []

    for root, path in tqdm(
        candidates,
        desc="Q1-R05D0 NeoPolyp one-shot exposure scan",
        unit="file",
        dynamic_ncols=True,
    ):
        path_ids = strong_identity_hits(str(path))

        # Strong identity in the DATA/CACHE path is direct asset evidence.
        # For OUTPUT text files, require local model-use/outcome evidence;
        # this avoids retiring a target for a simple narrative mention.
        root_name = root.name.lower()

        if path_ids and root_name in {"data", "cache"}:
            rows.append({
                "root": str(root),
                "file": str(path),
                "evidence_type": "strong_identity_in_data_or_cache_path",
                "location": "path",
                "identity_tokens": ";".join(path_ids),
                "outcome_use_tokens": "",
            })
            continue

        if (
            path.suffix.lower() in TEXT_SUFFIXES
            and path.stat().st_size <= MAX_TEXT_BYTES
        ):
            local_hits = text_local_exposure(path)
            for hit in local_hits:
                rows.append({
                    "root": str(root),
                    "file": str(path),
                    "evidence_type": "local_identity_plus_model_use_or_outcome",
                    "location": hit["location"],
                    "identity_tokens": hit["identity_tokens"],
                    "outcome_use_tokens": hit["outcome_use_tokens"],
                })

        elif path_ids:
            # Non-text binary artifact whose own path is strongly NeoPolyp
            # identified is treated as direct historical data/model asset.
            rows.append({
                "root": str(root),
                "file": str(path),
                "evidence_type": "strong_identity_binary_path",
                "location": "path",
                "identity_tokens": ";".join(path_ids),
                "outcome_use_tokens": "",
            })

    # Deduplicate exact evidence rows.
    dedup = {}
    for r in rows:
        key = (
            r["file"],
            r["evidence_type"],
            r["location"],
            r["identity_tokens"],
            r["outcome_use_tokens"],
        )
        dedup[key] = r

    return list(dedup.values()), len(candidates)


def scan_code_docs_references():
    rows = []

    for root in REFERENCE_ROOTS:
        for path in list_files(root):
            try:
                if (
                    path.suffix.lower() not in TEXT_SUFFIXES
                    or path.stat().st_size > MAX_TEXT_BYTES
                ):
                    continue
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            ids = strong_identity_hits(str(path) + "\n" + text)
            if ids:
                rows.append({
                    "root": str(root),
                    "file": str(path),
                    "identity_tokens": ";".join(ids),
                    "reference_only_not_target_exposure": 1,
                })

    return rows


def preflight():
    upstream = validate_upstream()

    print("===== Q1-R05D0 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r05b2_lock_sha256={EXPECTED_R05B2_LOCK_SHA256}")
    print(f"r05b2_decision={upstream['r05b2_decision']}")
    print(f"frozen_paot_predictor_packages={upstream['predictor_packages']}")
    print("backup_confirmatory_dataset=BKAI-IGH NeoPolyp-Small")
    print("frozen_target_pool=official_1000_train_plus_train_gt")
    print("NeoPolyp_downloaded=NO")
    print("NeoPolyp_image_pixels_opened=NO")
    print("NeoPolyp_GT_pixels_opened=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("predictor_fit=NO")
    print("manual_rescue_audit_if_hit=NO")
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
        "R05D0 protocol copy",
    )

    exposure_rows, scanned_files = scan_historical_exposure()
    exposure_path = build_dir / "historical_neopolyp_exposure_scan.csv"
    write_csv(
        exposure_path,
        exposure_rows,
        [
            "root",
            "file",
            "evidence_type",
            "location",
            "identity_tokens",
            "outcome_use_tokens",
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
            "identity_tokens",
            "reference_only_not_target_exposure",
        ],
    )

    upstream.update({
        "backup_confirmatory_dataset": "BKAI-IGH NeoPolyp-Small",
        "frozen_target_pool": "official_1000_train_plus_train_gt",
        "historical_files_scanned": scanned_files,
        "true_historical_neopolyp_exposure_rows": len(exposure_rows),
        "code_docs_reference_rows": len(reference_rows),
        "neopolyp_downloaded_by_r05d0": False,
        "neopolyp_image_pixels_opened_by_r05d0": False,
        "neopolyp_gt_pixels_opened_by_r05d0": False,
        "target_model_inference_run": False,
        "target_tta_run": False,
        "target_predictor_fit": False,
        "manual_rescue_audit_if_hit": False,
    })

    upstream_path = build_dir / "upstream_r05b2_audit.json"
    write_json(upstream_path, upstream)

    checks = {
        "r05b2_lock_verified": True,
        "paot_predictor_packages_12":
            upstream["predictor_packages"] == EXPECTED_PAOT_PACKAGES,
        "r05b2_target_data_closed":
            upstream["target_data_used"] is False,
        "historical_neopolyp_exposure_zero":
            len(exposure_rows) == 0,
        "neopolyp_not_downloaded_by_r05d0": True,
        "neopolyp_pixels_not_opened_by_r05d0": True,
        "target_model_inference_not_run": True,
        "target_tta_not_run": True,
        "target_predictor_not_fit": True,
    }

    decision = (
        DECISION_LOCKED
        if all(checks.values())
        else DECISION_RETIRED
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05D0 NEOPOLYP BACKUP CONFIRMATORY TARGET LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen source predictor:",
        f"  R05B2 lock={EXPECTED_R05B2_LOCK_SHA256}",
        f"  predictor packages={upstream['predictor_packages']}",
        "",
        "Backup target:",
        "  dataset=BKAI-IGH NeoPolyp-Small",
        "  target pool=official 1000 train + train_gt",
        "  study training use=NO",
        "  validation/tuning use=NO",
        "",
        "Historical exposure:",
        f"  files scanned={scanned_files}",
        f"  TRUE historical NeoPolyp exposure rows={len(exposure_rows)}",
        f"  code/docs reference rows={len(reference_rows)}",
        "  manual rescue audit if hit=NO",
        "",
        "Information boundary:",
        "  NeoPolyp downloaded=NO",
        "  NeoPolyp image pixels opened=NO",
        "  NeoPolyp GT pixels opened=NO",
        "  target inference=NO",
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
            "Next:",
            "  Q1-R05D1 official Kaggle acquisition + 1000 pairing + duplicate QC",
        ]
    else:
        lines += [
            "",
            "NeoPolyp is retired as untouched backup.",
            "Do not enter a manual rescue audit loop.",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "historical_neopolyp_exposure_scan": exposure_path,
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
        "backup_confirmatory_dataset": "BKAI-IGH NeoPolyp-Small",
        "frozen_target_pool": "official_1000_train_plus_train_gt",
        "neopolyp_downloaded": False,
        "neopolyp_image_pixels_opened": False,
        "neopolyp_gt_pixels_opened": False,
        "target_model_inference_run": False,
        "target_tta_run": False,
        "target_predictor_fit": False,
        "historical_neopolyp_exposure_rows": len(exposure_rows),
        "manual_rescue_audit_if_hit": False,
        "decision": decision,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = build_dir / "Q1_R05D0_NEOPOLYP_SELECTION_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R05D0 LOCK:",
        args.output_dir / "Q1_R05D0_NEOPOLYP_SELECTION_LOCK.json",
    )
    print("Q1-R05D0 LOCK SHA256:", lock_sha)


def self_test():
    # Strong identities.
    assert strong_identity_hits(
        "BKAI-IGH-NeoPolyp"
    ) == ["bkai-igh-neopolyp"]
    assert strong_identity_hits(
        "NeoPolyp-Small"
    ) == ["neopolyp-small"]

    # Generic neoplasia words must not trigger dataset identity.
    negatives = [
        "neoplasia",
        "neoplastic",
        "neo",
        "neoplasm segmentation",
        "ETIS-LaribPolypDB",
    ]
    for value in negatives:
        assert strong_identity_hits(value) == [], value

    # Local outcome/use semantics.
    uses = local_outcome_use_hits(
        "BKAI-IGH-NeoPolyp source_dice prediction TTA train_gt"
    )
    assert "dice" in uses
    assert "prediction" in uses
    assert "tta" in uses
    assert "train_gt" in uses

    fake_self = Path(
        r"F:\MEDSEG_SAFETTA\outputs\Q1_R05D0_neopolyp_backup_confirmatory_target_selection_lock_v1\run_log.txt"
    )
    assert is_self_artifact(fake_self)

    print("STRONG_NEOPOLYP_IDENTITY_TEST_PASS")
    print("NEOPLASIA_COLLISION_GUARD_TEST_PASS")
    print("LOCAL_OUTCOME_USE_TEST_PASS")
    print("SELF_ARTIFACT_EXCLUSION_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05D0: one-shot untouched eligibility lock for "
            "BKAI-IGH NeoPolyp-Small before data access."
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
