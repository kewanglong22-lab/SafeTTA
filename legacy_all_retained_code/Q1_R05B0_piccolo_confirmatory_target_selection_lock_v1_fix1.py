#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05B0 — PICCOLO confirmatory target selection lock.

No PICCOLO download.
No image/mask pixel access.
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


VERSION = "2026-08-19-Q1-R05B0-v1-fix1"
BUILD = "Q1_R05B0_PICCOLO_SELECTION_LOCK_DOI_CONTEXT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05B0_piccolo_confirmatory_target_selection_lock_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = "5cf67e2e15ccd5f887fdf350b73174ebfea1cd88acc25ff1d61683696950c8fd"

R05A_FIX1_DIR = (
    ROOT / "outputs"
    / "Q1_R05A_confirmatory_target_eligibility_audit_v1_fix1"
)
R05A_FIX1_LOCK = (
    R05A_FIX1_DIR
    / "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_LOCK.json"
)
EXPECTED_R05A_FIX1_LOCK_SHA256 = (
    "ed4ddccf06d128ecc5065d4933f1e22e73b3fe18183b86439da7d9fe930e68a6"
)
EXPECTED_R05A_DECISION = "CONFIRMATORY_TARGET_ELIGIBILITY_UNRESOLVED"
EXPECTED_OLD_EXPOSED_SAMPLES = 636

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05B0_piccolo_confirmatory_target_selection_lock_v1_fix1"
)

DIRECT_SEARCH_TOKENS = (
    "piccolo",
    "pd178",
)

DOI_SUFFIXES = (
    "4279017",
    "4279016",
)

DOI_CONTEXT_TOKENS = (
    "zenodo",
    "10.5281",
)

TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".txt",
    ".log", ".md", ".yaml", ".yml",
}
MAX_TEXT_BYTES = 128 * 1024 * 1024

DISQUALIFY_ROOTS = (
    ROOT / "data",
    ROOT / "outputs",
    ROOT / "cache",
)

REFERENCE_ROOTS = (
    ROOT / "code",
    ROOT / "docs",
)

SELF_TOKENS = (
    "Q1_R05B0_piccolo_confirmatory_target_selection_lock",
)

DECISION_LOCKED = "PICCOLO_CONFIRMATORY_TARGET_SELECTION_LOCKED"
DECISION_UNRESOLVED = "PICCOLO_CONFIRMATORY_TARGET_SELECTION_UNRESOLVED"


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
    s = str(path)
    return any(token in s for token in SELF_TOKENS)


def token_hits(text: str):
    """
    FIX1 historical-evidence matcher.

    Direct dataset identifiers count anywhere:
      - piccolo
      - pd178

    Bare Zenodo record-number suffixes do NOT count unless the same local
    context contains a Zenodo/DOI marker. This prevents accidental matches in
    floating-point numerical result tables.
    """
    low = text.lower()
    hits = set()

    for token in DIRECT_SEARCH_TOKENS:
        if token in low:
            hits.add(token)

    for suffix in DOI_SUFFIXES:
        for match in re.finditer(re.escape(suffix), low):
            start = max(0, match.start() - 80)
            end = min(len(low), match.end() + 80)
            context = low[start:end]
            if any(marker in context for marker in DOI_CONTEXT_TOKENS):
                hits.add(suffix)
                break

    return sorted(hits)


def scan_asset_roots():
    rows = []

    candidates = []
    for root in DISQUALIFY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if is_self_artifact(path):
                    continue
                candidates.append((root, path))
            except OSError:
                continue

    for root, path in tqdm(
        candidates,
        desc="Q1-R05B0 historical PICCOLO asset scan",
        unit="file",
        dynamic_ncols=True,
    ):
        path_hits = token_hits(str(path))

        content_hits = []
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            and path.stat().st_size <= MAX_TEXT_BYTES
        ):
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                content_hits = token_hits(text)
            except Exception:
                content_hits = []

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
            "disqualifying_historical_asset": 1,
        })

    return rows, len(candidates)


def scan_reference_roots():
    rows = []
    for root in REFERENCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if is_self_artifact(path):
                    continue
                if (
                    path.suffix.lower() not in TEXT_SUFFIXES
                    or path.stat().st_size > MAX_TEXT_BYTES
                ):
                    continue

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                hits = token_hits(str(path) + "\n" + text)
                if hits:
                    rows.append({
                        "root": str(root),
                        "file": str(path),
                        "tokens": ";".join(hits),
                        "reference_only_not_target_exposure": 1,
                    })
            except Exception:
                continue
    return rows


def validate_upstream():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "Q1-R05B0 protocol",
    )
    validate_sha(
        R05A_FIX1_LOCK,
        EXPECTED_R05A_FIX1_LOCK_SHA256,
        "Q1-R05A FIX1 lock",
    )

    lock = json.loads(
        R05A_FIX1_LOCK.read_text(encoding="utf-8")
    )

    if lock.get("decision") != EXPECTED_R05A_DECISION:
        raise RuntimeError(
            f"Unexpected R05A decision: {lock.get('decision')}"
        )

    exposed = int(
        lock.get("outcome_bearing_unique_candidate_samples", -1)
    )
    if exposed != EXPECTED_OLD_EXPOSED_SAMPLES:
        raise RuntimeError(
            f"Expected old exposed candidate count 636, got {exposed}"
        )

    if bool(lock.get("target_image_pixels_opened", True)):
        raise RuntimeError("R05A reports target image pixel access.")
    if bool(lock.get("target_mask_pixels_opened", True)):
        raise RuntimeError("R05A reports target mask pixel access.")

    return {
        "r05a_fix1_lock_sha256": EXPECTED_R05A_FIX1_LOCK_SHA256,
        "r05a_decision": lock.get("decision"),
        "old_candidate_outcome_exposed_unique_samples": exposed,
        "r05a_target_image_pixels_opened":
            lock.get("target_image_pixels_opened"),
        "r05a_target_mask_pixels_opened":
            lock.get("target_mask_pixels_opened"),
    }


def preflight():
    upstream = validate_upstream()

    print("===== Q1-R05B0 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(
        "r05a_fix1_lock_sha256="
        f"{EXPECTED_R05A_FIX1_LOCK_SHA256}"
    )
    print(
        "old_candidate_outcome_exposed_unique_samples="
        f"{upstream['old_candidate_outcome_exposed_unique_samples']}"
    )
    print("new_confirmatory_dataset=PICCOLO")
    print("frozen_subset=official_test_only")
    print("expected_test_images=333")
    print("PICCOLO_download=NO")
    print("PICCOLO_image_pixels_opened=NO")
    print("PICCOLO_mask_pixels_opened=NO")
    print("model_inference=NO")
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
        "Q1-R05B0 protocol copy",
    )

    asset_rows, scanned_files = scan_asset_roots()
    asset_path = build_dir / "historical_piccolo_asset_scan.csv"
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
            "disqualifying_historical_asset",
        ],
    )

    reference_rows = scan_reference_roots()
    reference_path = build_dir / "literature_reference_scan.csv"
    write_csv(
        reference_path,
        reference_rows,
        [
            "root",
            "file",
            "tokens",
            "reference_only_not_target_exposure",
        ],
    )

    upstream.update({
        "historical_data_output_cache_files_scanned": scanned_files,
        "historical_piccolo_disqualifying_assets": len(asset_rows),
        "code_docs_reference_rows": len(reference_rows),
        "piccolo_downloaded_by_r05b0": False,
        "piccolo_image_pixels_opened_by_r05b0": False,
        "piccolo_mask_pixels_opened_by_r05b0": False,
        "model_inference_run": False,
        "predictor_fit": False,
        "frozen_dataset": "PICCOLO",
        "frozen_subset": "official_test_only",
        "expected_test_images": 333,
    })

    upstream_path = build_dir / "upstream_r05a_audit.json"
    write_json(upstream_path, upstream)

    checks = {
        "r05a_fix1_lock_verified": True,
        "old_636_candidates_retired_as_untouched": (
            upstream["old_candidate_outcome_exposed_unique_samples"] == 636
        ),
        "historical_piccolo_assets_zero": len(asset_rows) == 0,
        "piccolo_not_downloaded_by_r05b0": True,
        "piccolo_pixels_not_opened_by_r05b0": True,
        "model_inference_not_run": True,
        "predictor_not_fit": True,
    }

    decision = (
        DECISION_LOCKED if all(checks.values())
        else DECISION_UNRESOLVED
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05B0 PICCOLO CONFIRMATORY TARGET SELECTION LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Retired prior candidate:",
        "  ColonDB/CVC-300/ETIS outcome-exposed unique samples=636",
        "",
        "New confirmatory target:",
        "  dataset=PICCOLO",
        "  subset=official test only",
        "  expected test images=333",
        "  PICCOLO train use=NO",
        "  PICCOLO validation use=NO",
        "",
        "Historical local exposure:",
        f"  data/outputs/cache files scanned={scanned_files}",
        f"  disqualifying PICCOLO assets={len(asset_rows)}",
        f"  code/docs reference rows={len(reference_rows)}",
        "",
        "Information boundary:",
        "  PICCOLO downloaded=NO",
        "  PICCOLO image pixels opened=NO",
        "  PICCOLO mask pixels opened=NO",
        "  model inference=NO",
        "  predictor fit=NO",
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
            "  next = Q1-R05B1 PICCOLO acquisition and structure/QC audit",
        ]
    else:
        lines += [
            "",
            "If unresolved:",
            "  inspect historical_piccolo_asset_scan.csv",
            "  do not acquire PICCOLO yet",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "historical_piccolo_asset_scan": asset_path,
        "literature_reference_scan": reference_path,
        "upstream_r05a_audit": upstream_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r05a_fix1_lock_sha256": EXPECTED_R05A_FIX1_LOCK_SHA256,
        "frozen_confirmatory_dataset": "PICCOLO",
        "frozen_confirmatory_subset": "official_test_only",
        "expected_test_images": 333,
        "piccolo_train_allowed": False,
        "piccolo_validation_allowed": False,
        "piccolo_downloaded": False,
        "piccolo_image_pixels_opened": False,
        "piccolo_mask_pixels_opened": False,
        "model_inference_run": False,
        "predictor_fit": False,
        "historical_piccolo_disqualifying_assets": len(asset_rows),
        "decision": decision,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = (
        build_dir / "Q1_R05B0_PICCOLO_SELECTION_LOCK.json"
    )
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
        "Q1-R05B0 LOCK:",
        args.output_dir / "Q1_R05B0_PICCOLO_SELECTION_LOCK.json",
    )
    print("Q1-R05B0 LOCK SHA256:", lock_sha)


def self_test():
    assert EXPECTED_OLD_EXPOSED_SAMPLES == 636
    assert "piccolo" in DIRECT_SEARCH_TOKENS
    assert "pd178" in DIRECT_SEARCH_TOKENS
    assert "4279017" in DOI_SUFFIXES

    assert token_hits("PICCOLO PD178") == ["pd178", "piccolo"]
    assert token_hits("nothing relevant") == []

    # Bare numeric DOI suffixes inside numerical result content are NOT
    # evidence of PICCOLO.
    assert token_hits("0.427901712345") == []
    assert token_hits("metric,0.1234279016789") == []

    # DOI/Zenodo contextual forms are evidence.
    assert token_hits("10.5281/zenodo.4279017") == ["4279017"]
    assert token_hits("Zenodo record 4279016") == ["4279016"]

    fake_self = Path(
        r"F:\MEDSEG_SAFETTA\docs\Q1_R05B0_piccolo_confirmatory_target_selection_lock_preregistered_protocol_v1_fix1.md"
    )
    assert is_self_artifact(fake_self)

    print("FROZEN_SELECTION_TEST_PASS")
    print("DIRECT_TOKEN_SCAN_TEST_PASS")
    print("BARE_DOI_SUFFIX_FALSE_POSITIVE_TEST_PASS")
    print("CONTEXTUAL_DOI_EVIDENCE_TEST_PASS")
    print("SELF_ARTIFACT_EXCLUSION_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05B0: freeze PICCOLO official test split as a new "
            "confirmatory PAOT target before dataset access."
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
