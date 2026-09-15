#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P09B — public release bundle

Creates a compact reproducibility bundle from the frozen P09A evidence state.

NO scientific metrics.
NO fitting/inference.
NO GitHub/network writes.
NO modification of frozen evidence.

Outputs:
- curated code/protocol snapshot
- compact aggregate evidence snapshot
- RELEASE_INDEX.md
- SHA256 manifest
- zip bundle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any


VERSION = "2026-09-15-B6-P09B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P09B_PUBLIC_RELEASE_BUNDLE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "3e2c64d1c34f6bfd45619479b43ae3b6377a0265eaf7aecfdd63362d2af6ad82"

P09A_DIR = ROOT / "B6_P09A_final_evidence_freeze_v1"
P09A_FREEZE = P09A_DIR / "B6_P09A_FINAL_EVIDENCE_FREEZE.json"
P09A_STAGE_REGISTRY = P09A_DIR / "B6_P09A_STAGE_REGISTRY.csv"
P09A_EVIDENCE_MANIFEST = P09A_DIR / "B6_P09A_EVIDENCE_SHA256_MANIFEST.csv"
P09A_CLAIM_MATRIX = P09A_DIR / "B6_P09A_CLAIM_MATRIX.csv"

EXPECTED_P09A_GATE = "PASS_B6_P09A_FINAL_EVIDENCE_FREEZE"
EXPECTED_RESOLVED_STAGES = 10
EXPECTED_EVIDENCE_FILES = 43
EXPECTED_SUPPORTED_CLAIMS = 6
EXPECTED_FORBIDDEN_CLAIMS = 8

OUT_DIR = ROOT / "B6_P09B_public_release_bundle_v1"
ZIP_BASE = ROOT / "B6_P09B_public_release_bundle_v1"

PASS_GATE = "PASS_B6_P09B_PUBLIC_RELEASE_BUNDLE"

# Code/protocol discovery is intentionally broad within the B6 line so that
# implementation fixes remain traceable. This is code only, not result data.
CODE_GLOBS = [
    "B6_P0*.json",
    "B6_FINAL*.json",
    "Q1_B6_P0*.py",
    "Q1_B6_Final*.py",
]

# Compact aggregate artifacts allowed from frozen stage directories.
ALLOW_AGGREGATE_TOKENS = (
    "REPORT",
    "AUDIT",
    "SUMMARY",
    "POINT_METRICS",
    "POINT_BY_ACTION",
    "EVENT_SUPPORT",
    "CLAIM",
    "STAGE_REGISTRY",
    "EVIDENCE_SHA256_MANIFEST",
    "FINAL_EVIDENCE_FREEZE",
)

# Hard exclusions even if an allowed token happens to appear elsewhere.
EXCLUDE_TOKENS = (
    "REPLICATES",
    "BOUND_PATIENT_UTILITY_ROWS",
    "PACKBITS",
    "MASKS",
    "FEATURE_TABLE",
    "FEATURES",
    "PREDICTIONS",
    "MODELCASE",
    "OUTCOMES",
    "CHECKPOINT",
    ".PTH",
    ".PT",
    ".NPZ",
    ".NPY",
)

ALLOWED_EXT = {".py", ".json", ".txt", ".csv", ".md"}


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P09B protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    d = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if d.get("status") != "FROZEN_AFTER_P09A_BEFORE_GITHUB_SYNC":
        raise RuntimeError(f"P09B protocol status changed: {d.get('status')}")
    return {"path": str(PROTOCOL), "sha256": got, "status": d["status"]}


def verify_p09a() -> Dict[str, Any]:
    for p in [
        P09A_FREEZE,
        P09A_STAGE_REGISTRY,
        P09A_EVIDENCE_MANIFEST,
        P09A_CLAIM_MATRIX,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    freeze = json.loads(P09A_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("gate") != EXPECTED_P09A_GATE:
        raise RuntimeError(f"P09A gate changed: {freeze.get('gate')}")

    stages = list(csv.DictReader(P09A_STAGE_REGISTRY.open("r", encoding="utf-8-sig")))
    resolved = sum(str(r.get("resolved", "")).lower() == "true" for r in stages)
    if resolved != EXPECTED_RESOLVED_STAGES:
        raise RuntimeError(f"P09A resolved stages={resolved}, expected={EXPECTED_RESOLVED_STAGES}")

    evidence = list(csv.DictReader(P09A_EVIDENCE_MANIFEST.open("r", encoding="utf-8-sig")))
    if len(evidence) != EXPECTED_EVIDENCE_FILES:
        raise RuntimeError(
            f"P09A evidence manifest rows={len(evidence)}, expected={EXPECTED_EVIDENCE_FILES}"
        )

    claims = list(csv.DictReader(P09A_CLAIM_MATRIX.open("r", encoding="utf-8-sig")))
    supported = sum(r.get("status") == "SUPPORTED" for r in claims)
    forbidden = sum(r.get("status") == "FORBIDDEN" for r in claims)
    if supported != EXPECTED_SUPPORTED_CLAIMS or forbidden != EXPECTED_FORBIDDEN_CLAIMS:
        raise RuntimeError(
            f"P09A claim counts supported/forbidden={supported}/{forbidden}, "
            f"expected={EXPECTED_SUPPORTED_CLAIMS}/{EXPECTED_FORBIDDEN_CLAIMS}"
        )

    return {
        "freeze_path": str(P09A_FREEZE),
        "freeze_sha256": sha256_file(P09A_FREEZE),
        "resolved_stages": resolved,
        "evidence_files": len(evidence),
        "supported_claims": supported,
        "forbidden_claims": forbidden,
        "stage_rows": stages,
        "evidence_rows": evidence,
    }


def discover_code_files() -> List[Path]:
    files = {}
    for pattern in CODE_GLOBS:
        for p in CODE.glob(pattern):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                files[str(p.resolve()).lower()] = p
    # Ensure the final P09 protocols/scripts are included.
    for p in [
        CODE / "B6_P09_FINAL_EVIDENCE_FREEZE_PROTOCOL_v1.json",
        CODE / "Q1_B6_P09A_final_evidence_freeze_v1.py",
        CODE / "B6_P09B_PUBLIC_RELEASE_BUNDLE_PROTOCOL_v1.json",
        CODE / "Q1_B6_P09B_public_release_bundle_v1.py",
    ]:
        if p.is_file():
            files[str(p.resolve()).lower()] = p
    return sorted(files.values())


def aggregate_allowed(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in ALLOWED_EXT:
        return False
    name = path.name.upper()
    if any(tok in name for tok in EXCLUDE_TOKENS):
        return False
    return any(tok in name for tok in ALLOW_AGGREGATE_TOKENS)


def discover_aggregate_evidence(p09: Dict[str, Any]) -> List[Path]:
    files = {}
    for row in p09["evidence_rows"]:
        p = Path(row["path"])
        if not aggregate_allowed(p):
            continue
        # Verify frozen manifest SHA before inclusion.
        if not p.is_file():
            raise FileNotFoundError(p)
        observed = sha256_file(p)
        if observed != row["sha256"]:
            raise RuntimeError(
                f"Frozen evidence SHA drift: {p}\n"
                f"manifest={row['sha256']} observed={observed}"
            )
        files[str(p.resolve()).lower()] = p

    # Always include the four P09A final manifests.
    for p in [
        P09A_FREEZE,
        P09A_STAGE_REGISTRY,
        P09A_EVIDENCE_MANIFEST,
        P09A_CLAIM_MATRIX,
    ]:
        files[str(p.resolve()).lower()] = p

    return sorted(files.values())


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_index(
    protocol_info: Dict[str, Any],
    p09: Dict[str, Any],
    code_files: List[Path],
    evidence_files: List[Path],
) -> str:
    lines = [
        "# SafeTTA Final Public-Release Bundle",
        "",
        f"- Bundle version: `{VERSION}`",
        f"- P09B protocol SHA256: `{protocol_info['sha256']}`",
        f"- P09A freeze SHA256: `{p09['freeze_sha256']}`",
        f"- Frozen evidence stages: **{p09['resolved_stages']}**",
        f"- P09A paper-facing hashed evidence files: **{p09['evidence_files']}**",
        f"- Supported claims: **{p09['supported_claims']}**",
        f"- Forbidden/overclaim statements: **{p09['forbidden_claims']}**",
        "",
        "## Scientific status",
        "",
        "P01–P08 scientific/statistical evidence is frozen. This bundle is a "
        "reproducibility and release snapshot; it introduces no new performance results.",
        "",
        "## Contents",
        "",
        f"- `code/`: {len(code_files)} B6 protocol/execution/audit scripts",
        f"- `evidence/`: {len(evidence_files)} compact aggregate evidence/manifests",
        "- `SHA256SUMS.csv`: file integrity manifest",
        "",
        "## Claim discipline",
        "",
        "The authoritative paper-facing claim matrix is:",
        "`evidence/P09A/B6_P09A_CLAIM_MATRIX.csv`.",
        "",
        "Do not reinterpret this bundle as pristine prospective validation, "
        "universal semantic-transition superiority, or universal Dice improvement.",
        "",
        "## Next release step",
        "",
        "Synchronize this bundle to a fresh GitHub branch, audit the diff, then create "
        "a final paper tag only after the synchronization audit passes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SafeTTA P09B public release bundle.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        assert "REPLICATES" in EXCLUDE_TOKENS
        assert "PREDICTIONS" in EXCLUDE_TOKENS
        assert "AUDIT" in ALLOW_AGGREGATE_TOKENS
        assert "REPORT" in ALLOW_AGGREGATE_TOKENS
        print("SELF_TEST_PUBLIC_FILTERS=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 172)
    print("SafeTTA B6-P09B — Public release bundle")
    print("Version          :", VERSION)
    print("Scientific metric: NO")
    print("GitHub/network   : NO")
    print("Raw predictions  : EXCLUDED")
    print("Masks/checkpoints: EXCLUDED")
    print("=" * 172)

    protocol_info = verify_protocol()
    p09 = verify_p09a()
    print("P09A_GATE=PASS")
    print("resolved_stages =", p09["resolved_stages"])
    print("evidence_manifest_rows =", p09["evidence_files"])
    print("claim_counts =", p09["supported_claims"], p09["forbidden_claims"])

    code_files = discover_code_files()
    evidence_files = discover_aggregate_evidence(p09)

    print("code_files =", len(code_files))
    print("compact_evidence_files =", len(evidence_files))

    if OUT_DIR.exists() or (ZIP_BASE.with_suffix(".zip")).exists():
        raise FileExistsError(
            f"Never overwrite public bundle: {OUT_DIR} / {ZIP_BASE.with_suffix('.zip')}\n"
            "Use _fix1 only for implementation repair."
        )

    (OUT_DIR / "code").mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "evidence").mkdir(parents=True, exist_ok=True)

    manifest = []

    for p in code_files:
        rel = Path("code") / p.name
        dst = OUT_DIR / rel
        safe_copy(p, dst)
        manifest.append({
            "category": "code",
            "source_path": str(p),
            "bundle_path": str(rel).replace("\\", "/"),
            "bytes": int(dst.stat().st_size),
            "sha256": sha256_file(dst),
        })

    for p in evidence_files:
        # Keep stage identity where possible to avoid filename collisions.
        stage_name = p.parent.name
        rel = Path("evidence") / stage_name / p.name
        dst = OUT_DIR / rel
        safe_copy(p, dst)
        manifest.append({
            "category": "evidence",
            "source_path": str(p),
            "bundle_path": str(rel).replace("\\", "/"),
            "bytes": int(dst.stat().st_size),
            "sha256": sha256_file(dst),
        })

    index = build_index(protocol_info, p09, code_files, evidence_files)
    index_path = OUT_DIR / "RELEASE_INDEX.md"
    index_path.write_text(index, encoding="utf-8")
    manifest.append({
        "category": "release",
        "source_path": "<generated>",
        "bundle_path": "RELEASE_INDEX.md",
        "bytes": int(index_path.stat().st_size),
        "sha256": sha256_file(index_path),
    })

    manifest_path = OUT_DIR / "SHA256SUMS.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "source_path", "bundle_path", "bytes", "sha256"],
        )
        w.writeheader()
        w.writerows(manifest)

    # Bundle-level audit
    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol_info,
        "p09a": {
            k: v for k, v in p09.items()
            if k not in {"stage_rows", "evidence_rows"}
        },
        "code_files": len(code_files),
        "compact_evidence_files": len(evidence_files),
        "bundle_manifest_rows_before_audit_json": len(manifest),
        "raw_prediction_tables_included": False,
        "mask_files_included": False,
        "checkpoint_files_included": False,
        "new_scientific_metrics": False,
        "github_network_write": False,
    }
    audit_path = OUT_DIR / "B6_P09B_PUBLIC_RELEASE_BUNDLE_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Append audit itself to manifest.
    with manifest_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "source_path", "bundle_path", "bytes", "sha256"],
        )
        w.writerow({
            "category": "release",
            "source_path": "<generated>",
            "bundle_path": audit_path.name,
            "bytes": int(audit_path.stat().st_size),
            "sha256": sha256_file(audit_path),
        })

    zip_path = Path(shutil.make_archive(str(ZIP_BASE), "zip", root_dir=OUT_DIR))

    print("bundle_dir =", OUT_DIR)
    print("zip =", zip_path)
    print("zip_bytes =", zip_path.stat().st_size)
    print("zip_sha256 =", sha256_file(zip_path))
    print("raw_predictions_included = NO")
    print("masks_checkpoints_included = NO")
    print("GITHUB_WRITE = NO")
    print("NEXT = exact GitHub synchronization + diff audit")
    print("GATE=" + PASS_GATE)
    print("=" * 172)


if __name__ == "__main__":
    main()
