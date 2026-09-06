#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05A — Confirmatory Target Eligibility Audit.

No target image/mask pixels are opened.
No model inference.
No TTA.
No predictor fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05A-v1"
BUILD = "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05A_confirmatory_target_eligibility_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "ff3332db190ac7337ecba102be7affeb83293c6fecac2cb5ffea299f7e7d9a5c"

S01_MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
EXPECTED_S01_MANIFEST_SHA256 = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)

R03_LOCK = (
    ROOT / "outputs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"
    / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
)
EXPECTED_R03_LOCK_SHA256 = (
    "2c773f5eb70bedb50e2c231ef2512c06dff5ecca425eb43b2719637c1bc06dad"
)

R04_LOCK = (
    ROOT / "outputs"
    / "Q1_R04_model_relative_prospective_utility_feasibility_loao_v1_fix1"
    / "Q1_R04_MODEL_RELATIVE_UTILITY_LOCK.json"
)
EXPECTED_R04_LOCK_SHA256 = (
    "9147a3b3f9f3d7526c057c4c049ac3119ad0f82dfc7b529b32883164ee03cc3a"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "Q1_R05A_confirmatory_target_eligibility_audit_v1"
)

EXPECTED_UNSEEN_COUNTS = {
    "ColonDB": 380,
    "CVC-300": 60,
    "ETIS": 196,
}
EXPECTED_SEEN_COUNTS = {
    "Kvasir": 100,
    "ClinicDB": 62,
}

TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".txt", ".log", ".md",
    ".yaml", ".yml",
}
MAX_TEXT_FILE_BYTES = 256 * 1024 * 1024

OUTCOME_TOKENS = (
    "dice",
    "delta",
    "harm",
    "benefit",
    "prediction",
    "probability",
    "score",
    "entropy",
    "loss",
    "auroc",
    "auprc",
    "action_dice",
    "source_dice",
)

METADATA_NAME_TOKENS = (
    "manifest",
    "inventory",
    "provenance",
    "protocol",
    "readme",
    "audit",
    "lock",
)

POLYPGEN_TOKENS = (
    "polypgen",
    "polyp_gen",
)

DECISION_ELIGIBLE = "CONFIRMATORY_TARGET_UNSEEN_LOCKED_ELIGIBLE"
DECISION_UNRESOLVED = "CONFIRMATORY_TARGET_ELIGIBILITY_UNRESOLVED"


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


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


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


def infer_dataset(row: dict) -> str:
    # Prefer explicit manifest dataset field.
    ds = (row.get("dataset") or "").strip()
    if ds:
        aliases = {
            "CVC_300": "CVC-300",
            "CVC300": "CVC-300",
            "CVC-300": "CVC-300",
            "ColonDB": "ColonDB",
            "ETIS": "ETIS",
            "Kvasir": "Kvasir",
            "ClinicDB": "ClinicDB",
            "CVC-ClinicDB": "ClinicDB",
        }
        if ds in aliases:
            return aliases[ds]

    hay = " ".join([
        row.get("sample_id", ""),
        row.get("image_relpath", ""),
        row.get("mask_relpath", ""),
        ds,
    ]).lower()

    rules = [
        ("cvc-300", "CVC-300"),
        ("cvc_300", "CVC-300"),
        ("cvc300", "CVC-300"),
        ("colondb", "ColonDB"),
        ("colon_db", "ColonDB"),
        ("etis", "ETIS"),
        ("kvasir", "Kvasir"),
        ("clinicdb", "ClinicDB"),
        ("cvc-clinicdb", "ClinicDB"),
    ]
    for token, name in rules:
        if token in hay:
            return name

    return ds or "UNKNOWN"


def load_manifest_roles():
    validate_sha(
        S01_MANIFEST,
        EXPECTED_S01_MANIFEST_SHA256,
        "S01 manifest",
    )
    rows, fields = read_csv(S01_MANIFEST)
    required = {"sample_id", "s01_role", "image_relpath", "mask_relpath"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"S01 manifest missing fields: {missing}")

    unseen = [r for r in rows if r["s01_role"] == "unseen_locked"]
    seen = [r for r in rows if r["s01_role"] == "seen_sanity"]

    unseen_counts = Counter(infer_dataset(r) for r in unseen)
    seen_counts = Counter(infer_dataset(r) for r in seen)

    return rows, unseen, seen, unseen_counts, seen_counts


def validate_upstream_boundaries():
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R04_LOCK, EXPECTED_R04_LOCK_SHA256, "R04 lock")

    r03 = json.loads(R03_LOCK.read_text(encoding="utf-8"))
    r04 = json.loads(R04_LOCK.read_text(encoding="utf-8"))

    if r03.get("decision") != "NINE_STATE_SOURCE_UTILITY_ASSET_READY":
        raise RuntimeError("Unexpected R03 decision.")
    if bool(r03.get("target_data_used", True)):
        raise RuntimeError("R03 target_data_used is not false.")

    if r04.get("decision") != (
        "STOP_MODEL_RELATIVE_UTILITY_NO_ROBUST_LOAO_SIGNAL"
    ):
        raise RuntimeError("Unexpected R04 decision.")
    if bool(r04.get("target_data_used", True)):
        raise RuntimeError("R04 target_data_used is not false.")

    return {
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_decision": r03.get("decision"),
        "r03_target_data_used": r03.get("target_data_used"),
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "r04_decision": r04.get("decision"),
        "r04_target_data_used": r04.get("target_data_used"),
        "candidate_target_pixels_opened": False,
        "model_inference_run": False,
        "tta_run": False,
        "predictor_fit": False,
    }


def list_text_files(root: Path):
    files = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        try:
            if (
                p.is_file()
                and p.suffix.lower() in TEXT_SUFFIXES
                and p.stat().st_size <= MAX_TEXT_FILE_BYTES
            ):
                files.append(p)
        except OSError:
            continue
    return sorted(files)


def build_sample_pattern(sample_ids):
    # Exact sample IDs are escaped; longest-first reduces partial alternatives.
    escaped = sorted(
        (re.escape(s) for s in set(sample_ids)),
        key=len,
        reverse=True,
    )
    if not escaped:
        raise RuntimeError("No sample IDs supplied.")
    return re.compile("|".join(escaped))


def classify_text_artifact(path: Path, text_lower: str) -> str:
    name_lower = path.name.lower()
    outcome = any(tok in text_lower for tok in OUTCOME_TOKENS)
    metadata_like = any(tok in name_lower for tok in METADATA_NAME_TOKENS)

    if outcome and not metadata_like:
        return "outcome_bearing"
    if outcome and metadata_like:
        # Conservative: protocol/audit/lock references are metadata unless
        # the file is clearly a row-level prediction/result table.
        if any(
            tok in name_lower
            for tok in (
                "prediction",
                "result",
                "metric",
                "utility",
                "table",
                "score",
            )
        ):
            return "outcome_bearing"
        return "metadata_only"
    return "metadata_only"


def scan_historical_sample_hits(sample_ids, output_root: Path):
    pattern = build_sample_pattern(sample_ids)
    files = list_text_files(output_root)
    rows = []

    for path in tqdm(
        files,
        desc="Q1-R05A historical output scan",
        unit="file",
        dynamic_ncols=True,
    ):
        # Never inspect the current R05A output itself.
        if "Q1_R05A_confirmatory_target_eligibility_audit" in str(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        matches = sorted(set(m.group(0) for m in pattern.finditer(text)))
        if not matches:
            continue

        lower = text.lower()
        exposure_type = classify_text_artifact(path, lower)

        for sid in matches:
            rows.append({
                "sample_id": sid,
                "file": str(path),
                "exposure_type": exposure_type,
                "file_size": path.stat().st_size,
                "file_sha256": sha256_file(path),
            })

    return rows, len(files)


def scan_code_references(code_root: Path, docs_root: Path):
    rows = []
    tokens = (
        "unseen_locked",
        "ColonDB",
        "CVC-300",
        "ETIS",
        "PolypGen",
    )

    files = list_text_files(code_root) + list_text_files(docs_root)
    for path in tqdm(
        files,
        desc="Q1-R05A code/docs reference scan",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        low = text.lower()
        found = [t for t in tokens if t.lower() in low]
        if found:
            rows.append({
                "file": str(path),
                "tokens": ";".join(found),
                "reference_only_not_exposure": 1,
            })
    return rows


def scan_polypgen_history(outputs_root: Path):
    evidence = []
    s06_s07_dirs = []

    if outputs_root.exists():
        for p in outputs_root.iterdir():
            if not p.is_dir():
                continue
            name_lower = p.name.lower()
            if name_lower.startswith("s06") or name_lower.startswith("s07"):
                s06_s07_dirs.append(str(p))

    for path in list_text_files(outputs_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        low = (path.name + "\n" + text).lower()
        if any(tok in low for tok in POLYPGEN_TOKENS):
            evidence.append({
                "file": str(path),
                "file_sha256": sha256_file(path),
                "contains_outcome_token": int(
                    any(tok in low for tok in OUTCOME_TOKENS)
                ),
            })

    development_exposed = bool(s06_s07_dirs) and bool(evidence)

    return {
        "POLYPGEN_DEVELOPMENT_EXPOSED": development_exposed,
        "s06_s07_output_directories": sorted(s06_s07_dirs),
        "polypgen_text_evidence_count": len(evidence),
        "polypgen_text_evidence": evidence[:200],
        "eligible_as_untouched_confirmatory_target": False,
    }


def run_preflight():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05A protocol",
    )
    _, unseen, seen, unseen_counts, seen_counts = load_manifest_roles()
    upstream = validate_upstream_boundaries()

    print("===== Q1-R05A PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"s01_manifest_sha256={EXPECTED_S01_MANIFEST_SHA256}")
    print(f"unseen_locked={len(unseen)}")
    print(f"unseen_counts={dict(sorted(unseen_counts.items()))}")
    print(f"seen_sanity={len(seen)}")
    print(f"seen_counts={dict(sorted(seen_counts.items()))}")
    print(f"r03_target_data_used={upstream['r03_target_data_used']}")
    print(f"r04_target_data_used={upstream['r04_target_data_used']}")
    print("target_image_pixels_opened=NO")
    print("target_mask_pixels_opened=NO")
    print("model_inference=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05A protocol",
    )
    all_rows, unseen, seen, unseen_counts, seen_counts = load_manifest_roles()
    upstream = validate_upstream_boundaries()

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
        "protocol copy",
    )

    summary_rows = []
    for role, counts, total in (
        ("unseen_locked", unseen_counts, len(unseen)),
        ("seen_sanity", seen_counts, len(seen)),
    ):
        for ds, count in sorted(counts.items()):
            summary_rows.append({
                "s01_role": role,
                "dataset": ds,
                "count": count,
                "role_total": total,
            })

    summary_path = build_dir / "candidate_manifest_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        ["s01_role", "dataset", "count", "role_total"],
    )

    sample_ids = [r["sample_id"] for r in unseen]
    hit_rows, scanned_files = scan_historical_sample_hits(
        sample_ids,
        ROOT / "outputs",
    )

    hits_path = build_dir / "historical_sample_id_hits.csv"
    write_csv(
        hits_path,
        hit_rows,
        [
            "sample_id",
            "file",
            "exposure_type",
            "file_size",
            "file_sha256",
        ],
    )

    code_rows = scan_code_references(
        ROOT / "code",
        ROOT / "docs",
    )
    code_path = build_dir / "code_reference_summary.csv"
    write_csv(
        code_path,
        code_rows,
        ["file", "tokens", "reference_only_not_exposure"],
    )

    polypgen = scan_polypgen_history(ROOT / "outputs")
    polypgen_path = build_dir / "polypgen_historical_exposure.json"
    write_json(polypgen_path, polypgen)

    upstream.update({
        "historical_output_text_files_scanned": scanned_files,
        "unseen_locked_sample_ids": len(sample_ids),
        "historical_hit_rows": len(hit_rows),
        "metadata_only_hit_rows": sum(
            1 for r in hit_rows if r["exposure_type"] == "metadata_only"
        ),
        "outcome_bearing_hit_rows": sum(
            1 for r in hit_rows if r["exposure_type"] == "outcome_bearing"
        ),
    })
    upstream_path = build_dir / "upstream_boundary_audit.json"
    write_json(upstream_path, upstream)

    exact_unseen = dict(unseen_counts) == EXPECTED_UNSEEN_COUNTS
    exact_seen = dict(seen_counts) == EXPECTED_SEEN_COUNTS
    outcome_hits = upstream["outcome_bearing_hit_rows"]

    checks = {
        "manifest_sha_match": True,
        "unseen_total_636": len(unseen) == 636,
        "unseen_exact_dataset_counts": exact_unseen,
        "seen_total_162": len(seen) == 162,
        "seen_exact_dataset_counts": exact_seen,
        "r03_target_closed": upstream["r03_target_data_used"] is False,
        "r04_target_closed": upstream["r04_target_data_used"] is False,
        "target_pixels_opened_by_r05a": False,
        "unseen_outcome_bearing_hits_zero": outcome_hits == 0,
    }

    decision = (
        DECISION_ELIGIBLE
        if all(checks.values())
        else DECISION_UNRESOLVED
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05A CONFIRMATORY TARGET ELIGIBILITY AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Candidate:",
        f"  unseen_locked total={len(unseen)}",
        f"  dataset_counts={dict(sorted(unseen_counts.items()))}",
        "",
        "Historical exposure scan:",
        f"  output text files scanned={scanned_files}",
        f"  metadata-only hit rows={upstream['metadata_only_hit_rows']}",
        f"  outcome-bearing hit rows={outcome_hits}",
        "",
        "PolypGen:",
        "  development_exposed="
        + ("YES" if polypgen["POLYPGEN_DEVELOPMENT_EXPOSED"] else "NO"),
        "  eligible_as_untouched_confirmatory_target=NO",
        "",
        "Information boundary:",
        "  target image pixels opened=NO",
        "  target mask pixels opened=NO",
        "  model inference=NO",
        "  TTA=NO",
        "  predictor fit=NO",
        "",
        "Checks:",
    ]
    for k, v in checks.items():
        lines.append(f"  {k}={'PASS' if v else 'FAIL'}")
    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]
    if decision == DECISION_ELIGIBLE:
        lines += [
            "",
            "If eligible:",
            "  next = Q1-R05B Frozen PAOT on ColonDB/CVC-300/ETIS",
        ]
    else:
        lines += [
            "",
            "If unresolved:",
            "  inspect historical_sample_id_hits.csv before any target use",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = {
        "protocol_copy": protocol_copy,
        "candidate_manifest_summary": summary_path,
        "historical_sample_id_hits": hits_path,
        "code_reference_summary": code_path,
        "polypgen_historical_exposure": polypgen_path,
        "upstream_boundary_audit": upstream_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "s01_manifest_sha256": EXPECTED_S01_MANIFEST_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "candidate_role": "unseen_locked",
        "candidate_total": len(unseen),
        "candidate_dataset_counts": dict(sorted(unseen_counts.items())),
        "target_pixels_opened": False,
        "model_inference_run": False,
        "predictor_fit": False,
        "outcome_bearing_hit_rows": outcome_hits,
        "polypgen_development_exposed": polypgen[
            "POLYPGEN_DEVELOPMENT_EXPOSED"
        ],
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
        build_dir / "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_LOCK.json"
    )
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        p = build_dir / meta["relative_path"]
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R05A LOCK:",
        args.output_dir / "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_LOCK.json",
    )
    print("Q1-R05A LOCK SHA256:", lock_sha)


def self_test():
    assert sum(EXPECTED_UNSEEN_COUNTS.values()) == 636
    assert sum(EXPECTED_SEEN_COUNTS.values()) == 162
    assert DECISION_ELIGIBLE != DECISION_UNRESOLVED

    toy = [
        {"dataset": "ColonDB", "sample_id": "a", "image_relpath": "", "mask_relpath": ""},
        {"dataset": "CVC_300", "sample_id": "b", "image_relpath": "", "mask_relpath": ""},
        {"dataset": "", "sample_id": "ETIS::c", "image_relpath": "", "mask_relpath": ""},
    ]
    assert [infer_dataset(r) for r in toy] == [
        "ColonDB", "CVC-300", "ETIS"
    ]

    pat = build_sample_pattern(["abc::1", "abc::2"])
    s = "x abc::2 y"
    assert [m.group(0) for m in pat.finditer(s)] == ["abc::2"]

    print("EXPECTED_COUNTS_TEST_PASS")
    print("DATASET_INFERENCE_TEST_PASS")
    print("SAMPLE_PATTERN_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05A: audit whether unseen_locked ColonDB/CVC-300/ETIS "
            "remain eligible for confirmatory PAOT validation without opening "
            "target image/mask pixels."
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
        run_preflight()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
