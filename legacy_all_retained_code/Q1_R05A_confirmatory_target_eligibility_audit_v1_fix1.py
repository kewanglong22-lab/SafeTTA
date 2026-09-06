#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05A FIX1 — structured confirmatory target eligibility re-audit.

This script NEVER opens candidate target image or mask files.
It only reads manifest / text / structured metadata artifacts.
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
from typing import Iterable, Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05A-v1-fix1"
BUILD = "Q1_R05A_STRUCTURED_HISTORICAL_EXPOSURE_REAUDIT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05A_confirmatory_target_eligibility_audit_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = "2a6868c73b59f31f087066c7b4ab2537f02b4f53a9f3436e00e026628a869d91"

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
    ROOT / "outputs"
    / "Q1_R05A_confirmatory_target_eligibility_audit_v1_fix1"
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
    "logit",
)

METADATA_FILENAME_TOKENS = (
    "manifest",
    "inventory",
    "provenance",
    "protocol",
    "readme",
    "audit",
    "lock",
)

TEXT_SUFFIXES = {
    ".csv", ".tsv", ".json", ".jsonl", ".txt",
    ".log", ".md", ".yaml", ".yml",
}

MAX_FILE_BYTES = 256 * 1024 * 1024

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


def read_csv(path: Path, delimiter=","):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader), reader.fieldnames or []


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


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def is_outcome_name(name: str) -> bool:
    norm = normalize_name(name)
    return any(token in norm for token in OUTCOME_TOKENS)


def is_metadata_filename(path: Path) -> bool:
    low = path.name.lower()
    return any(token in low for token in METADATA_FILENAME_TOKENS)


def exact_candidate_ids_in_text(text: str, candidate_ids: set[str]) -> set[str]:
    # This helper is only used for short/local contexts, not whole large files.
    found = set()
    for sid in candidate_ids:
        if sid in text:
            found.add(sid)
    return found


def infer_dataset(row: dict) -> str:
    ds = (row.get("dataset") or "").strip()

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
    for token, label in rules:
        if token in hay:
            return label

    return ds or "UNKNOWN"


def load_manifest_roles():
    validate_sha(
        S01_MANIFEST,
        EXPECTED_S01_MANIFEST_SHA256,
        "S01 manifest",
    )
    rows, fields = read_csv(S01_MANIFEST)
    required = {
        "sample_id",
        "s01_role",
        "image_relpath",
        "mask_relpath",
    }
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
        raise RuntimeError("R03 target boundary violated.")

    if r04.get("decision") != (
        "STOP_MODEL_RELATIVE_UTILITY_NO_ROBUST_LOAO_SIGNAL"
    ):
        raise RuntimeError("Unexpected R04 decision.")
    if bool(r04.get("target_data_used", True)):
        raise RuntimeError("R04 target boundary violated.")

    return {
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_decision": r03.get("decision"),
        "r03_target_data_used": r03.get("target_data_used"),
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "r04_decision": r04.get("decision"),
        "r04_target_data_used": r04.get("target_data_used"),
        "candidate_target_image_pixels_opened": False,
        "candidate_target_mask_pixels_opened": False,
        "model_inference_run": False,
        "tta_run": False,
        "predictor_fit": False,
    }


def list_historical_text_files(root: Path):
    files = []
    if not root.exists():
        return files

    for p in root.rglob("*"):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            # Exclude ALL R05A audit generations, including superseded v1.
            if "Q1_R05A_confirmatory_target_eligibility_audit" in str(p):
                continue
            files.append(p)
        except OSError:
            continue

    return sorted(files)


def build_candidate_index(candidate_rows):
    candidate_ids = {r["sample_id"] for r in candidate_rows}
    if len(candidate_ids) != len(candidate_rows):
        raise RuntimeError("Duplicate unseen_locked sample IDs.")
    return candidate_ids


def candidate_ids_from_row_values(
    row: dict,
    candidate_ids: set[str],
) -> set[str]:
    found = set()
    for value in row.values():
        if value is None:
            continue
        s = str(value)
        if s in candidate_ids:
            found.add(s)
    return found


def scan_csv_tsv(
    path: Path,
    candidate_ids: set[str],
):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    rows = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = reader.fieldnames or []
            outcome_fields = [x for x in fields if is_outcome_name(x)]
            row_number = 1

            for record in reader:
                row_number += 1
                ids = candidate_ids_from_row_values(record, candidate_ids)
                if not ids:
                    continue

                exposure = (
                    "outcome_bearing"
                    if outcome_fields
                    else "metadata_only"
                )
                for sid in ids:
                    rows.append({
                        "sample_id": sid,
                        "file": str(path),
                        "artifact_type": path.suffix.lower(),
                        "location": f"row:{row_number}",
                        "exposure_type": exposure,
                        "evidence": (
                            "outcome_columns="
                            + ";".join(outcome_fields)
                            if outcome_fields
                            else "no_outcome_columns"
                        ),
                    })
    except Exception:
        return []

    return rows


def direct_string_values(obj: dict):
    for key, value in obj.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            yield key, value


def scan_json_object_records(
    obj,
    path: Path,
    candidate_ids: set[str],
    location: str = "$",
):
    rows = []

    if isinstance(obj, dict):
        # Direct-record rule only: candidate ID and outcome key must be direct
        # members of the SAME dict. This avoids file/global branch leakage.
        ids = set()
        for key, value in direct_string_values(obj):
            if isinstance(value, str) and value in candidate_ids:
                ids.add(value)

        direct_outcome_keys = [
            str(k) for k in obj.keys() if is_outcome_name(str(k))
        ]

        if ids:
            exposure = (
                "outcome_bearing"
                if direct_outcome_keys
                else "metadata_only"
            )
            for sid in ids:
                rows.append({
                    "sample_id": sid,
                    "file": str(path),
                    "artifact_type": path.suffix.lower(),
                    "location": location,
                    "exposure_type": exposure,
                    "evidence": (
                        "direct_outcome_keys="
                        + ";".join(direct_outcome_keys)
                        if direct_outcome_keys
                        else "no_direct_outcome_keys"
                    ),
                })

        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                rows.extend(
                    scan_json_object_records(
                        value,
                        path,
                        candidate_ids,
                        f"{location}.{key}",
                    )
                )

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                rows.extend(
                    scan_json_object_records(
                        value,
                        path,
                        candidate_ids,
                        f"{location}[{i}]",
                    )
                )

    return rows


def scan_json_jsonl(
    path: Path,
    candidate_ids: set[str],
):
    try:
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    recs = scan_json_object_records(
                        obj,
                        path,
                        candidate_ids,
                        location=f"$line[{line_no}]",
                    )
                    rows.extend(recs)
            return rows

        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return scan_json_object_records(
            obj,
            path,
            candidate_ids,
        )
    except Exception:
        return []


def local_window_has_outcome(lines, idx: int) -> tuple[bool, str]:
    start = max(0, idx - 1)
    end = min(len(lines), idx + 2)
    window = "\n".join(lines[start:end])
    low = window.lower()
    tokens = sorted({
        tok for tok in OUTCOME_TOKENS if tok in low
    })
    return bool(tokens), ";".join(tokens)


def line_is_rowlike_outcome(line: str) -> bool:
    low = line.lower()
    has_outcome = any(tok in low for tok in OUTCOME_TOKENS)
    assignment_like = (
        "=" in line
        or "," in line
        or "\t" in line
        or ":" in line
    )
    return has_outcome and assignment_like


def scan_plain_text(
    path: Path,
    candidate_ids: set[str],
):
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception:
        return []

    rows = []
    metadata_file = is_metadata_filename(path)

    # Candidate ID lookup is performed line by line.
    for i, line in enumerate(lines):
        ids = {sid for sid in candidate_ids if sid in line}
        if not ids:
            continue

        has_local_outcome, tokens = local_window_has_outcome(lines, i)

        if metadata_file:
            # Metadata files are outcome-bearing only for clear row-like local
            # outcome assignments on the same line.
            exposure = (
                "outcome_bearing"
                if line_is_rowlike_outcome(line)
                else "metadata_only"
            )
        else:
            exposure = (
                "outcome_bearing"
                if has_local_outcome
                else "metadata_only"
            )

        for sid in ids:
            rows.append({
                "sample_id": sid,
                "file": str(path),
                "artifact_type": path.suffix.lower(),
                "location": f"line:{i+1}",
                "exposure_type": exposure,
                "evidence": (
                    f"local_outcome_tokens={tokens}"
                    if has_local_outcome
                    else "no_local_outcome_tokens"
                ),
            })

    return rows


def scan_historical_outputs(
    candidate_rows,
    outputs_root: Path,
):
    candidate_ids = build_candidate_index(candidate_rows)
    files = list_historical_text_files(outputs_root)

    all_hits = []

    for path in tqdm(
        files,
        desc="Q1-R05A FIX1 structured historical scan",
        unit="file",
        dynamic_ncols=True,
    ):
        suffix = path.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            hits = scan_csv_tsv(path, candidate_ids)
        elif suffix in {".json", ".jsonl"}:
            hits = scan_json_jsonl(path, candidate_ids)
        else:
            hits = scan_plain_text(path, candidate_ids)

        all_hits.extend(hits)

    # Exact duplicate audit records do not add scientific information.
    dedup = {}
    for r in all_hits:
        key = (
            r["sample_id"],
            r["file"],
            r["location"],
            r["exposure_type"],
            r["evidence"],
        )
        dedup[key] = r

    return list(dedup.values()), len(files)


def build_outcome_file_summary(hit_rows):
    by_file = defaultdict(lambda: {
        "samples": set(),
        "rows": 0,
        "artifact_types": set(),
    })

    for r in hit_rows:
        if r["exposure_type"] != "outcome_bearing":
            continue
        entry = by_file[r["file"]]
        entry["samples"].add(r["sample_id"])
        entry["rows"] += 1
        entry["artifact_types"].add(r["artifact_type"])

    rows = []
    for file, entry in by_file.items():
        rows.append({
            "file": file,
            "artifact_types": ";".join(sorted(entry["artifact_types"])),
            "outcome_bearing_hit_rows": entry["rows"],
            "unique_candidate_samples": len(entry["samples"]),
            "candidate_sample_ids": ";".join(sorted(entry["samples"])),
        })

    rows.sort(
        key=lambda r: (
            -int(r["unique_candidate_samples"]),
            -int(r["outcome_bearing_hit_rows"]),
            r["file"],
        )
    )
    return rows


def list_text_files(root: Path):
    files = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        try:
            if (
                p.is_file()
                and p.suffix.lower() in TEXT_SUFFIXES
                and p.stat().st_size <= MAX_FILE_BYTES
            ):
                files.append(p)
        except OSError:
            continue
    return sorted(files)


def scan_code_references(code_root: Path, docs_root: Path):
    tokens = (
        "unseen_locked",
        "ColonDB",
        "CVC-300",
        "ETIS",
        "PolypGen",
    )
    rows = []

    files = list_text_files(code_root) + list_text_files(docs_root)
    for path in tqdm(
        files,
        desc="Q1-R05A FIX1 code/docs references",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).lower()
        except Exception:
            continue

        found = [
            token
            for token in tokens
            if token.lower() in text
        ]
        if found:
            rows.append({
                "file": str(path),
                "tokens": ";".join(found),
                "reference_only_not_exposure": 1,
            })
    return rows


def scan_polypgen_history(outputs_root: Path):
    s06_s07_dirs = []
    evidence = []

    if outputs_root.exists():
        for p in outputs_root.iterdir():
            if p.is_dir():
                low = p.name.lower()
                if low.startswith("s06") or low.startswith("s07"):
                    s06_s07_dirs.append(str(p))

    for path in list_text_files(outputs_root):
        if "Q1_R05A_confirmatory_target_eligibility_audit" in str(path):
            continue
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).lower()
        except Exception:
            continue

        low = path.name.lower() + "\n" + text
        if "polypgen" in low or "polyp_gen" in low:
            evidence.append({
                "file": str(path),
                "file_sha256": sha256_file(path),
            })

    return {
        "POLYPGEN_DEVELOPMENT_EXPOSED": bool(s06_s07_dirs and evidence),
        "eligible_as_untouched_confirmatory_target": False,
        "s06_s07_output_directories": sorted(s06_s07_dirs),
        "polypgen_reference_files": evidence[:200],
        "polypgen_reference_file_count": len(evidence),
    }


def run_preflight():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05A FIX1 protocol",
    )
    _, unseen, seen, unseen_counts, seen_counts = load_manifest_roles()
    upstream = validate_upstream_boundaries()

    print("===== Q1-R05A FIX1 PREFLIGHT =====")
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
    print("TTA=NO")
    print("predictor_fit=NO")
    print("historical_scan=NOT_RUN_IN_PREFLIGHT")
    print("PREFLIGHT_PASS")


def run(args):
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R05A FIX1 protocol",
    )
    _, unseen, seen, unseen_counts, seen_counts = load_manifest_roles()
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
        "R05A FIX1 protocol copy",
    )

    manifest_summary_rows = []
    for role, counts, total in (
        ("unseen_locked", unseen_counts, len(unseen)),
        ("seen_sanity", seen_counts, len(seen)),
    ):
        for dataset, count in sorted(counts.items()):
            manifest_summary_rows.append({
                "s01_role": role,
                "dataset": dataset,
                "count": count,
                "role_total": total,
            })

    manifest_summary_path = build_dir / "candidate_manifest_summary.csv"
    write_csv(
        manifest_summary_path,
        manifest_summary_rows,
        ["s01_role", "dataset", "count", "role_total"],
    )

    hits, scanned_files = scan_historical_outputs(
        unseen,
        ROOT / "outputs",
    )

    hits.sort(
        key=lambda r: (
            0 if r["exposure_type"] == "outcome_bearing" else 1,
            r["file"],
            r["sample_id"],
            r["location"],
        )
    )

    hits_path = build_dir / "historical_sample_id_hits.csv"
    write_csv(
        hits_path,
        hits,
        [
            "sample_id",
            "file",
            "artifact_type",
            "location",
            "exposure_type",
            "evidence",
        ],
    )

    outcome_summary = build_outcome_file_summary(hits)
    outcome_summary_path = build_dir / "outcome_bearing_file_summary.csv"
    write_csv(
        outcome_summary_path,
        outcome_summary,
        [
            "file",
            "artifact_types",
            "outcome_bearing_hit_rows",
            "unique_candidate_samples",
            "candidate_sample_ids",
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

    outcome_hits = [
        r for r in hits if r["exposure_type"] == "outcome_bearing"
    ]
    metadata_hits = [
        r for r in hits if r["exposure_type"] == "metadata_only"
    ]

    outcome_unique_samples = sorted({
        r["sample_id"] for r in outcome_hits
    })
    outcome_unique_files = sorted({
        r["file"] for r in outcome_hits
    })

    upstream.update({
        "historical_output_text_files_scanned": scanned_files,
        "unseen_locked_sample_ids": len(unseen),
        "structured_hit_rows_total": len(hits),
        "metadata_only_hit_rows": len(metadata_hits),
        "outcome_bearing_hit_rows": len(outcome_hits),
        "outcome_bearing_unique_candidate_samples": len(
            outcome_unique_samples
        ),
        "outcome_bearing_unique_files": len(outcome_unique_files),
        "target_pixels_not_opened_by_r05a_fix1": True,
    })

    upstream_path = build_dir / "upstream_boundary_audit.json"
    write_json(upstream_path, upstream)

    checks = {
        "manifest_sha_match": True,
        "unseen_total_636": len(unseen) == 636,
        "unseen_exact_dataset_counts":
            dict(unseen_counts) == EXPECTED_UNSEEN_COUNTS,
        "seen_total_162": len(seen) == 162,
        "seen_exact_dataset_counts":
            dict(seen_counts) == EXPECTED_SEEN_COUNTS,
        "r03_target_closed":
            upstream["r03_target_data_used"] is False,
        "r04_target_closed":
            upstream["r04_target_data_used"] is False,
        "target_pixels_not_opened_by_r05a":
            upstream["target_pixels_not_opened_by_r05a_fix1"] is True,
        "structured_outcome_bearing_unique_samples_zero":
            len(outcome_unique_samples) == 0,
    }

    decision = (
        DECISION_ELIGIBLE
        if all(checks.values())
        else DECISION_UNRESOLVED
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05A FIX1 CONFIRMATORY TARGET ELIGIBILITY RE-AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Candidate:",
        f"  unseen_locked total={len(unseen)}",
        f"  dataset_counts={dict(sorted(unseen_counts.items()))}",
        "",
        "Structured historical exposure scan:",
        f"  historical text files scanned={scanned_files}",
        f"  metadata-only hit rows={len(metadata_hits)}",
        f"  outcome-bearing hit rows={len(outcome_hits)}",
        (
            "  outcome-bearing unique candidate samples="
            f"{len(outcome_unique_samples)}"
        ),
        f"  outcome-bearing unique files={len(outcome_unique_files)}",
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
    for key, passed in checks.items():
        lines.append(
            f"  {key}={'PASS' if passed else 'FAIL'}"
        )

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    if decision == DECISION_ELIGIBLE:
        lines += [
            "",
            "If eligible:",
            "  next = Q1-R05B Frozen PAOT confirmation on "
            "ColonDB/CVC-300/ETIS",
        ]
    else:
        lines += [
            "",
            "If unresolved:",
            "  inspect outcome_bearing_file_summary.csv",
            "  do not open candidate target pixels for R05B",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "candidate_manifest_summary": manifest_summary_path,
        "historical_sample_id_hits": hits_path,
        "outcome_bearing_file_summary": outcome_summary_path,
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
        "candidate_dataset_counts":
            dict(sorted(unseen_counts.items())),
        "target_image_pixels_opened": False,
        "target_mask_pixels_opened": False,
        "model_inference_run": False,
        "tta_run": False,
        "predictor_fit": False,
        "metadata_only_hit_rows": len(metadata_hits),
        "outcome_bearing_hit_rows": len(outcome_hits),
        "outcome_bearing_unique_candidate_samples":
            len(outcome_unique_samples),
        "outcome_bearing_unique_files":
            len(outcome_unique_files),
        "polypgen_development_exposed":
            polypgen["POLYPGEN_DEVELOPMENT_EXPOSED"],
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
        build_dir
        / "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_LOCK.json"
    )
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        artifact = build_dir / meta["relative_path"]
        if sha256_file(artifact) != meta["sha256"]:
            raise RuntimeError(
                f"Artifact changed before commit: {name}"
            )

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R05A FIX1 LOCK:",
        args.output_dir
        / "Q1_R05A_CONFIRMATORY_TARGET_ELIGIBILITY_LOCK.json",
    )
    print("Q1-R05A FIX1 LOCK SHA256:", lock_sha)


def self_test():
    assert sum(EXPECTED_UNSEEN_COUNTS.values()) == 636
    assert sum(EXPECTED_SEEN_COUNTS.values()) == 162

    assert is_outcome_name("source_dice")
    assert is_outcome_name("benefit_probability")
    assert not is_outcome_name("sample_id")

    candidate_ids = {"sample::1", "sample::2"}
    row = {
        "sample_id": "sample::1",
        "source_dice": "0.8",
    }
    assert candidate_ids_from_row_values(
        row,
        candidate_ids,
    ) == {"sample::1"}

    # Direct JSON-record locality.
    toy = {
        "metadata": {
            "sample_id": "sample::1",
        },
        "metrics": {
            "dice": 0.9,
        },
    }
    recs = scan_json_object_records(
        toy,
        Path("toy.json"),
        candidate_ids,
    )
    assert recs
    assert all(
        r["exposure_type"] == "metadata_only"
        for r in recs
    )

    toy2 = {
        "sample_id": "sample::1",
        "source_dice": 0.9,
    }
    recs2 = scan_json_object_records(
        toy2,
        Path("toy.json"),
        candidate_ids,
    )
    assert len(recs2) == 1
    assert recs2[0]["exposure_type"] == "outcome_bearing"

    print("EXPECTED_COUNTS_TEST_PASS")
    print("OUTCOME_SCHEMA_TEST_PASS")
    print("CSV_ROW_LOCALITY_TEST_PASS")
    print("JSON_RECORD_LOCALITY_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05A FIX1: structured historical-exposure re-audit for "
            "unseen_locked ColonDB/CVC-300/ETIS without opening target pixels."
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
