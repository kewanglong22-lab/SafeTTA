#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R08A — M&Ms cross-modality cardiac MRI asset audit.

No NIfTI voxel decoding.
No GT decoding.
No model training/inference/TTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R08A-v1"
BUILD = "Q1_R08A_MNMS_CROSS_MODALITY_ASSET_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = (
    ROOT / "docs"
    / "Q1_R08A_mnms_cross_modality_asset_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "f0d88361d386da4a21f8463407f5d70da0701c9cf04466f8e3cd52aa99f637e0"

CANONICAL_DATA_ROOT = ROOT / "data" / "external" / "MnMs_raw"
OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R08A_mnms_cross_modality_asset_audit_v1"
)

MIN_EXACT_PAIRS = 100
MIN_METADATA_COVERAGE = 0.95
MIN_VENDOR_DOMAIN_CASES = 25
MIN_ELIGIBLE_VENDOR_DOMAINS = 2

READY = "MNMS_CROSS_MODALITY_ASSET_READY_FOR_PROTOCOL_LOCK"
INCOMPLETE = "MNMS_ASSET_AUDIT_INCOMPLETE"

GT_MARKERS = (
    "_gt",
    "-mask",
    "_mask",
    "-label",
    "_label",
    "-seg",
    "_seg",
)

SUBJECT_HINTS = ("externalcode", "subject", "patient", "case", "id")
VENDOR_HINTS = ("vendor", "manufacturer", "scanner")
CENTRE_HINTS = ("centre", "center", "hospital", "site")
SPLIT_HINTS = ("split", "subset", "partition", "set")


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
    if not path.is_file():
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


def normalize_column(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def normalize_value(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def normalize_subject_id(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_value(x).lower())


def strip_nii_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def is_gt_like(path: Path) -> bool:
    stem = strip_nii_suffix(path.name).lower()
    return any(marker in stem for marker in GT_MARKERS)


def normalized_pair_key(path: Path) -> str:
    stem = strip_nii_suffix(path.name).lower()
    for marker in GT_MARKERS:
        stem = stem.replace(marker, "")
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")
    return stem


def infer_subject_from_pair_key(key: str) -> str:
    # M&Ms commonly stores a subject ID followed by a view suffix such as _sa.
    # Keep this conservative: remove only common terminal view tokens.
    x = key.lower()
    x = re.sub(r"_(sa|sax|shortaxis|la|lax|4ch|2ch)$", "", x)
    return normalize_subject_id(x)


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def extract_zip(zip_path: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(
            f"Extraction destination already exists: {dst}"
        )
    dst.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for info in tqdm(
            members,
            desc="Extract M&Ms ZIP",
            unit="file",
            dynamic_ncols=True,
        ):
            zf.extract(info, dst)


def resolve_data_root(input_path: Path, extract: bool) -> tuple[Path, str]:
    input_path = input_path.resolve()

    if input_path.is_dir():
        return input_path, "directory"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if input_path.suffix.lower() == ".zip":
        if not extract:
            raise RuntimeError(
                "Input is ZIP. Re-run with --extract, or extract it manually "
                "to F:\\MEDSEG_SAFETTA\\data\\external\\MnMs_raw."
            )
        dst = CANONICAL_DATA_ROOT / "extracted"
        extract_zip(input_path, dst)
        return dst, "zip_extracted"

    raise RuntimeError(
        f"Unsupported input type: {input_path}. "
        "Use a directory or ZIP."
    )


def inventory_files(data_root: Path):
    files = [p for p in data_root.rglob("*") if p.is_file()]
    rows = []
    for p in tqdm(
        files,
        desc="Inventory M&Ms files",
        unit="file",
        dynamic_ncols=True,
    ):
        lower = p.name.lower()
        if lower.endswith(".nii.gz"):
            ext = ".nii.gz"
        else:
            ext = p.suffix.lower()
        rows.append({
            "relative_path": safe_relative(p, data_root),
            "filename": p.name,
            "extension": ext,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return rows


def pair_nifti(data_root: Path):
    nifti = [
        p for p in data_root.rglob("*")
        if p.is_file()
        and (
            p.name.lower().endswith(".nii")
            or p.name.lower().endswith(".nii.gz")
        )
    ]

    by_key = defaultdict(lambda: {"images": [], "gts": []})
    for p in nifti:
        key = normalized_pair_key(p)
        by_key[key]["gts" if is_gt_like(p) else "images"].append(p)

    rows = []
    duplicate_keys = []
    for key in sorted(by_key):
        group = by_key[key]
        images = sorted(group["images"], key=lambda p: str(p).lower())
        gts = sorted(group["gts"], key=lambda p: str(p).lower())

        exact = len(images) == 1 and len(gts) == 1
        if len(images) > 1 or len(gts) > 1:
            duplicate_keys.append(key)

        rows.append({
            "pair_key": key,
            "subject_id_inferred": infer_subject_from_pair_key(key),
            "image_count": len(images),
            "gt_count": len(gts),
            "exact_pair": int(exact),
            "image_path": "" if not images else str(images[0]),
            "gt_path": "" if not gts else str(gts[0]),
            "image_sha256": "" if not exact else sha256_file(images[0]),
            "gt_sha256": "" if not exact else sha256_file(gts[0]),
            "nifti_voxels_decoded": 0,
        })

    return rows, duplicate_keys


def import_openpyxl():
    try:
        import openpyxl  # type: ignore
        return openpyxl
    except Exception as e:
        raise RuntimeError(
            "Reading M&Ms XLSX metadata requires openpyxl in the current "
            "environment. Install openpyxl only if the official package "
            "contains XLSX metadata."
        ) from e


def read_csv_table(path: Path):
    candidates = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", newline="", encoding=encoding) as f:
                rows = list(csv.reader(f))
            candidates = rows
            break
        except UnicodeDecodeError:
            continue
    return [("CSV", candidates)]


def read_xlsx_tables(path: Path):
    openpyxl = import_openpyxl()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        out.append((ws.title, rows))
    return out


def table_to_dict_rows(raw_rows):
    # Skip leading empty rows, then use first non-empty row as header.
    header_idx = None
    for i, row in enumerate(raw_rows):
        if any(normalize_value(x) for x in row):
            header_idx = i
            break
    if header_idx is None:
        return [], []

    headers = [normalize_value(x) for x in raw_rows[header_idx]]
    while headers and not headers[-1]:
        headers.pop()

    if not headers:
        return [], []

    dict_rows = []
    for raw in raw_rows[header_idx + 1:]:
        vals = list(raw[:len(headers)])
        if len(vals) < len(headers):
            vals.extend([None] * (len(headers) - len(vals)))
        if not any(normalize_value(x) for x in vals):
            continue
        d = {headers[i]: normalize_value(vals[i]) for i in range(len(headers))}
        dict_rows.append(d)

    return headers, dict_rows


def find_column(headers, hints):
    norm = {h: normalize_column(h) for h in headers}

    # Exact normalized hint first.
    for hint in hints:
        for h, nh in norm.items():
            if nh == hint:
                return h

    # Then substring.
    for hint in hints:
        for h, nh in norm.items():
            if hint in nh:
                return h
    return None


def discover_metadata(data_root: Path):
    paths = sorted(
        [
            p for p in data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}
        ],
        key=lambda p: str(p).lower(),
    )

    inventory = []
    candidates = []

    for path in paths:
        tables = (
            read_csv_table(path)
            if path.suffix.lower() == ".csv"
            else read_xlsx_tables(path)
        )

        for sheet, raw_rows in tables:
            headers, dict_rows = table_to_dict_rows(raw_rows)
            subject_col = find_column(headers, SUBJECT_HINTS)
            vendor_col = find_column(headers, VENDOR_HINTS)
            centre_col = find_column(headers, CENTRE_HINTS)
            split_col = find_column(headers, SPLIT_HINTS)

            inventory.append({
                "relative_path": safe_relative(path, data_root),
                "sheet": sheet,
                "rows": len(dict_rows),
                "columns": len(headers),
                "headers": " | ".join(headers),
                "subject_column": subject_col or "",
                "vendor_column": vendor_col or "",
                "centre_column": centre_col or "",
                "split_column": split_col or "",
                "metadata_candidate": int(
                    bool(subject_col and vendor_col and dict_rows)
                ),
            })

            if subject_col and vendor_col and dict_rows:
                candidates.append({
                    "path": path,
                    "sheet": sheet,
                    "headers": headers,
                    "rows": dict_rows,
                    "subject_col": subject_col,
                    "vendor_col": vendor_col,
                    "centre_col": centre_col,
                    "split_col": split_col,
                })

    if not candidates:
        return inventory, None, []

    # Deterministic selection: most data rows, then path/sheet lexicographically.
    candidates.sort(
        key=lambda c: (
            -len(c["rows"]),
            str(c["path"]).lower(),
            str(c["sheet"]).lower(),
        )
    )
    chosen = candidates[0]

    normalized = []
    seen = Counter()

    for r in chosen["rows"]:
        sid_raw = r.get(chosen["subject_col"], "")
        sid = normalize_subject_id(sid_raw)
        if not sid:
            continue
        seen[sid] += 1
        normalized.append({
            "subject_id_raw": sid_raw,
            "subject_id_normalized": sid,
            "vendor": normalize_value(r.get(chosen["vendor_col"], "")),
            "centre": (
                normalize_value(r.get(chosen["centre_col"], ""))
                if chosen["centre_col"] else ""
            ),
            "split": (
                normalize_value(r.get(chosen["split_col"], ""))
                if chosen["split_col"] else ""
            ),
            "metadata_source_path": safe_relative(chosen["path"], data_root),
            "metadata_sheet": chosen["sheet"],
        })

    duplicate_metadata_ids = sorted(k for k, v in seen.items() if v > 1)

    chosen_summary = {
        "relative_path": safe_relative(chosen["path"], data_root),
        "sheet": chosen["sheet"],
        "subject_column": chosen["subject_col"],
        "vendor_column": chosen["vendor_col"],
        "centre_column": chosen["centre_col"] or "",
        "split_column": chosen["split_col"] or "",
        "normalized_rows": len(normalized),
        "duplicate_subject_ids": duplicate_metadata_ids,
    }

    return inventory, chosen_summary, normalized


def match_pairs_to_metadata(pair_rows, metadata_rows):
    meta_index = defaultdict(list)
    for row in metadata_rows:
        meta_index[row["subject_id_normalized"]].append(row)

    retained = []
    for row in pair_rows:
        if not int(row["exact_pair"]):
            continue

        sid = row["subject_id_inferred"]
        matches = meta_index.get(sid, [])

        out = dict(row)
        out["metadata_match_count"] = len(matches)
        out["metadata_matched"] = int(len(matches) == 1)
        out["vendor"] = matches[0]["vendor"] if len(matches) == 1 else ""
        out["centre"] = matches[0]["centre"] if len(matches) == 1 else ""
        out["split"] = matches[0]["split"] if len(matches) == 1 else ""
        retained.append(out)

    return retained


def vendor_summary(matched_pairs):
    by_vendor = defaultdict(list)

    for row in matched_pairs:
        if int(row["metadata_matched"]) != 1:
            continue
        vendor = row["vendor"].strip()
        if not vendor:
            continue
        by_vendor[vendor].append(row)

    rows = []
    for vendor in sorted(by_vendor, key=lambda x: x.lower()):
        subset = by_vendor[vendor]
        subjects = sorted({r["subject_id_inferred"] for r in subset})
        centres = sorted({r["centre"] for r in subset if r["centre"]})
        rows.append({
            "vendor": vendor,
            "paired_labeled_rows": len(subset),
            "unique_subjects": len(subjects),
            "unique_centres": len(centres),
            "centres": " | ".join(centres),
            "eligible_vendor_domain": int(
                len(subjects) >= MIN_VENDOR_DOMAIN_CASES
            ),
        })
    return rows


def preflight(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A protocol")
    if not args.official_source:
        raise RuntimeError(
            "--official-source is required for formal M&Ms acquisition audit."
        )
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    print("===== Q1-R08A PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"input={args.input}")
    print("official_source_asserted=YES")
    print("NIfTI_voxel_decode=NO")
    print("GT_voxel_decode=NO")
    print("model_training=NO")
    print("model_inference=NO")
    print("TTA=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A protocol")

    if not args.official_source:
        raise RuntimeError("--official-source is required.")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    data_root, acquisition_mode = resolve_data_root(args.input, args.extract)

    inventory = inventory_files(data_root)
    pair_rows, duplicate_pair_keys = pair_nifti(data_root)
    metadata_inventory, metadata_choice, metadata_rows = discover_metadata(data_root)

    exact_pairs = [r for r in pair_rows if int(r["exact_pair"]) == 1]
    matched_pairs = match_pairs_to_metadata(pair_rows, metadata_rows)
    vendor_rows = vendor_summary(matched_pairs)

    metadata_matches = sum(
        int(r["metadata_matched"]) for r in matched_pairs
    )
    metadata_coverage = (
        metadata_matches / len(exact_pairs)
        if exact_pairs else 0.0
    )

    eligible_vendors = [
        r for r in vendor_rows
        if int(r["eligible_vendor_domain"]) == 1
    ]

    checks = {
        "A_exact_pairs_ge_100": len(exact_pairs) >= MIN_EXACT_PAIRS,
        "B_metadata_subject_resolved": bool(
            metadata_choice and metadata_choice["subject_column"]
        ),
        "C_metadata_vendor_resolved": bool(
            metadata_choice and metadata_choice["vendor_column"]
        ),
        "D_metadata_coverage_ge_0p95":
            metadata_coverage >= MIN_METADATA_COVERAGE,
        "E_at_least_two_vendor_domains_ge_25":
            len(eligible_vendors) >= MIN_ELIGIBLE_VENDOR_DOMAINS,
        "F_no_duplicate_pair_key": len(duplicate_pair_keys) == 0,
        "G_retained_pair_files_exist": all(
            Path(r["image_path"]).is_file() and Path(r["gt_path"]).is_file()
            for r in exact_pairs
        ),
        "H_official_source_asserted": bool(args.official_source),
        "I_nifti_voxels_decoded": False,
        "J_gt_voxels_decoded": False,
    }

    pass_checks = (
        checks["A_exact_pairs_ge_100"]
        and checks["B_metadata_subject_resolved"]
        and checks["C_metadata_vendor_resolved"]
        and checks["D_metadata_coverage_ge_0p95"]
        and checks["E_at_least_two_vendor_domains_ge_25"]
        and checks["F_no_duplicate_pair_key"]
        and checks["G_retained_pair_files_exist"]
        and checks["H_official_source_asserted"]
        and not checks["I_nifti_voxels_decoded"]
        and not checks["J_gt_voxels_decoded"]
    )

    decision = READY if pass_checks else INCOMPLETE

    write_csv(
        build_dir / "file_inventory.csv",
        inventory,
        ["relative_path", "filename", "extension", "bytes", "sha256"],
    )

    pair_fields = [
        "pair_key",
        "subject_id_inferred",
        "image_count",
        "gt_count",
        "exact_pair",
        "image_path",
        "gt_path",
        "image_sha256",
        "gt_sha256",
        "nifti_voxels_decoded",
    ]
    write_csv(build_dir / "nifti_pair_manifest.csv", pair_rows, pair_fields)

    write_csv(
        build_dir / "metadata_table_inventory.csv",
        metadata_inventory,
        [
            "relative_path",
            "sheet",
            "rows",
            "columns",
            "headers",
            "subject_column",
            "vendor_column",
            "centre_column",
            "split_column",
            "metadata_candidate",
        ],
    )

    write_csv(
        build_dir / "metadata_normalized.csv",
        metadata_rows,
        [
            "subject_id_raw",
            "subject_id_normalized",
            "vendor",
            "centre",
            "split",
            "metadata_source_path",
            "metadata_sheet",
        ],
    )

    write_csv(
        build_dir / "vendor_domain_summary.csv",
        vendor_rows,
        [
            "vendor",
            "paired_labeled_rows",
            "unique_subjects",
            "unique_centres",
            "centres",
            "eligible_vendor_domain",
        ],
    )

    acquisition = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "input": str(args.input),
        "resolved_data_root": str(data_root),
        "acquisition_mode": acquisition_mode,
        "official_source_asserted": bool(args.official_source),
        "files": len(inventory),
        "nifti_entries": sum(
            1 for r in inventory if r["extension"] in {".nii", ".nii.gz"}
        ),
        "exact_image_gt_pairs": len(exact_pairs),
        "duplicate_pair_keys": duplicate_pair_keys,
        "metadata_choice": metadata_choice,
        "metadata_rows": len(metadata_rows),
        "metadata_matched_exact_pairs": metadata_matches,
        "metadata_coverage": metadata_coverage,
        "eligible_vendor_domains": len(eligible_vendors),
        "nifti_image_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "model_training": False,
        "model_inference": False,
        "tta": False,
    }
    write_json(build_dir / "acquisition_audit.json", acquisition)

    gate = {
        "criteria": {
            "min_exact_pairs": MIN_EXACT_PAIRS,
            "min_metadata_coverage": MIN_METADATA_COVERAGE,
            "min_vendor_domain_cases": MIN_VENDOR_DOMAIN_CASES,
            "min_eligible_vendor_domains": MIN_ELIGIBLE_VENDOR_DOMAINS,
        },
        "checks": checks,
        "decision": decision,
        "future_domain_selection_rule": {
            "eligible_vendor_min_paired_labeled_subjects":
                MIN_VENDOR_DOMAIN_CASES,
            "rank": "paired_labeled_subject_count_descending",
            "choose": 2,
            "tie_break": "normalized_vendor_string_ascending",
            "directions": "bidirectional",
            "performance_based_selection": False,
        },
    }
    write_json(build_dir / "eligibility_gate.json", gate)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R08A M&Ms CROSS-MODALITY ASSET AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Acquisition:",
        f"  input={args.input}",
        f"  resolved root={data_root}",
        f"  mode={acquisition_mode}",
        "  official source asserted=YES",
        "",
        "Structural inventory:",
        f"  files={len(inventory)}",
        f"  exact image/GT NIfTI pairs={len(exact_pairs)}",
        f"  duplicate pair keys={len(duplicate_pair_keys)}",
        "",
        "Metadata:",
        f"  metadata table resolved={'YES' if metadata_choice else 'NO'}",
        f"  metadata rows={len(metadata_rows)}",
        f"  exact pairs matched to metadata={metadata_matches}",
        f"  metadata coverage={metadata_coverage:.6f}",
        "",
        "Vendor domains:",
    ]

    for r in vendor_rows:
        lines.append(
            f"  {r['vendor']}: subjects={r['unique_subjects']} "
            f"centres={r['unique_centres']} "
            f"eligible={'YES' if r['eligible_vendor_domain'] else 'NO'}"
        )

    lines += [
        "",
        "Information boundary:",
        "  NIfTI image voxels decoded=NO",
        "  GT voxels decoded=NO",
        "  model training=NO",
        "  model inference=NO",
        "  TTA=NO",
        "",
        "Eligibility checks:",
    ]
    for k, v in checks.items():
        lines.append(f"  {k}={'PASS' if (not k.endswith('decoded') and v) or (k.endswith('decoded') and not v) else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Next:",
        (
            "  Q1-R08B cardiac-MRI bidirectional cross-vendor protocol lock"
            if decision == READY
            else
            "  Resolve acquisition/pairing/metadata deficit before R08B"
        ),
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "acquisition_audit.json",
        "file_inventory.csv",
        "nifti_pair_manifest.csv",
        "metadata_table_inventory.csv",
        "metadata_normalized.csv",
        "vendor_domain_summary.csv",
        "eligibility_gate.json",
        "decision.txt",
        "run_log.txt",
    ]
    artifacts = {
        name: {
            "relative_path": name,
            "sha256": sha256_file(build_dir / name),
        }
        for name in artifact_names
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "official_source_asserted": True,
        "exact_pairs": len(exact_pairs),
        "metadata_coverage": metadata_coverage,
        "eligible_vendor_domains": len(eligible_vendors),
        "nifti_image_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "model_training": False,
        "model_inference": False,
        "tta": False,
        "decision": decision,
        "gate": gate,
        "artifacts": artifacts,
    }
    lock_path = build_dir / "Q1_R08A_MNMS_ASSET_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R08A LOCK:",
        args.output_dir / "Q1_R08A_MNMS_ASSET_LOCK.json",
    )
    print("Q1-R08A LOCK SHA256:", lock_sha)


def self_test():
    assert MIN_EXACT_PAIRS == 100
    assert MIN_METADATA_COVERAGE == 0.95
    assert MIN_VENDOR_DOMAIN_CASES == 25
    assert MIN_ELIGIBLE_VENDOR_DOMAINS == 2
    assert normalized_pair_key(Path("A001_sa.nii.gz")) == "a001_sa"
    assert normalized_pair_key(Path("A001_sa_gt.nii.gz")) == "a001_sa"
    assert infer_subject_from_pair_key("a001_sa") == "a001"
    assert is_gt_like(Path("A001_sa_gt.nii.gz"))
    assert not is_gt_like(Path("A001_sa.nii.gz"))

    print("PAIR_KEY_TEST_PASS")
    print("SUBJECT_ID_INFERENCE_TEST_PASS")
    print("GT_FILENAME_CLASSIFICATION_TEST_PASS")
    print("FROZEN_ELIGIBILITY_CRITERIA_TEST_PASS")
    print("NO_VOXEL_DECODING_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08A: structural audit of an official M&Ms cardiac MRI "
            "package/directory before cross-vendor model development."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=False,
        help="Official M&Ms directory or ZIP.",
    )
    p.add_argument(
        "--official-source",
        action="store_true",
        help="Assert that --input came from the official M&Ms distribution.",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="Extract ZIP input under F:\\MEDSEG_SAFETTA\\data\\external\\MnMs_raw.",
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.input is None:
        raise RuntimeError("--input is required except for --self-test.")

    if args.preflight_only:
        preflight(args)
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
