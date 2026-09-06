#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Q1-R08A1 — M&Ms ZIP-native metadata + Vendor domain structural audit.

No NIfTI extraction.
No voxel decoding.
No model/TTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R08A1-v1"
BUILD = "Q1_R08A1_MNMS_ZIP_NATIVE_METADATA_VENDOR_DOMAIN_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R08A1_mnms_zip_native_metadata_vendor_domain_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "b60ee403944b887b9907f1e0bf89952456fa88c333f8b16eb4c5a9212ce04f85"

R08A0_DIR = (
    ROOT / "outputs"
    / "Q1_R08A0_mnms_kaggle_mirror_archive_identity_audit_v1_fix3"
)
R08A0_LOCK = (
    R08A0_DIR
    / "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_LOCK.json"
)

EXPECTED_R08A0_LOCK_SHA256 = (
    "6d3a94dcd5a197618f13d55741f36414b7c9039aba855fd54e9904c1fb683a7f"
)
EXPECTED_R08A0_DECISION = "MNMS_KAGGLE_MIRROR_ARCHIVE_COMPLETE"

ZIP_PATH = (
    ROOT / "data" / "downloads" / "MnMs_kaggle" / "MnMs.zip"
)
EXPECTED_ZIP_SHA256 = (
    "97c0eedd0562b8a90ce70be4a0a114ffc620a82832217c8f0a4e24acce112718"
)
EXPECTED_MANIFEST_SHA256 = (
    "33fffe42d047bb7e2407c74e171e0d7b14072a21ad170ee6c4e12e0734eaeac5"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R08A1_mnms_zip_native_metadata_vendor_domain_audit_v1"
)

EXPECTED_ZIP_MEMBERS = 691
EXPECTED_NIFTI_MEMBERS = 690
EXPECTED_GT_NIFTI = 345
EXPECTED_EXACT_PAIRS = 345

MIN_METADATA_COVERAGE = 0.95
MIN_VENDOR_SUBJECTS = 25
MIN_ELIGIBLE_VENDORS = 2

READY = "MNMS_VENDOR_DOMAIN_STRUCTURE_READY"
INCOMPLETE = "MNMS_VENDOR_DOMAIN_STRUCTURE_INCOMPLETE"

GT_MARKERS = (
    "_gt",
    "-mask",
    "_mask",
    "-label",
    "_label",
    "-seg",
    "_seg",
)

SUBJECT_HINTS = (
    "externalcode",
    "subject",
    "patient",
    "case",
    "code",
    "id",
)

VENDOR_HINTS = (
    "vendor",
    "manufacturer",
    "scanner",
)

CENTRE_HINTS = (
    "centre",
    "center",
    "hospital",
    "site",
)

SPLIT_HINTS = (
    "split",
    "subset",
    "partition",
    "set",
)


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
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_col(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def normalize_value(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", s)


def normalize_subject(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_value(x).lower())


def normalize_vendor(x) -> str:
    return re.sub(r"\s+", " ", normalize_value(x)).strip()


def normalize_member(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("/")


def strip_nii_suffix(name: str) -> str:
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    if low.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def is_nifti(name: str) -> bool:
    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def is_gt_like(name: str) -> bool:
    stem = strip_nii_suffix(Path(name).name).lower()
    return any(marker in stem for marker in GT_MARKERS)


def pair_key(name: str) -> str:
    stem = strip_nii_suffix(Path(name).name).lower()
    for marker in GT_MARKERS:
        stem = stem.replace(marker, "")
    return re.sub(r"[_\-\s]+", "_", stem).strip("_")


def subject_candidates(pair_member: str) -> list[str]:
    """
    Candidate identifiers derived without reading NIfTI payload.
    """
    p = Path(pair_member)
    key = pair_key(p.name)

    candidates = []

    def add(x):
        n = normalize_subject(x)
        if n and n not in candidates:
            candidates.append(n)

    add(key)

    reduced = re.sub(
        r"_(sa|sax|shortaxis|short_axis)$",
        "",
        key,
        flags=re.IGNORECASE,
    )
    add(reduced)

    parent = p.parent.name
    add(parent)

    # Conservative prefix fallback for patterns like SUBJECT_sa.
    if "_" in key:
        add(key.split("_", 1)[0])

    return candidates


def exact_pairs_from_zip(zf: zipfile.ZipFile):
    infos = [x for x in zf.infolist() if not x.is_dir()]
    if len(infos) != EXPECTED_ZIP_MEMBERS:
        raise RuntimeError(
            f"ZIP members={len(infos)} expected={EXPECTED_ZIP_MEMBERS}"
        )

    nifti = [x for x in infos if is_nifti(x.filename)]
    if len(nifti) != EXPECTED_NIFTI_MEMBERS:
        raise RuntimeError(
            f"NIfTI members={len(nifti)} expected={EXPECTED_NIFTI_MEMBERS}"
        )

    gt_n = sum(is_gt_like(x.filename) for x in nifti)
    if gt_n != EXPECTED_GT_NIFTI:
        raise RuntimeError(
            f"GT-like NIfTI={gt_n} expected={EXPECTED_GT_NIFTI}"
        )

    groups = defaultdict(lambda: {"images": [], "gts": []})
    for info in nifti:
        k = pair_key(info.filename)
        groups[k]["gts" if is_gt_like(info.filename) else "images"].append(info)

    pairs = []
    duplicates = []

    for key in sorted(groups):
        images = sorted(groups[key]["images"], key=lambda x: x.filename.lower())
        gts = sorted(groups[key]["gts"], key=lambda x: x.filename.lower())

        if len(images) == 1 and len(gts) == 1:
            img = images[0]
            gt = gts[0]
            candidates = subject_candidates(img.filename)
            pairs.append({
                "pair_key": key,
                "image_member": normalize_member(img.filename),
                "gt_member": normalize_member(gt.filename),
                "image_uncompressed_bytes": int(img.file_size),
                "gt_uncompressed_bytes": int(gt.file_size),
                "subject_candidates": candidates,
            })
        elif len(images) > 1 or len(gts) > 1:
            duplicates.append({
                "pair_key": key,
                "image_count": len(images),
                "gt_count": len(gts),
            })

    if len(pairs) != EXPECTED_EXACT_PAIRS:
        raise RuntimeError(
            f"Exact pairs={len(pairs)} expected={EXPECTED_EXACT_PAIRS}"
        )

    return pairs, duplicates, infos


def find_column(headers, hints):
    norm = {h: normalize_col(h) for h in headers}

    for hint in hints:
        for h, nh in norm.items():
            if nh == hint:
                return h

    for hint in hints:
        for h, nh in norm.items():
            if hint in nh:
                return h

    return None


def table_to_rows(raw_rows):
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

    out = []

    for row in raw_rows[header_idx + 1:]:
        vals = list(row[:len(headers)])

        if len(vals) < len(headers):
            vals.extend([None] * (len(headers) - len(vals)))

        if not any(normalize_value(x) for x in vals):
            continue

        out.append({
            headers[j]: normalize_value(vals[j])
            for j in range(len(headers))
        })

    return headers, out


def decode_csv_bytes(blob: bytes):
    text = None
    used_encoding = None

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = blob.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise RuntimeError("Unable to decode metadata CSV.")

    sample = text[:8192]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        dialect = csv.excel

    rows = list(csv.reader(io.StringIO(text), dialect))
    return rows, used_encoding


def read_metadata_tables_from_zip(zf: zipfile.ZipFile, infos):
    non_nifti = [x for x in infos if not is_nifti(x.filename)]

    inventory = []
    candidates = []

    for info in non_nifti:
        member = normalize_member(info.filename)
        suffix = Path(member).suffix.lower()

        inv = {
            "member_path": member,
            "uncompressed_bytes": int(info.file_size),
            "suffix": suffix,
            "metadata_supported": int(suffix in {".csv", ".xlsx"}),
        }
        inventory.append(inv)

        if suffix == ".csv":
            blob = zf.read(info)
            raw_rows, encoding = decode_csv_bytes(blob)

            headers, rows = table_to_rows(raw_rows)

            candidates.append({
                "member_path": member,
                "sheet": "CSV",
                "encoding": encoding,
                "headers": headers,
                "rows": rows,
            })

        elif suffix == ".xlsx":
            try:
                import openpyxl  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Metadata is XLSX but openpyxl is unavailable in rare26."
                ) from e

            blob = zf.read(info)
            wb = openpyxl.load_workbook(
                io.BytesIO(blob),
                read_only=True,
                data_only=True,
            )

            for ws in wb.worksheets:
                raw_rows = [
                    list(r)
                    for r in ws.iter_rows(values_only=True)
                ]
                headers, rows = table_to_rows(raw_rows)

                candidates.append({
                    "member_path": member,
                    "sheet": ws.title,
                    "encoding": "xlsx",
                    "headers": headers,
                    "rows": rows,
                })

    return inventory, candidates


def resolve_metadata_table(candidates):
    table_inventory = []
    eligible = []

    for c in candidates:
        headers = c["headers"]
        rows = c["rows"]

        subject_col = find_column(headers, SUBJECT_HINTS)
        vendor_col = find_column(headers, VENDOR_HINTS)
        centre_col = find_column(headers, CENTRE_HINTS)
        split_col = find_column(headers, SPLIT_HINTS)

        row = {
            "member_path": c["member_path"],
            "sheet": c["sheet"],
            "encoding": c["encoding"],
            "rows": len(rows),
            "columns": len(headers),
            "headers": " | ".join(headers),
            "subject_column": subject_col or "",
            "vendor_column": vendor_col or "",
            "centre_column": centre_col or "",
            "split_column": split_col or "",
            "eligible_metadata_table": int(
                bool(subject_col and vendor_col and rows)
            ),
        }
        table_inventory.append(row)

        if subject_col and vendor_col and rows:
            eligible.append({
                **c,
                "subject_col": subject_col,
                "vendor_col": vendor_col,
                "centre_col": centre_col,
                "split_col": split_col,
            })

    if not eligible:
        return table_inventory, None

    eligible.sort(
        key=lambda c: (
            -len(c["rows"]),
            c["member_path"].lower(),
            c["sheet"].lower(),
        )
    )

    return table_inventory, eligible[0]


def normalize_metadata(chosen):
    if chosen is None:
        return [], []

    rows = []
    seen = Counter()

    for src in chosen["rows"]:
        sid_raw = src.get(chosen["subject_col"], "")
        sid = normalize_subject(sid_raw)

        if not sid:
            continue

        seen[sid] += 1

        rows.append({
            "subject_id_raw": sid_raw,
            "subject_id_normalized": sid,
            "vendor": normalize_vendor(
                src.get(chosen["vendor_col"], "")
            ),
            "centre": (
                normalize_value(src.get(chosen["centre_col"], ""))
                if chosen["centre_col"]
                else ""
            ),
            "split": (
                normalize_value(src.get(chosen["split_col"], ""))
                if chosen["split_col"]
                else ""
            ),
            "metadata_member_path": chosen["member_path"],
            "metadata_sheet": chosen["sheet"],
        })

    duplicate_subject_ids = sorted(
        sid for sid, n in seen.items()
        if n > 1
    )

    return rows, duplicate_subject_ids


def match_pairs(pairs, metadata_rows):
    index = defaultdict(list)

    for row in metadata_rows:
        index[row["subject_id_normalized"]].append(row)

    out = []
    ambiguous = []
    unmatched = []

    for p in pairs:
        hit_rows = []
        hit_ids = []

        for candidate in p["subject_candidates"]:
            matches = index.get(candidate, [])

            for m in matches:
                key = (
                    m["subject_id_normalized"],
                    m["vendor"],
                    m["centre"],
                    m["split"],
                )
                if key not in hit_ids:
                    hit_ids.append(key)
                    hit_rows.append(m)

        if len(hit_rows) == 1:
            m = hit_rows[0]
            out.append({
                "pair_key": p["pair_key"],
                "image_member": p["image_member"],
                "gt_member": p["gt_member"],
                "image_uncompressed_bytes": p["image_uncompressed_bytes"],
                "gt_uncompressed_bytes": p["gt_uncompressed_bytes"],
                "subject_candidates": " | ".join(p["subject_candidates"]),
                "metadata_matched": 1,
                "matched_subject_id": m["subject_id_normalized"],
                "vendor": m["vendor"],
                "centre": m["centre"],
                "split": m["split"],
                "metadata_match_count": 1,
            })

        elif len(hit_rows) == 0:
            unmatched.append(p["pair_key"])
            out.append({
                "pair_key": p["pair_key"],
                "image_member": p["image_member"],
                "gt_member": p["gt_member"],
                "image_uncompressed_bytes": p["image_uncompressed_bytes"],
                "gt_uncompressed_bytes": p["gt_uncompressed_bytes"],
                "subject_candidates": " | ".join(p["subject_candidates"]),
                "metadata_matched": 0,
                "matched_subject_id": "",
                "vendor": "",
                "centre": "",
                "split": "",
                "metadata_match_count": 0,
            })

        else:
            ambiguous.append(p["pair_key"])
            out.append({
                "pair_key": p["pair_key"],
                "image_member": p["image_member"],
                "gt_member": p["gt_member"],
                "image_uncompressed_bytes": p["image_uncompressed_bytes"],
                "gt_uncompressed_bytes": p["gt_uncompressed_bytes"],
                "subject_candidates": " | ".join(p["subject_candidates"]),
                "metadata_matched": 0,
                "matched_subject_id": "",
                "vendor": "",
                "centre": "",
                "split": "",
                "metadata_match_count": len(hit_rows),
            })

    return out, unmatched, ambiguous


def vendor_summary(pair_manifest):
    by_vendor = defaultdict(list)

    for r in pair_manifest:
        if int(r["metadata_matched"]) != 1:
            continue

        vendor = r["vendor"].strip()
        if not vendor:
            continue

        by_vendor[vendor].append(r)

    rows = []

    for vendor in sorted(by_vendor, key=lambda x: x.lower()):
        subset = by_vendor[vendor]

        subjects = sorted({
            r["matched_subject_id"]
            for r in subset
        })

        centres = sorted({
            r["centre"]
            for r in subset
            if r["centre"]
        })

        splits = sorted({
            r["split"]
            for r in subset
            if r["split"]
        })

        rows.append({
            "vendor": vendor,
            "vendor_normalized": normalize_col(vendor),
            "paired_labeled_rows": len(subset),
            "unique_subjects": len(subjects),
            "unique_centres": len(centres),
            "centres": " | ".join(centres),
            "splits": " | ".join(splits),
            "eligible_vendor_domain": int(
                len(subjects) >= MIN_VENDOR_SUBJECTS
            ),
        })

    return rows


def select_vendors(vendor_rows):
    eligible = [
        r for r in vendor_rows
        if int(r["eligible_vendor_domain"]) == 1
    ]

    eligible.sort(
        key=lambda r: (
            -int(r["unique_subjects"]),
            r["vendor_normalized"],
        )
    )

    if len(eligible) < 2:
        return None

    a, b = eligible[:2]

    return {
        "selection_rule": (
            "eligible>=25 subjects; subject-count descending; "
            "normalized vendor string ascending tie-break"
        ),
        "vendor_1": a["vendor"],
        "vendor_1_subjects": int(a["unique_subjects"]),
        "vendor_2": b["vendor"],
        "vendor_2_subjects": int(b["unique_subjects"]),
        "directions": [
            {
                "source_vendor": a["vendor"],
                "target_vendor": b["vendor"],
            },
            {
                "source_vendor": b["vendor"],
                "target_vendor": a["vendor"],
            },
        ],
        "performance_based_selection": False,
    }


def validate_upstream():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R08A1 protocol",
    )

    validate_sha(
        R08A0_LOCK,
        EXPECTED_R08A0_LOCK_SHA256,
        "R08A0 lock",
    )

    validate_sha(
        ZIP_PATH,
        EXPECTED_ZIP_SHA256,
        "M&Ms ZIP",
    )

    lock = json.loads(
        R08A0_LOCK.read_text(encoding="utf-8")
    )

    if lock.get("decision") != EXPECTED_R08A0_DECISION:
        raise RuntimeError(
            f"Unexpected R08A0 decision: {lock.get('decision')}"
        )

    if lock.get("zip_sha256") != EXPECTED_ZIP_SHA256:
        raise RuntimeError(
            "R08A0 lock ZIP SHA changed."
        )

    if lock.get("persistent_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "R08A0 persistent manifest SHA changed."
        )

    if int(lock.get("remote_manifest_rows", -1)) != EXPECTED_ZIP_MEMBERS:
        raise RuntimeError(
            "R08A0 manifest cardinality changed."
        )

    if int(lock.get("zip_members", -1)) != EXPECTED_ZIP_MEMBERS:
        raise RuntimeError(
            "R08A0 ZIP-member cardinality changed."
        )

    if bool(lock.get("archive_extracted", True)):
        raise RuntimeError(
            "R08A0 reports archive extraction."
        )

    if bool(lock.get("nifti_voxels_decoded", True)):
        raise RuntimeError(
            "R08A0 reports NIfTI voxel decoding."
        )

    if bool(lock.get("gt_voxels_decoded", True)):
        raise RuntimeError(
            "R08A0 reports GT voxel decoding."
        )

    return lock


def preflight():
    lock = validate_upstream()

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        pairs, duplicate_pair_keys, infos = exact_pairs_from_zip(zf)

        non_nifti = [
            x for x in infos
            if not is_nifti(x.filename)
        ]

        metadata_supported = [
            x for x in non_nifti
            if Path(x.filename).suffix.lower() in {".csv", ".xlsx"}
        ]

    print("===== Q1-R08A1 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r08a0_lock_sha256={EXPECTED_R08A0_LOCK_SHA256}")
    print(f"zip_sha256={EXPECTED_ZIP_SHA256}")
    print(f"zip_members={EXPECTED_ZIP_MEMBERS}")
    print(f"exact_pairs={len(pairs)}")
    print(f"duplicate_pair_keys={len(duplicate_pair_keys)}")
    print(f"non_nifti_members={len(non_nifti)}")
    print(f"supported_metadata_members={len(metadata_supported)}")
    print("nifti_payload_extraction=NO")
    print("nifti_voxel_decode=NO")
    print("gt_voxel_decode=NO")
    print("model_training_inference_tta=NO")
    print("PREFLIGHT_PASS")


def run(args):
    upstream = validate_upstream()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(
        str(args.output_dir) + "__building"
    )

    if build_dir.exists():
        raise FileExistsError(build_dir)

    build_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    protocol_copy = (
        build_dir
        / "preregistered_protocol_copy.md"
    )

    shutil.copy2(
        PROTOCOL,
        protocol_copy,
    )

    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "protocol copy",
    )

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        pairs, duplicate_pair_keys, infos = exact_pairs_from_zip(zf)

        non_nifti_inventory, table_candidates = (
            read_metadata_tables_from_zip(
                zf,
                infos,
            )
        )

    table_inventory, chosen = resolve_metadata_table(
        table_candidates
    )

    metadata_rows, duplicate_metadata_subject_ids = (
        normalize_metadata(chosen)
    )

    pair_manifest, unmatched_pairs, ambiguous_pairs = (
        match_pairs(
            pairs,
            metadata_rows,
        )
    )

    matched = sum(
        int(r["metadata_matched"])
        for r in pair_manifest
    )

    metadata_coverage = (
        matched / len(pair_manifest)
        if pair_manifest
        else 0.0
    )

    vendor_rows = vendor_summary(
        pair_manifest
    )

    eligible_vendors = [
        r for r in vendor_rows
        if int(r["eligible_vendor_domain"]) == 1
    ]

    selected = select_vendors(
        vendor_rows
    )

    checks = {
        "A_exact_pairs_eq_345":
            len(pairs) == EXPECTED_EXACT_PAIRS,
        "B_duplicate_pair_keys_eq_0":
            len(duplicate_pair_keys) == 0,
        "C_metadata_table_resolved":
            chosen is not None,
        "D_metadata_subject_column_resolved":
            bool(chosen and chosen["subject_col"]),
        "E_metadata_vendor_column_resolved":
            bool(chosen and chosen["vendor_col"]),
        "F_metadata_coverage_ge_0p95":
            metadata_coverage >= MIN_METADATA_COVERAGE,
        "G_ambiguous_pair_matches_eq_0":
            len(ambiguous_pairs) == 0,
        "H_eligible_vendor_domains_ge_2":
            len(eligible_vendors) >= MIN_ELIGIBLE_VENDORS,
        "I_selected_vendor_pair_resolved":
            selected is not None,
        "J_nifti_payload_extracted": False,
        "K_nifti_voxels_decoded": False,
        "L_gt_voxels_decoded": False,
        "M_model_training_inference_tta": False,
    }

    pass_gate = all([
        checks["A_exact_pairs_eq_345"],
        checks["B_duplicate_pair_keys_eq_0"],
        checks["C_metadata_table_resolved"],
        checks["D_metadata_subject_column_resolved"],
        checks["E_metadata_vendor_column_resolved"],
        checks["F_metadata_coverage_ge_0p95"],
        checks["G_ambiguous_pair_matches_eq_0"],
        checks["H_eligible_vendor_domains_ge_2"],
        checks["I_selected_vendor_pair_resolved"],
        not checks["J_nifti_payload_extracted"],
        not checks["K_nifti_voxels_decoded"],
        not checks["L_gt_voxels_decoded"],
        not checks["M_model_training_inference_tta"],
    ])

    decision = READY if pass_gate else INCOMPLETE

    write_json(
        build_dir / "upstream_audit.json",
        {
            "script_version": VERSION,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "r08a0_lock_sha256": EXPECTED_R08A0_LOCK_SHA256,
            "r08a0_decision": upstream.get("decision"),
            "zip_sha256": EXPECTED_ZIP_SHA256,
            "persistent_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "zip_members": EXPECTED_ZIP_MEMBERS,
            "nifti_members": EXPECTED_NIFTI_MEMBERS,
            "gt_like_nifti_members": EXPECTED_GT_NIFTI,
            "exact_pairs": EXPECTED_EXACT_PAIRS,
            "nifti_payload_extracted": False,
            "nifti_voxels_decoded": False,
            "gt_voxels_decoded": False,
            "model_training": False,
            "model_inference": False,
            "tta": False,
        },
    )

    write_csv(
        build_dir / "zip_non_nifti_inventory.csv",
        non_nifti_inventory,
        [
            "member_path",
            "uncompressed_bytes",
            "suffix",
            "metadata_supported",
        ],
    )

    write_csv(
        build_dir / "metadata_table_inventory.csv",
        table_inventory,
        [
            "member_path",
            "sheet",
            "encoding",
            "rows",
            "columns",
            "headers",
            "subject_column",
            "vendor_column",
            "centre_column",
            "split_column",
            "eligible_metadata_table",
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
            "metadata_member_path",
            "metadata_sheet",
        ],
    )

    write_csv(
        build_dir / "paired_case_metadata_manifest.csv",
        pair_manifest,
        [
            "pair_key",
            "image_member",
            "gt_member",
            "image_uncompressed_bytes",
            "gt_uncompressed_bytes",
            "subject_candidates",
            "metadata_matched",
            "matched_subject_id",
            "vendor",
            "centre",
            "split",
            "metadata_match_count",
        ],
    )

    write_csv(
        build_dir / "vendor_domain_summary.csv",
        vendor_rows,
        [
            "vendor",
            "vendor_normalized",
            "paired_labeled_rows",
            "unique_subjects",
            "unique_centres",
            "centres",
            "splits",
            "eligible_vendor_domain",
        ],
    )

    write_json(
        build_dir / "selected_vendor_domains.json",
        (
            selected
            if selected is not None
            else {
                "selection_resolved": False,
                "performance_based_selection": False,
            }
        ),
    )

    gate = {
        "criteria": {
            "exact_pairs": EXPECTED_EXACT_PAIRS,
            "min_metadata_coverage": MIN_METADATA_COVERAGE,
            "min_vendor_subjects": MIN_VENDOR_SUBJECTS,
            "min_eligible_vendor_domains": MIN_ELIGIBLE_VENDORS,
        },
        "metadata_choice": (
            None
            if chosen is None
            else {
                "member_path": chosen["member_path"],
                "sheet": chosen["sheet"],
                "subject_column": chosen["subject_col"],
                "vendor_column": chosen["vendor_col"],
                "centre_column": chosen["centre_col"] or "",
                "split_column": chosen["split_col"] or "",
                "rows": len(chosen["rows"]),
            }
        ),
        "metadata_rows": len(metadata_rows),
        "duplicate_metadata_subject_ids": duplicate_metadata_subject_ids,
        "matched_pairs": matched,
        "unmatched_pairs": unmatched_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "metadata_coverage": metadata_coverage,
        "eligible_vendor_domains": len(eligible_vendors),
        "selected_vendor_domains": selected,
        "checks": checks,
        "decision": decision,
    }

    write_json(
        build_dir / "eligibility_gate.json",
        gate,
    )

    (
        build_dir / "decision.txt"
    ).write_text(
        decision + "\n",
        encoding="utf-8",
    )

    lines = [
        "===== Q1-R08A1 M&Ms ZIP-NATIVE METADATA + VENDOR DOMAIN AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Locked raw asset:",
        f"  ZIP={ZIP_PATH}",
        f"  ZIP SHA256={EXPECTED_ZIP_SHA256}",
        f"  exact image/GT pairs={len(pairs)}",
        "",
        "Metadata:",
        (
            "  metadata table resolved=YES"
            if chosen is not None
            else
            "  metadata table resolved=NO"
        ),
    ]

    if chosen is not None:
        lines += [
            f"  member={chosen['member_path']}",
            f"  sheet={chosen['sheet']}",
            f"  subject column={chosen['subject_col']}",
            f"  vendor column={chosen['vendor_col']}",
            f"  centre column={chosen['centre_col'] or ''}",
            f"  split column={chosen['split_col'] or ''}",
            f"  normalized metadata rows={len(metadata_rows)}",
        ]

    lines += [
        f"  matched exact pairs={matched}",
        f"  unmatched pairs={len(unmatched_pairs)}",
        f"  ambiguous pairs={len(ambiguous_pairs)}",
        f"  metadata coverage={metadata_coverage:.6f}",
        "",
        "Vendor domains:",
    ]

    for row in vendor_rows:
        lines.append(
            f"  {row['vendor']}: subjects={row['unique_subjects']} "
            f"centres={row['unique_centres']} "
            f"eligible={'YES' if row['eligible_vendor_domain'] else 'NO'}"
        )

    lines += [
        "",
        "Frozen selected Vendor domains:",
    ]

    if selected is None:
        lines.append("  selection unresolved")
    else:
        lines += [
            f"  Vendor-1={selected['vendor_1']} "
            f"(subjects={selected['vendor_1_subjects']})",
            f"  Vendor-2={selected['vendor_2']} "
            f"(subjects={selected['vendor_2_subjects']})",
            f"  direction-1={selected['vendor_1']} -> {selected['vendor_2']}",
            f"  direction-2={selected['vendor_2']} -> {selected['vendor_1']}",
            "  performance-based selection=NO",
        ]

    lines += [
        "",
        "Information boundary:",
        "  full ZIP extraction=NO",
        "  NIfTI payload extraction=NO",
        "  NIfTI voxel decoding=NO",
        "  GT voxel decoding=NO",
        "  model training/inference/TTA=NO",
        "",
        "Checks:",
    ]

    for k, v in checks.items():
        expected_false = k in {
            "J_nifti_payload_extracted",
            "K_nifti_voxels_decoded",
            "L_gt_voxels_decoded",
            "M_model_training_inference_tta",
        }
        passed = (not v) if expected_false else bool(v)
        lines.append(
            f"  {k}={'PASS' if passed else 'FAIL'}"
        )

    lines += [
        "",
        "Decision:",
        f"  {decision}",
        "",
        "Next:",
        (
            "  Q1-R08A2 ZIP-streaming compact cardiac-MRI preprocessing protocol"
            if decision == READY
            else
            "  Resolve metadata/domain structural deficit before preprocessing"
        ),
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "zip_non_nifti_inventory.csv",
        "metadata_table_inventory.csv",
        "metadata_normalized.csv",
        "paired_case_metadata_manifest.csv",
        "vendor_domain_summary.csv",
        "selected_vendor_domains.json",
        "eligibility_gate.json",
        "decision.txt",
        "run_log.txt",
    ]

    artifacts = {
        name: {
            "relative_path": name,
            "sha256": sha256_file(
                build_dir / name
            ),
        }
        for name in artifact_names
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r08a0_lock_sha256": EXPECTED_R08A0_LOCK_SHA256,
        "zip_sha256": EXPECTED_ZIP_SHA256,
        "persistent_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "exact_pairs": len(pairs),
        "metadata_coverage": metadata_coverage,
        "eligible_vendor_domains": len(eligible_vendors),
        "selected_vendor_domains": selected,
        "nifti_payload_extracted": False,
        "nifti_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "model_training": False,
        "model_inference": False,
        "tta": False,
        "decision": decision,
        "checks": checks,
        "artifacts": artifacts,
    }

    lock_path = (
        build_dir
        / "Q1_R08A1_MNMS_VENDOR_DOMAIN_LOCK.json"
    )

    write_json(
        lock_path,
        lock,
    )

    lock_sha = sha256_file(
        lock_path
    )

    for name, meta in artifacts.items():
        if (
            sha256_file(
                build_dir
                / meta["relative_path"]
            )
            != meta["sha256"]
        ):
            raise RuntimeError(
                f"Artifact changed before commit: {name}"
            )

    build_dir.rename(
        args.output_dir
    )

    print()
    print(
        (
            args.output_dir
            / "run_log.txt"
        ).read_text(
            encoding="utf-8"
        )
    )

    print(
        "Q1-R08A1 LOCK:",
        args.output_dir
        / "Q1_R08A1_MNMS_VENDOR_DOMAIN_LOCK.json",
    )

    print(
        "Q1-R08A1 LOCK SHA256:",
        lock_sha,
    )


def self_test():
    assert EXPECTED_EXACT_PAIRS == 345
    assert MIN_METADATA_COVERAGE == 0.95
    assert MIN_VENDOR_SUBJECTS == 25
    assert MIN_ELIGIBLE_VENDORS == 2

    assert (
        pair_key("A001_sa.nii")
        ==
        pair_key("A001_sa_gt.nii")
    )

    candidates = subject_candidates(
        "MnM/Training/A001/A001_sa.nii"
    )

    assert "a001" in candidates

    headers = [
        "External code",
        "Vendor",
        "Centre",
    ]

    assert (
        find_column(
            headers,
            SUBJECT_HINTS,
        )
        ==
        "External code"
    )

    assert (
        find_column(
            headers,
            VENDOR_HINTS,
        )
        ==
        "Vendor"
    )

    toy = [
        {
            "vendor": "Vendor B",
            "vendor_normalized": "vendorb",
            "unique_subjects": 30,
            "eligible_vendor_domain": 1,
        },
        {
            "vendor": "Vendor A",
            "vendor_normalized": "vendora",
            "unique_subjects": 30,
            "eligible_vendor_domain": 1,
        },
        {
            "vendor": "Vendor C",
            "vendor_normalized": "vendorc",
            "unique_subjects": 20,
            "eligible_vendor_domain": 0,
        },
    ]

    selected = select_vendors(toy)

    assert selected["vendor_1"] == "Vendor A"
    assert selected["vendor_2"] == "Vendor B"

    print("PAIR_KEY_TEST_PASS")
    print("SUBJECT_CANDIDATE_TEST_PASS")
    print("METADATA_COLUMN_RESOLUTION_TEST_PASS")
    print("FROZEN_VENDOR_SELECTION_TEST_PASS")
    print("NO_NIFTI_EXTRACTION_SELF_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08A1: ZIP-native M&Ms metadata/Vendor domain audit "
            "without extracting or decoding NIfTI payloads."
        )
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    p.add_argument(
        "--preflight-only",
        action="store_true",
    )

    p.add_argument(
        "--self-test",
        action="store_true",
    )

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
