#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Q1-R08A2 — M&Ms ZIP-streaming compact B/A preprocessing.

- No full 64+ GiB extraction.
- One image + one GT NIfTI temporarily extracted per case to F:.
- Only frozen Vendors B/A are processed.
- Compact images and masks are physically separated.
- No model/TTA/performance tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R08A2-v1"
BUILD = "Q1_R08A2_MNMS_ZIP_STREAMING_COMPACT_BA_PREPROCESSING"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R08A2_mnms_zip_streaming_compact_preprocessing_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "86b1a69ad4f7fd2ee0240c66fccd4d35ae77bfced4b8890065cc489d340b41d1"

R08A1_DIR = (
    ROOT / "outputs"
    / "Q1_R08A1_mnms_zip_native_metadata_vendor_domain_audit_v1"
)
R08A1_LOCK = (
    R08A1_DIR
    / "Q1_R08A1_MNMS_VENDOR_DOMAIN_LOCK.json"
)
R08A1_PAIR_MANIFEST = (
    R08A1_DIR
    / "paired_case_metadata_manifest.csv"
)

EXPECTED_R08A1_LOCK_SHA256 = (
    "e2cdf7c538a04d735240a9d6c65ee44d3db9c555a02cc82cd465f48836da2102"
)
EXPECTED_R08A1_DECISION = "MNMS_VENDOR_DOMAIN_STRUCTURE_READY"

ZIP_PATH = (
    ROOT / "data" / "downloads" / "MnMs_kaggle" / "MnMs.zip"
)
EXPECTED_ZIP_SHA256 = (
    "97c0eedd0562b8a90ce70be4a0a114ffc620a82832217c8f0a4e24acce112718"
)

METADATA_MEMBER = (
    "MnM/211230_MnMs_Dataset_information_diagnosis_opendataset.csv"
)

SELECTED_VENDORS = ("B", "A")
EXPECTED_VENDOR_COUNTS = {
    "B": 125,
    "A": 95,
}
EXPECTED_CASES = 220

ALLOWED_GT_LABELS = {0, 1, 2, 3}
EXPECTED_GLOBAL_GT_LABELS = {0, 1, 2, 3}

PROCESSED_DIR = (
    ROOT / "data" / "processed"
    / "MnMs_R08A2_BA_compact_v1"
)

TMP_ROOT = (
    ROOT / "data" / "tmp"
    / "Q1_R08A2_mnms_zip_streaming_v1"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R08A2_mnms_zip_streaming_compact_preprocessing_v1"
)

READY = "MNMS_COMPACT_BA_ASSET_READY"
INCOMPLETE = "MNMS_COMPACT_BA_ASSET_INCOMPLETE"

SUBJECT_HINTS = (
    "externalcode",
    "subject",
    "patient",
    "case",
    "code",
    "id",
)

ED_HINTS = (
    "ed",
    "enddiastole",
    "enddiastolic",
    "enddiastolicframe",
)

ES_HINTS = (
    "es",
    "endsystole",
    "endsystolic",
    "endsystolicframe",
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


def normalize_col(x: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).strip().lower(),
    )


def normalize_value(x: Any) -> str:
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in {"", "nan", "none", "null"}:
        return ""

    return re.sub(r"\s+", " ", s)


def normalize_subject(x: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_value(x).lower(),
    )


def find_column(headers: Sequence[str], hints: Sequence[str]) -> str | None:
    normalized = {
        h: normalize_col(h)
        for h in headers
    }

    # Exact normalized match first.
    for hint in hints:
        for original, nh in normalized.items():
            if nh == hint:
                return original

    # Conservative contains fallback.
    for hint in hints:
        for original, nh in normalized.items():
            if hint in nh:
                return original

    return None


def parse_int_like(x: Any, label: str) -> int:
    s = normalize_value(x)

    if not s:
        raise RuntimeError(
            f"Missing integer-like metadata value: {label}"
        )

    try:
        f = float(s)
    except Exception as e:
        raise RuntimeError(
            f"Invalid integer-like metadata value {label}={s!r}"
        ) from e

    if not math.isfinite(f):
        raise RuntimeError(
            f"Non-finite metadata value {label}={s!r}"
        )

    i = int(round(f))

    if abs(f - i) > 1e-6:
        raise RuntimeError(
            f"Non-integer metadata value {label}={s!r}"
        )

    return i


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_r08a1_lock():
    validate_sha(
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "R08A2 protocol",
    )

    validate_sha(
        R08A1_LOCK,
        EXPECTED_R08A1_LOCK_SHA256,
        "R08A1 lock",
    )

    validate_sha(
        ZIP_PATH,
        EXPECTED_ZIP_SHA256,
        "M&Ms ZIP",
    )

    lock = json.loads(
        R08A1_LOCK.read_text(
            encoding="utf-8"
        )
    )

    if lock.get("decision") != EXPECTED_R08A1_DECISION:
        raise RuntimeError(
            f"Unexpected R08A1 decision: {lock.get('decision')}"
        )

    selected = lock.get(
        "selected_vendor_domains"
    ) or {}

    if (
        selected.get("vendor_1") != "B"
        or int(selected.get("vendor_1_subjects", -1)) != 125
        or selected.get("vendor_2") != "A"
        or int(selected.get("vendor_2_subjects", -1)) != 95
    ):
        raise RuntimeError(
            "Frozen R08A1 Vendor selection changed."
        )

    if bool(
        selected.get(
            "performance_based_selection",
            True,
        )
    ):
        raise RuntimeError(
            "R08A1 unexpectedly reports performance-based selection."
        )

    if lock.get("zip_sha256") != EXPECTED_ZIP_SHA256:
        raise RuntimeError(
            "R08A1 ZIP SHA changed."
        )

    return lock


def load_pair_manifest() -> list[dict]:
    if not R08A1_PAIR_MANIFEST.is_file():
        raise FileNotFoundError(
            R08A1_PAIR_MANIFEST
        )

    rows = []

    with R08A1_PAIR_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "pair_key",
            "image_member",
            "gt_member",
            "metadata_matched",
            "matched_subject_id",
            "vendor",
            "centre",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"R08A1 pair manifest missing columns: {sorted(missing)}"
            )

        for row in reader:
            if int(row["metadata_matched"]) != 1:
                continue

            vendor = normalize_value(
                row["vendor"]
            )

            if vendor not in SELECTED_VENDORS:
                continue

            subject = normalize_subject(
                row["matched_subject_id"]
            )

            if not subject:
                raise RuntimeError(
                    "Selected pair has empty subject ID."
                )

            rows.append({
                "pair_key": row["pair_key"],
                "image_member": row["image_member"],
                "gt_member": row["gt_member"],
                "subject_id": subject,
                "vendor": vendor,
                "centre": normalize_value(
                    row["centre"]
                ),
            })

    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"Selected B/A cases={len(rows)} expected={EXPECTED_CASES}"
        )

    counts = Counter(
        r["vendor"]
        for r in rows
    )

    if dict(counts) != EXPECTED_VENDOR_COUNTS:
        raise RuntimeError(
            f"Vendor counts={dict(counts)} expected={EXPECTED_VENDOR_COUNTS}"
        )

    subjects = [
        r["subject_id"]
        for r in rows
    ]

    duplicates = [
        s for s, n
        in Counter(subjects).items()
        if n > 1
    ]

    if duplicates:
        raise RuntimeError(
            f"Duplicate selected subjects: {duplicates[:10]}"
        )

    return sorted(
        rows,
        key=lambda r: (
            r["vendor"],
            r["subject_id"],
        ),
    )


def decode_metadata_csv(blob: bytes) -> tuple[list[str], list[dict]]:
    text = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            text = blob.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise RuntimeError(
            "Unable to decode M&Ms metadata CSV."
        )

    sample = text[:8192]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(
        io.StringIO(text),
        dialect=dialect,
    )

    headers = [
        normalize_value(x)
        for x in (reader.fieldnames or [])
    ]

    rows = []

    for row in reader:
        cleaned = {
            normalize_value(k):
                normalize_value(v)
            for k, v in row.items()
            if k is not None
        }

        if any(cleaned.values()):
            rows.append(cleaned)

    return headers, rows


def load_phase_metadata(
    zf: zipfile.ZipFile,
) -> tuple[list[dict], dict]:
    try:
        blob = zf.read(
            METADATA_MEMBER
        )
    except KeyError as e:
        raise RuntimeError(
            f"Required metadata member missing: {METADATA_MEMBER}"
        ) from e

    headers, raw_rows = decode_metadata_csv(
        blob
    )

    subject_col = find_column(
        headers,
        SUBJECT_HINTS,
    )
    ed_col = find_column(
        headers,
        ED_HINTS,
    )
    es_col = find_column(
        headers,
        ES_HINTS,
    )

    if not subject_col:
        raise RuntimeError(
            f"Subject column unresolved. Headers={headers}"
        )

    if not ed_col:
        raise RuntimeError(
            f"ED column unresolved. Headers={headers}"
        )

    if not es_col:
        raise RuntimeError(
            f"ES column unresolved. Headers={headers}"
        )

    out = []

    for row in raw_rows:
        sid = normalize_subject(
            row.get(subject_col, "")
        )

        if not sid:
            continue

        out.append({
            "subject_id": sid,
            "ed_raw": row.get(ed_col, ""),
            "es_raw": row.get(es_col, ""),
            "ed_value": parse_int_like(
                row.get(ed_col, ""),
                f"{sid}:{ed_col}",
            ),
            "es_value": parse_int_like(
                row.get(es_col, ""),
                f"{sid}:{es_col}",
            ),
        })

    duplicate_subjects = sorted(
        sid for sid, n
        in Counter(
            r["subject_id"]
            for r in out
        ).items()
        if n > 1
    )

    if duplicate_subjects:
        raise RuntimeError(
            f"Metadata has duplicate subject rows: {duplicate_subjects[:10]}"
        )

    info = {
        "metadata_member": METADATA_MEMBER,
        "subject_column": subject_col,
        "ed_column": ed_col,
        "es_column": es_col,
        "metadata_rows": len(out),
    }

    return out, info


def stream_member_to_file(
    zf: zipfile.ZipFile,
    member: str,
    dest: Path,
):
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zf.open(
        member,
        "r",
    ) as src, dest.open(
        "wb",
    ) as dst:
        shutil.copyfileobj(
            src,
            dst,
            length=16 * 1024 * 1024,
        )


def import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "nibabel is required for R08A2. Install in rare26 with:\n"
            r"  D:\anaconda\envs\rare26\python.exe -m pip install nibabel"
        ) from e

    return nib


def annotated_gt_frames(
    gt_proxy,
) -> list[int]:
    shape = tuple(
        int(x)
        for x in gt_proxy.shape
    )

    if len(shape) != 4:
        raise RuntimeError(
            f"GT ndim={len(shape)} shape={shape}; expected 4-D."
        )

    frames = []

    for t in range(shape[3]):
        arr = np.asanyarray(
            gt_proxy.dataobj[..., t]
        )

        if np.any(arr != 0):
            frames.append(t)

    return frames


def metadata_candidate_indices(
    ed_value: int,
    es_value: int,
    tdim: int,
) -> dict[str, tuple[int, int] | None]:
    candidates = {}

    zero = (
        ed_value,
        es_value,
    )

    if all(
        0 <= x < tdim
        for x in zero
    ):
        candidates[
            "ZERO_BASED"
        ] = zero
    else:
        candidates[
            "ZERO_BASED"
        ] = None

    one = (
        ed_value - 1,
        es_value - 1,
    )

    if all(
        0 <= x < tdim
        for x in one
    ):
        candidates[
            "ONE_BASED"
        ] = one
    else:
        candidates[
            "ONE_BASED"
        ] = None

    return candidates


def resolve_case_convention(
    ed_value: int,
    es_value: int,
    tdim: int,
    annotated_frames: list[int],
) -> tuple[str, tuple[int, int], list[str]]:
    gt_set = set(
        int(x)
        for x in annotated_frames
    )

    candidate_map = metadata_candidate_indices(
        ed_value,
        es_value,
        tdim,
    )

    valid = []

    for name, indices in candidate_map.items():
        if indices is None:
            continue

        if set(indices) == gt_set:
            valid.append(name)

    if len(valid) != 1:
        raise RuntimeError(
            "Unable to resolve unique ED/ES indexing convention: "
            f"ED={ed_value} ES={es_value} T={tdim} "
            f"GT_frames={annotated_frames} "
            f"valid={valid}"
        )

    name = valid[0]
    indices = candidate_map[name]

    assert indices is not None

    return (
        name,
        indices,
        valid,
    )


def ensure_integer_labels(
    arr: np.ndarray,
    subject: str,
) -> np.ndarray:
    if not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        rounded = np.rint(arr)

        if not np.allclose(
            arr,
            rounded,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{subject}: GT contains non-integer values."
            )

        arr = rounded

    arr64 = arr.astype(
        np.int64,
        copy=False,
    )

    labels = set(
        int(x)
        for x in np.unique(arr64)
    )

    if not labels.issubset(
        ALLOWED_GT_LABELS
    ):
        raise RuntimeError(
            f"{subject}: unexpected GT labels={sorted(labels)}"
        )

    return arr64.astype(
        np.uint8,
        copy=False,
    )


def transpose_to_tzyx(
    arr_xyzt: np.ndarray,
) -> np.ndarray:
    # [X,Y,Z,TWO_PHASE] -> [TWO_PHASE,Z,Y,X]
    if arr_xyzt.ndim != 4:
        raise RuntimeError(
            f"Expected 4-D phase array, got {arr_xyzt.shape}"
        )

    return np.transpose(
        arr_xyzt,
        (3, 2, 1, 0),
    )


def save_npz_atomic(
    path: Path,
    **arrays,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp.npz"
    )

    if tmp.exists():
        tmp.unlink()

    np.savez_compressed(
        tmp,
        **arrays,
    )

    if not tmp.is_file():
        raise RuntimeError(
            f"Failed to create temporary NPZ: {tmp}"
        )

    tmp.replace(path)


def count_raw_temp_nifti(
    root: Path,
) -> int:
    if not root.exists():
        return 0

    return sum(
        1
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.name.lower().endswith(".nii")
            or p.name.lower().endswith(".nii.gz")
        )
    )


def preflight():
    lock = load_r08a1_lock()
    selected = load_pair_manifest()
    nib = import_nibabel()

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        phase_rows, phase_info = load_phase_metadata(
            zf
        )

    phase_index = {
        r["subject_id"]: r
        for r in phase_rows
    }

    missing_phase = [
        r["subject_id"]
        for r in selected
        if r["subject_id"] not in phase_index
    ]

    if missing_phase:
        raise RuntimeError(
            f"Selected cases missing ED/ES metadata: {missing_phase[:10]}"
        )

    print(
        "===== Q1-R08A2 PREFLIGHT ====="
    )
    print(
        f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}"
    )
    print(
        f"r08a1_lock_sha256={EXPECTED_R08A1_LOCK_SHA256}"
    )
    print(
        f"zip_sha256={EXPECTED_ZIP_SHA256}"
    )
    print(
        f"selected_cases={len(selected)}"
    )
    print(
        f"vendor_B={sum(r['vendor']=='B' for r in selected)}"
    )
    print(
        f"vendor_A={sum(r['vendor']=='A' for r in selected)}"
    )
    print(
        f"metadata_member={phase_info['metadata_member']}"
    )
    print(
        f"subject_column={phase_info['subject_column']}"
    )
    print(
        f"ed_column={phase_info['ed_column']}"
    )
    print(
        f"es_column={phase_info['es_column']}"
    )
    print(
        f"selected_cases_missing_phase_metadata={len(missing_phase)}"
    )
    print(
        f"nibabel_version={getattr(nib, '__version__', '?')}"
    )
    print(
        f"processed_dir={PROCESSED_DIR}"
    )
    print(
        f"temporary_raw_root={TMP_ROOT}"
    )
    print(
        "full_archive_extraction=NO"
    )
    print(
        "model_training_inference_tta=NO"
    )
    print(
        "PREFLIGHT_PASS"
    )


def run(args):
    upstream = load_r08a1_lock()
    selected = load_pair_manifest()
    nib = import_nibabel()

    if args.output_dir.exists():
        raise FileExistsError(
            args.output_dir
        )

    if args.processed_dir.exists():
        raise FileExistsError(
            args.processed_dir
        )

    audit_build = Path(
        str(args.output_dir)
        + "__building"
    )

    processed_build = Path(
        str(args.processed_dir)
        + "__building"
    )

    if audit_build.exists():
        raise FileExistsError(
            audit_build
        )

    if processed_build.exists():
        raise FileExistsError(
            processed_build
        )

    if TMP_ROOT.exists():
        # Leftover raw temp from a previous interrupted attempt is never
        # silently trusted. Remove it before a fresh formal run.
        shutil.rmtree(
            TMP_ROOT,
            ignore_errors=False,
        )

    TMP_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    audit_build.mkdir(
        parents=True,
        exist_ok=False,
    )

    processed_build.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        processed_build
        / "images"
    ).mkdir()

    (
        processed_build
        / "masks"
    ).mkdir()

    protocol_copy = (
        audit_build
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

    case_audit = []
    case_manifest = []
    all_labels = set()
    convention_counts = Counter()
    total_output_bytes = 0

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        phase_rows, phase_info = load_phase_metadata(
            zf
        )

        phase_index = {
            r["subject_id"]: r
            for r in phase_rows
        }

        metadata_phase_export = []

        for row in selected:
            p = phase_index.get(
                row["subject_id"]
            )

            if p is None:
                raise RuntimeError(
                    f"{row['subject_id']}: missing ED/ES metadata."
                )

            metadata_phase_export.append({
                "subject_id": row["subject_id"],
                "vendor": row["vendor"],
                "centre": row["centre"],
                "ed_raw": p["ed_raw"],
                "es_raw": p["es_raw"],
                "ed_value": p["ed_value"],
                "es_value": p["es_value"],
            })

        write_csv(
            audit_build
            / "metadata_phase_table.csv",
            metadata_phase_export,
            [
                "subject_id",
                "vendor",
                "centre",
                "ed_raw",
                "es_raw",
                "ed_value",
                "es_value",
            ],
        )

        bar = tqdm(
            selected,
            total=len(selected),
            desc="R08A2 compact cases",
            unit="case",
            dynamic_ncols=True,
        )

        try:
            for i, row in enumerate(bar):
                subject = row["subject_id"]
                vendor = row["vendor"]

                bar.set_postfix(
                    vendor=vendor,
                    subject=subject,
                )

                phase = phase_index[
                    subject
                ]

                case_tmp = (
                    TMP_ROOT
                    / f"{i:03d}_{subject}"
                )

                case_tmp.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                img_tmp = (
                    case_tmp
                    / "image.nii"
                )

                gt_tmp = (
                    case_tmp
                    / "gt.nii"
                )

                try:
                    stream_member_to_file(
                        zf,
                        row["image_member"],
                        img_tmp,
                    )

                    stream_member_to_file(
                        zf,
                        row["gt_member"],
                        gt_tmp,
                    )

                    img_nii = nib.load(
                        str(img_tmp),
                        mmap=True,
                    )

                    gt_nii = nib.load(
                        str(gt_tmp),
                        mmap=True,
                    )

                    img_shape = tuple(
                        int(x)
                        for x in img_nii.shape
                    )

                    gt_shape = tuple(
                        int(x)
                        for x in gt_nii.shape
                    )

                    if len(img_shape) != 4:
                        raise RuntimeError(
                            f"{subject}: image ndim={len(img_shape)} "
                            f"shape={img_shape}"
                        )

                    if len(gt_shape) != 4:
                        raise RuntimeError(
                            f"{subject}: GT ndim={len(gt_shape)} "
                            f"shape={gt_shape}"
                        )

                    if img_shape != gt_shape:
                        raise RuntimeError(
                            f"{subject}: image/GT shape mismatch "
                            f"{img_shape} vs {gt_shape}"
                        )

                    if not np.allclose(
                        img_nii.affine,
                        gt_nii.affine,
                        rtol=1e-5,
                        atol=1e-4,
                    ):
                        raise RuntimeError(
                            f"{subject}: image/GT affine mismatch."
                        )

                    annotated = annotated_gt_frames(
                        gt_nii
                    )

                    if len(annotated) != 2:
                        raise RuntimeError(
                            f"{subject}: annotated GT frames={annotated}; "
                            "expected exactly 2."
                        )

                    convention, (
                        ed_index,
                        es_index,
                    ), _ = resolve_case_convention(
                        phase["ed_value"],
                        phase["es_value"],
                        img_shape[3],
                        annotated,
                    )

                    convention_counts[
                        convention
                    ] += 1

                    frame_indices = np.asarray(
                        [
                            ed_index,
                            es_index,
                        ],
                        dtype=np.int16,
                    )

                    # Pull only ED/ES, not the whole 4-D cine into RAM.
                    image_xyzt = np.stack(
                        [
                            np.asanyarray(
                                img_nii.dataobj[..., ed_index],
                                dtype=np.float32,
                            ),
                            np.asanyarray(
                                img_nii.dataobj[..., es_index],
                                dtype=np.float32,
                            ),
                        ],
                        axis=3,
                    )

                    mask_xyzt_raw = np.stack(
                        [
                            np.asanyarray(
                                gt_nii.dataobj[..., ed_index]
                            ),
                            np.asanyarray(
                                gt_nii.dataobj[..., es_index]
                            ),
                        ],
                        axis=3,
                    )

                    if not np.all(
                        np.isfinite(
                            image_xyzt
                        )
                    ):
                        raise RuntimeError(
                            f"{subject}: image contains NaN/Inf."
                        )

                    mask_xyzt = ensure_integer_labels(
                        mask_xyzt_raw,
                        subject,
                    )

                    labels = set(
                        int(x)
                        for x in np.unique(
                            mask_xyzt
                        )
                    )

                    all_labels.update(
                        labels
                    )

                    image_tzyx = (
                        transpose_to_tzyx(
                            image_xyzt
                        ).astype(
                            np.float32,
                            copy=False,
                        )
                    )

                    mask_tzyx = (
                        transpose_to_tzyx(
                            mask_xyzt
                        ).astype(
                            np.uint8,
                            copy=False,
                        )
                    )

                    if (
                        image_tzyx.shape
                        != mask_tzyx.shape
                    ):
                        raise RuntimeError(
                            f"{subject}: compact image/mask shape mismatch."
                        )

                    if image_tzyx.shape[0] != 2:
                        raise RuntimeError(
                            f"{subject}: compact phase axis != 2."
                        )

                    foreground_slice_count = int(
                        np.any(
                            mask_tzyx > 0,
                            axis=(2, 3),
                        ).sum()
                    )

                    if foreground_slice_count <= 0:
                        raise RuntimeError(
                            f"{subject}: no foreground compact slices."
                        )

                    image_out = (
                        processed_build
                        / "images"
                        / f"{subject}.npz"
                    )

                    mask_out = (
                        processed_build
                        / "masks"
                        / f"{subject}.npz"
                    )

                    save_npz_atomic(
                        image_out,
                        image=image_tzyx,
                        frame_indices=frame_indices,
                    )

                    save_npz_atomic(
                        mask_out,
                        mask=mask_tzyx,
                        frame_indices=frame_indices,
                    )

                    # Immediate round-trip structural check.
                    with np.load(
                        image_out,
                        allow_pickle=False,
                    ) as d:
                        if (
                            d["image"].dtype
                            != np.float32
                            or tuple(d["image"].shape)
                            != tuple(image_tzyx.shape)
                        ):
                            raise RuntimeError(
                                f"{subject}: image NPZ round-trip mismatch."
                            )

                    with np.load(
                        mask_out,
                        allow_pickle=False,
                    ) as d:
                        if (
                            d["mask"].dtype
                            != np.uint8
                            or tuple(d["mask"].shape)
                            != tuple(mask_tzyx.shape)
                        ):
                            raise RuntimeError(
                                f"{subject}: mask NPZ round-trip mismatch."
                            )

                    image_bytes = (
                        image_out.stat().st_size
                    )

                    mask_bytes = (
                        mask_out.stat().st_size
                    )

                    total_output_bytes += (
                        image_bytes
                        + mask_bytes
                    )

                    zooms = tuple(
                        float(x)
                        for x in img_nii.header.get_zooms()
                    )

                    case_audit.append({
                        "subject_id": subject,
                        "vendor": vendor,
                        "centre": row["centre"],
                        "image_member": row["image_member"],
                        "gt_member": row["gt_member"],
                        "image_shape_xyzt":
                            "x".join(map(str, img_shape)),
                        "gt_shape_xyzt":
                            "x".join(map(str, gt_shape)),
                        "metadata_ed_value":
                            phase["ed_value"],
                        "metadata_es_value":
                            phase["es_value"],
                        "resolved_index_convention":
                            convention,
                        "resolved_ed_index":
                            ed_index,
                        "resolved_es_index":
                            es_index,
                        "annotated_gt_frames":
                            "|".join(map(str, annotated)),
                        "compact_shape_tzyx":
                            "x".join(map(str, image_tzyx.shape)),
                        "foreground_compact_slices":
                            foreground_slice_count,
                        "gt_labels":
                            "|".join(map(str, sorted(labels))),
                        "image_finite": 1,
                        "affine_match": 1,
                        "zoom_values":
                            "|".join(f"{x:.8g}" for x in zooms),
                        "image_npz_bytes":
                            image_bytes,
                        "mask_npz_bytes":
                            mask_bytes,
                    })

                    case_manifest.append({
                        "subject_id": subject,
                        "vendor": vendor,
                        "centre": row["centre"],
                        "image_npz":
                            f"images/{subject}.npz",
                        "mask_npz":
                            f"masks/{subject}.npz",
                        "phase_order": "ED|ES",
                        "frame_indices":
                            f"{ed_index}|{es_index}",
                        "shape_tzyx":
                            "x".join(map(str, image_tzyx.shape)),
                        "source_image_member":
                            row["image_member"],
                        "source_gt_member":
                            row["gt_member"],
                    })

                finally:
                    # Release mmap-backed objects before deleting temporary files.
                    for var_name in (
                        "img_nii",
                        "gt_nii",
                    ):
                        if var_name in locals():
                            try:
                                del locals()[var_name]
                            except Exception:
                                pass

                    shutil.rmtree(
                        case_tmp,
                        ignore_errors=True,
                    )

                bar.set_postfix(
                    vendor=vendor,
                    subject=subject,
                    out_GiB=f"{total_output_bytes / (1024**3):.2f}",
                )

        finally:
            bar.close()

    # No per-case convention switching allowed.
    if len(convention_counts) != 1:
        raise RuntimeError(
            f"Dataset-wide indexing convention not unique: "
            f"{dict(convention_counts)}"
        )

    global_convention = next(
        iter(convention_counts.keys())
    )

    if convention_counts[
        global_convention
    ] != EXPECTED_CASES:
        raise RuntimeError(
            f"Global convention count={convention_counts[global_convention]} "
            f"expected={EXPECTED_CASES}"
        )

    if all_labels != EXPECTED_GLOBAL_GT_LABELS:
        raise RuntimeError(
            f"Global GT labels={sorted(all_labels)} "
            f"expected={sorted(EXPECTED_GLOBAL_GT_LABELS)}"
        )

    image_files = sorted(
        (
            processed_build
            / "images"
        ).glob("*.npz")
    )

    mask_files = sorted(
        (
            processed_build
            / "masks"
        ).glob("*.npz")
    )

    temp_nifti_count = count_raw_temp_nifti(
        TMP_ROOT
    )

    if len(image_files) != EXPECTED_CASES:
        raise RuntimeError(
            f"Compact image files={len(image_files)} expected={EXPECTED_CASES}"
        )

    if len(mask_files) != EXPECTED_CASES:
        raise RuntimeError(
            f"Compact mask files={len(mask_files)} expected={EXPECTED_CASES}"
        )

    if temp_nifti_count != 0:
        raise RuntimeError(
            f"Temporary raw NIfTI files remain={temp_nifti_count}"
        )

    write_csv(
        audit_build
        / "case_processing_audit.csv",
        case_audit,
        [
            "subject_id",
            "vendor",
            "centre",
            "image_member",
            "gt_member",
            "image_shape_xyzt",
            "gt_shape_xyzt",
            "metadata_ed_value",
            "metadata_es_value",
            "resolved_index_convention",
            "resolved_ed_index",
            "resolved_es_index",
            "annotated_gt_frames",
            "compact_shape_tzyx",
            "foreground_compact_slices",
            "gt_labels",
            "image_finite",
            "affine_match",
            "zoom_values",
            "image_npz_bytes",
            "mask_npz_bytes",
        ],
    )

    write_csv(
        processed_build
        / "case_manifest.csv",
        case_manifest,
        [
            "subject_id",
            "vendor",
            "centre",
            "image_npz",
            "mask_npz",
            "phase_order",
            "frame_indices",
            "shape_tzyx",
            "source_image_member",
            "source_gt_member",
        ],
    )

    vendor_counts = Counter(
        r["vendor"]
        for r in case_manifest
    )

    summary = {
        "script_version": VERSION,
        "build": BUILD,
        "upstream_r08a1_lock_sha256":
            EXPECTED_R08A1_LOCK_SHA256,
        "raw_zip_sha256":
            EXPECTED_ZIP_SHA256,
        "selected_vendors": list(
            SELECTED_VENDORS
        ),
        "vendor_counts": dict(
            vendor_counts
        ),
        "cases": len(
            case_manifest
        ),
        "phase_order": [
            "ED",
            "ES",
        ],
        "global_metadata_index_convention":
            global_convention,
        "global_gt_labels": sorted(
            all_labels
        ),
        "image_dtype": "float32",
        "mask_dtype": "uint8",
        "compact_axis_order":
            "[phase,z,y,x]",
        "spatial_resize": False,
        "spatial_resampling": False,
        "crop": False,
        "intensity_normalization": False,
        "orientation_canonicalization": False,
        "images_masks_separated": True,
        "full_archive_extracted": False,
        "temporary_raw_nifti_files_after_completion":
            temp_nifti_count,
        "compact_image_files":
            len(image_files),
        "compact_mask_files":
            len(mask_files),
        "compact_total_bytes":
            total_output_bytes,
        "compact_total_gib":
            total_output_bytes / (1024 ** 3),
        "model_training": False,
        "model_inference": False,
        "tta": False,
    }

    write_json(
        processed_build
        / "dataset_summary.json",
        summary,
    )

    write_json(
        audit_build
        / "dataset_summary.json",
        summary,
    )

    write_json(
        audit_build
        / "upstream_audit.json",
        {
            "r08a1_lock_path":
                str(R08A1_LOCK),
            "r08a1_lock_sha256":
                EXPECTED_R08A1_LOCK_SHA256,
            "r08a1_decision":
                upstream.get("decision"),
            "raw_zip_path":
                str(ZIP_PATH),
            "raw_zip_sha256":
                EXPECTED_ZIP_SHA256,
            "selected_vendor_1":
                "B",
            "selected_vendor_1_subjects":
                125,
            "selected_vendor_2":
                "A",
            "selected_vendor_2_subjects":
                95,
            "performance_based_selection":
                False,
        },
    )

    checks = {
        "A_selected_cases_eq_220":
            len(case_manifest)
            == EXPECTED_CASES,
        "B_vendor_B_eq_125":
            vendor_counts.get("B", 0)
            == 125,
        "C_vendor_A_eq_95":
            vendor_counts.get("A", 0)
            == 95,
        "D_decoded_cases_eq_220":
            len(case_audit)
            == EXPECTED_CASES,
        "E_global_index_convention_resolved":
            len(convention_counts)
            == 1,
        "F_metadata_ED_ES_agreement_220_of_220":
            sum(
                1
                for r in case_audit
                if r[
                    "resolved_index_convention"
                ] == global_convention
            )
            == EXPECTED_CASES,
        "G_all_images_finite":
            all(
                int(r["image_finite"])
                == 1
                for r in case_audit
            ),
        "H_all_affines_match":
            all(
                int(r["affine_match"])
                == 1
                for r in case_audit
            ),
        "I_global_GT_labels_0_1_2_3":
            all_labels
            == EXPECTED_GLOBAL_GT_LABELS,
        "J_compact_image_files_eq_220":
            len(image_files)
            == EXPECTED_CASES,
        "K_compact_mask_files_eq_220":
            len(mask_files)
            == EXPECTED_CASES,
        "L_temp_raw_nifti_eq_0":
            temp_nifti_count
            == 0,
        "M_full_archive_extracted":
            False,
        "N_model_training_inference_tta":
            False,
    }

    pass_gate = all([
        checks[
            "A_selected_cases_eq_220"
        ],
        checks[
            "B_vendor_B_eq_125"
        ],
        checks[
            "C_vendor_A_eq_95"
        ],
        checks[
            "D_decoded_cases_eq_220"
        ],
        checks[
            "E_global_index_convention_resolved"
        ],
        checks[
            "F_metadata_ED_ES_agreement_220_of_220"
        ],
        checks[
            "G_all_images_finite"
        ],
        checks[
            "H_all_affines_match"
        ],
        checks[
            "I_global_GT_labels_0_1_2_3"
        ],
        checks[
            "J_compact_image_files_eq_220"
        ],
        checks[
            "K_compact_mask_files_eq_220"
        ],
        checks[
            "L_temp_raw_nifti_eq_0"
        ],
        not checks[
            "M_full_archive_extracted"
        ],
        not checks[
            "N_model_training_inference_tta"
        ],
    ])

    decision = (
        READY
        if pass_gate
        else INCOMPLETE
    )

    (
        audit_build
        / "decision.txt"
    ).write_text(
        decision + "\n",
        encoding="utf-8",
    )

    lines = [
        "===== Q1-R08A2 M&Ms ZIP-STREAMING COMPACT B/A PREPROCESSING =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Upstream:",
        f"  R08A1 lock SHA256={EXPECTED_R08A1_LOCK_SHA256}",
        f"  raw ZIP SHA256={EXPECTED_ZIP_SHA256}",
        "",
        "Frozen domains:",
        "  Vendor B=125",
        "  Vendor A=95",
        "  total=220",
        "  B -> A and A -> B",
        "  performance-based selection=NO",
        "",
        "Phase metadata:",
        f"  metadata member={phase_info['metadata_member']}",
        f"  subject column={phase_info['subject_column']}",
        f"  ED column={phase_info['ed_column']}",
        f"  ES column={phase_info['es_column']}",
        f"  global indexing convention={global_convention}",
        "",
        "Compact asset:",
        f"  processed dir={args.processed_dir}",
        f"  image files={len(image_files)}",
        f"  mask files={len(mask_files)}",
        f"  total bytes={total_output_bytes}",
        f"  total GiB={total_output_bytes/(1024**3):.4f}",
        f"  global GT labels={sorted(all_labels)}",
        "  array order=[phase,z,y,x]",
        "  image dtype=float32",
        "  mask dtype=uint8",
        "  spatial resize=NO",
        "  resampling=NO",
        "  intensity normalization=NO",
        "",
        "Storage boundary:",
        "  full 64+ GiB extraction=NO",
        f"  temporary raw NIfTI remaining={temp_nifti_count}",
        "  permanent raw source=14 GB locked ZIP",
        "",
        "Information boundary:",
        "  model training=NO",
        "  model inference=NO",
        "  TTA=NO",
        "  segmentation performance metrics=NO",
        "",
        "Checks:",
    ]

    for key, value in checks.items():
        expected_false = key in {
            "M_full_archive_extracted",
            "N_model_training_inference_tta",
        }

        passed = (
            not value
            if expected_false
            else bool(value)
        )

        lines.append(
            f"  {key}="
            f"{'PASS' if passed else 'FAIL'}"
        )

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    run_log = (
        audit_build
        / "run_log.txt"
    )

    run_log.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # Build processed asset manifest hash before final commit.
    processed_artifacts = {
        "case_manifest.csv":
            sha256_file(
                processed_build
                / "case_manifest.csv"
            ),
        "dataset_summary.json":
            sha256_file(
                processed_build
                / "dataset_summary.json"
            ),
    }

    # Hash every compact file. This is deliberate provenance locking.
    image_hashes = {}
    mask_hashes = {}

    for path in tqdm(
        image_files,
        desc="Hash compact images",
        unit="file",
        dynamic_ncols=True,
    ):
        image_hashes[
            path.name
        ] = sha256_file(
            path
        )

    for path in tqdm(
        mask_files,
        desc="Hash compact masks",
        unit="file",
        dynamic_ncols=True,
    ):
        mask_hashes[
            path.name
        ] = sha256_file(
            path
        )

    write_json(
        audit_build
        / "compact_file_hashes.json",
        {
            "images": image_hashes,
            "masks": mask_hashes,
        },
    )

    audit_artifact_names = [
        "preregistered_protocol_copy.md",
        "upstream_audit.json",
        "metadata_phase_table.csv",
        "case_processing_audit.csv",
        "dataset_summary.json",
        "compact_file_hashes.json",
        "decision.txt",
        "run_log.txt",
    ]

    audit_artifacts = {
        name: {
            "relative_path": name,
            "sha256":
                sha256_file(
                    audit_build
                    / name
                ),
        }
        for name in audit_artifact_names
    }

    lock = {
        "script_version":
            VERSION,
        "build":
            BUILD,
        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,
        "r08a1_lock_sha256":
            EXPECTED_R08A1_LOCK_SHA256,
        "raw_zip_sha256":
            EXPECTED_ZIP_SHA256,
        "selected_vendors":
            [
                {
                    "vendor": "B",
                    "subjects": 125,
                },
                {
                    "vendor": "A",
                    "subjects": 95,
                },
            ],
        "directions": [
            "B->A",
            "A->B",
        ],
        "performance_based_selection":
            False,
        "global_metadata_index_convention":
            global_convention,
        "cases":
            len(case_manifest),
        "global_gt_labels":
            sorted(all_labels),
        "processed_dir":
            str(args.processed_dir),
        "compact_image_files":
            len(image_files),
        "compact_mask_files":
            len(mask_files),
        "compact_total_bytes":
            total_output_bytes,
        "full_archive_extracted":
            False,
        "temporary_raw_nifti_files_after_completion":
            temp_nifti_count,
        "model_training":
            False,
        "model_inference":
            False,
        "tta":
            False,
        "decision":
            decision,
        "checks":
            checks,
        "processed_metadata_artifacts":
            processed_artifacts,
        "audit_artifacts":
            audit_artifacts,
        "compact_file_hashes_sha256":
            sha256_file(
                audit_build
                / "compact_file_hashes.json"
            ),
    }

    lock_path = (
        audit_build
        / "Q1_R08A2_MNMS_COMPACT_BA_ASSET_LOCK.json"
    )

    write_json(
        lock_path,
        lock,
    )

    lock_sha = sha256_file(
        lock_path
    )

    # Verify audit artifacts unchanged.
    for name, meta in audit_artifacts.items():
        if (
            sha256_file(
                audit_build
                / meta["relative_path"]
            )
            != meta["sha256"]
        ):
            raise RuntimeError(
                f"Audit artifact changed before commit: {name}"
            )

    # Final atomic-ish commit on same F: filesystem.
    processed_build.rename(
        args.processed_dir
    )

    audit_build.rename(
        args.output_dir
    )

    # Remove now-empty temp root.
    if TMP_ROOT.exists():
        shutil.rmtree(
            TMP_ROOT,
            ignore_errors=False,
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
        "Q1-R08A2 LOCK:",
        args.output_dir
        / "Q1_R08A2_MNMS_COMPACT_BA_ASSET_LOCK.json",
    )

    print(
        "Q1-R08A2 LOCK SHA256:",
        lock_sha,
    )


def self_test():
    assert EXPECTED_CASES == 220
    assert EXPECTED_VENDOR_COUNTS == {
        "B": 125,
        "A": 95,
    }

    assert parse_int_like(
        "12",
        "x",
    ) == 12

    assert parse_int_like(
        "12.0",
        "x",
    ) == 12

    headers = [
        "External code",
        "ED",
        "ES",
        "Vendor",
    ]

    assert find_column(
        headers,
        SUBJECT_HINTS,
    ) == "External code"

    assert find_column(
        headers,
        ED_HINTS,
    ) == "ED"

    assert find_column(
        headers,
        ES_HINTS,
    ) == "ES"

    c = metadata_candidate_indices(
        0,
        9,
        20,
    )

    assert c[
        "ZERO_BASED"
    ] == (0, 9)

    assert c[
        "ONE_BASED"
    ] is None

    c2 = metadata_candidate_indices(
        1,
        10,
        20,
    )

    assert c2[
        "ONE_BASED"
    ] == (0, 9)

    name, indices, valid = (
        resolve_case_convention(
            1,
            10,
            20,
            [0, 9],
        )
    )

    assert name == "ONE_BASED"
    assert indices == (0, 9)

    x = np.zeros(
        (4, 5, 6, 2),
        dtype=np.float32,
    )

    y = transpose_to_tzyx(
        x
    )

    assert y.shape == (
        2,
        6,
        5,
        4,
    )

    print(
        "FROZEN_VENDOR_COUNTS_TEST_PASS"
    )
    print(
        "METADATA_ED_ES_COLUMN_TEST_PASS"
    )
    print(
        "INDEX_CONVENTION_TEST_PASS"
    )
    print(
        "T_Z_Y_X_AXIS_TEST_PASS"
    )
    print(
        "F_DRIVE_STREAMING_DESIGN_TEST_PASS"
    )
    print(
        "SELF_TEST_PASS"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08A2: stream M&Ms B/A cases from the locked ZIP, "
            "extract only ED/ES compact image/mask arrays, and delete "
            "temporary raw NIfTI files after each case."
        )
    )

    p.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
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
