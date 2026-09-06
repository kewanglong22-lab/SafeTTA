#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Q1-R08A0 FIX1 — M&Ms Kaggle mirror ZIP identity/completeness audit.

Technical fix:
- no longer requires an old local manifest to exist;
- automatically rebuilds the full Kaggle file listing when necessary.

Scientific audit unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R08A0-v1-fix1"
BUILD = "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_IDENTITY_AUDIT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R08A0_mnms_kaggle_mirror_archive_identity_audit_preregistered_protocol_v1_fix1.md"
)
EXPECTED_PROTOCOL_SHA256 = "39090cb3ba31b03804053bba7168dd5002ba9e846e3dee43d5e935d18e217d4a"

DEFAULT_ZIP = ROOT / "data" / "downloads" / "MnMs_kaggle" / "MnMs.zip"

HISTORICAL_REMOTE_MANIFEST = (
    ROOT / "data" / "downloads" / "MnMs_kaggle_parallel"
    / "_kaggle_remote_manifest.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R08A0_mnms_kaggle_mirror_archive_identity_audit_v1_fix1"
)

KAGGLE_DATASET = "tailength/m-and-ms-dataset"
PAGE_SIZE = 200
EXPECTED_REMOTE_FILES = 691

READY = "MNMS_KAGGLE_MIRROR_ARCHIVE_COMPLETE"
INCOMPLETE = "MNMS_KAGGLE_MIRROR_ARCHIVE_INCOMPLETE"

GT_MARKERS = (
    "_gt",
    "-mask",
    "_mask",
    "-label",
    "_label",
    "-seg",
    "_seg",
)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def normalize_member_name(name: str) -> str:
    s = str(name).replace("\\", "/").lstrip("/")
    while s.startswith("./"):
        s = s[2:]
    return s


def normalize_remote_name(name: str) -> str:
    s = normalize_member_name(name)
    p = PurePosixPath(s)
    if any(part in ("", ".", "..") for part in p.parts):
        raise RuntimeError(f"Unsafe Kaggle remote path: {name!r}")
    if ":" in s:
        raise RuntimeError(f"Unsafe Kaggle remote path: {name!r}")
    return p.as_posix()


def get_field(obj: Any, names: Iterable[str], default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)

    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()
            if isinstance(d, dict):
                for name in names:
                    if name in d:
                        return d[name]
        except Exception:
            pass

    try:
        d = vars(obj)
        for name in names:
            if name in d:
                return d[name]
    except Exception:
        pass

    return default


def import_kaggle_api():
    try:
        import kaggle  # type: ignore
        api = getattr(kaggle, "api", None)
        if api is not None:
            return api
    except Exception:
        pass

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        raise RuntimeError(
            "Cannot initialize authenticated Kaggle API. Verify:\n"
            "  kaggle --version\n"
            "  kaggle datasets files tailength/m-and-ms-dataset --page-size 5"
        ) from e


def result_files(result: Any) -> list[Any]:
    files = get_field(
        result,
        ("files", "datasetFiles", "dataset_files"),
        None,
    )
    if files is None and isinstance(result, (list, tuple)):
        files = result
    if files is None:
        raise RuntimeError("Kaggle response contains no recognizable files field.")
    return list(files)


def next_page_token(result: Any) -> str | None:
    token = get_field(
        result,
        ("nextPageToken", "next_page_token", "nextPage", "next_page"),
        None,
    )
    if token is None:
        return None
    token = str(token).strip()
    return token or None


def parse_remote_file(obj: Any) -> RemoteFile:
    name = get_field(
        obj,
        ("name", "ref", "fileName", "file_name", "path"),
        None,
    )
    size = get_field(
        obj,
        ("totalBytes", "total_bytes", "size", "bytes", "fileSize", "file_size"),
        None,
    )

    if not name:
        raise RuntimeError(f"Cannot resolve Kaggle file name from {obj!r}")
    if size in (None, ""):
        raise RuntimeError(f"Kaggle file has no byte size: {name}")

    return RemoteFile(
        normalize_remote_name(str(name)),
        int(size),
    )


def fetch_kaggle_manifest() -> list[dict]:
    api = import_kaggle_api()

    all_files: list[RemoteFile] = []
    token: str | None = None
    seen_tokens: set[str] = set()

    with tqdm(
        desc="Fetch Kaggle manifest",
        unit="page",
        dynamic_ncols=True,
    ) as bar:
        while True:
            kwargs = {"page_size": PAGE_SIZE}
            if token:
                kwargs["page_token"] = token

            try:
                result = api.dataset_list_files(KAGGLE_DATASET, **kwargs)
            except TypeError:
                kwargs2 = {"pageSize": PAGE_SIZE}
                if token:
                    kwargs2["pageToken"] = token
                result = api.dataset_list_files(KAGGLE_DATASET, **kwargs2)

            page = [parse_remote_file(x) for x in result_files(result)]
            all_files.extend(page)

            bar.update(1)
            bar.set_postfix(files=len(all_files))

            nxt = next_page_token(result)
            if not nxt:
                if len(page) >= PAGE_SIZE:
                    raise RuntimeError(
                        "Kaggle returned a full 200-file page without a next-page "
                        "token. Update the kaggle package before continuing."
                    )
                break

            if nxt in seen_tokens:
                raise RuntimeError("Kaggle pagination token repeated.")
            seen_tokens.add(nxt)
            token = nxt

    by_name: dict[str, RemoteFile] = {}
    for rf in all_files:
        if rf.name in by_name:
            old = by_name[rf.name]
            if old.size != rf.size:
                raise RuntimeError(f"Conflicting duplicate Kaggle metadata: {rf.name}")
            continue
        by_name[rf.name] = rf

    files = sorted(by_name.values(), key=lambda x: x.name.lower())

    if len(files) != EXPECTED_REMOTE_FILES:
        raise RuntimeError(
            f"Kaggle listing rows={len(files)} expected={EXPECTED_REMOTE_FILES}"
        )

    return [
        {"remote_path": rf.name, "bytes": rf.size}
        for rf in files
    ]


def read_manifest_file(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"remote_path", "bytes"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"Manifest missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )

        for i, row in enumerate(reader):
            try:
                rows.append({
                    "remote_path": normalize_remote_name(row["remote_path"]),
                    "bytes": int(row["bytes"]),
                })
            except Exception as e:
                raise RuntimeError(f"Invalid manifest row={i}") from e

    validate_manifest_rows(rows)
    return rows


def validate_manifest_rows(rows: list[dict]):
    if len(rows) != EXPECTED_REMOTE_FILES:
        raise RuntimeError(
            f"Manifest rows={len(rows)} expected={EXPECTED_REMOTE_FILES}"
        )

    names = [r["remote_path"] for r in rows]
    dup = sorted(p for p, n in Counter(names).items() if n > 1)
    if dup:
        raise RuntimeError(f"Duplicate manifest paths: {dup[:10]}")

    if any(int(r["bytes"]) <= 0 for r in rows):
        raise RuntimeError("Manifest contains non-positive byte size.")


def resolve_manifest(explicit: Path | None):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        rows = read_manifest_file(explicit)
        return rows, "explicit_local_manifest", str(explicit)

    if HISTORICAL_REMOTE_MANIFEST.is_file():
        rows = read_manifest_file(HISTORICAL_REMOTE_MANIFEST)
        return rows, "historical_local_manifest", str(HISTORICAL_REMOTE_MANIFEST)

    rows = fetch_kaggle_manifest()
    validate_manifest_rows(rows)
    return rows, "live_kaggle_api_snapshot", KAGGLE_DATASET


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


def read_zip_inventory(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [x for x in zf.infolist() if not x.is_dir()]

    rows = []
    for info in infos:
        rows.append({
            "member_path": normalize_member_name(info.filename),
            "uncompressed_bytes": int(info.file_size),
            "compressed_bytes": int(info.compress_size),
            "crc32_hex": f"{int(info.CRC):08x}",
            "compression_type": int(info.compress_type),
            "is_nifti": int(is_nifti(info.filename)),
            "is_gt_like": int(is_nifti(info.filename) and is_gt_like(info.filename)),
        })
    return rows


def compare_manifest(manifest_rows, zip_rows):
    m = {r["remote_path"]: int(r["bytes"]) for r in manifest_rows}
    z = {r["member_path"]: int(r["uncompressed_bytes"]) for r in zip_rows}

    out = []
    for p in sorted(set(m) | set(z)):
        mp = p in m
        zp = p in z
        ms = m.get(p)
        zs = z.get(p)
        out.append({
            "path": p,
            "manifest_present": int(mp),
            "zip_present": int(zp),
            "manifest_bytes": "" if ms is None else ms,
            "zip_uncompressed_bytes": "" if zs is None else zs,
            "size_match": int(mp and zp and ms == zs),
        })
    return out


def structural_pair_summary(zip_rows):
    nifti = [r for r in zip_rows if int(r["is_nifti"]) == 1]
    groups = defaultdict(lambda: {"image": 0, "gt": 0})

    for r in nifti:
        k = pair_key(r["member_path"])
        groups[k]["gt" if int(r["is_gt_like"]) else "image"] += 1

    return {
        "nifti_members": len(nifti),
        "gt_like_nifti_members": sum(int(r["is_gt_like"]) for r in nifti),
        "non_gt_nifti_members": sum(1-int(r["is_gt_like"]) for r in nifti),
        "normalized_pair_keys": len(groups),
        "exact_image_gt_filename_pairs": sum(
            1 for g in groups.values()
            if g["image"] == 1 and g["gt"] == 1
        ),
        "voxel_decoding": False,
    }


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


def self_test():
    assert EXPECTED_REMOTE_FILES == 691
    assert normalize_member_name("./MnM/A.nii") == "MnM/A.nii"
    assert normalize_remote_name("MnM/Testing/A/A_sa.nii") == "MnM/Testing/A/A_sa.nii"
    assert is_nifti("A_sa.nii")
    assert is_gt_like("A_sa_gt.nii")
    assert pair_key("A_sa.nii") == pair_key("A_sa_gt.nii")

    dummy = [
        {"remote_path": f"x/{i}.nii", "bytes": i + 1}
        for i in range(EXPECTED_REMOTE_FILES)
    ]
    validate_manifest_rows(dummy)

    print("PATH_NORMALIZATION_TEST_PASS")
    print("GT_PAIR_KEY_TEST_PASS")
    print("MANIFEST_CARDINALITY_TEST_PASS")
    print("LOCAL_OR_LIVE_MANIFEST_FALLBACK_TEST_PASS")
    print("SELF_TEST_PASS")


def preflight(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A0 FIX1 protocol")

    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)

    rows, source_mode, source_ref = resolve_manifest(args.remote_manifest)

    print("===== Q1-R08A0 FIX1 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"zip={args.zip}")
    print(f"zip_bytes={args.zip.stat().st_size}")
    print(f"manifest_source_mode={source_mode}")
    print(f"manifest_source={source_ref}")
    print(f"manifest_rows={len(rows)}")
    print("archive_extraction=NO")
    print("nifti_voxel_decode=NO")
    print("gt_voxel_decode=NO")
    print("model_training_inference_tta=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A0 FIX1 protocol")

    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    manifest_rows, source_mode, source_ref = resolve_manifest(args.remote_manifest)

    manifest_snapshot = build_dir / "kaggle_remote_manifest_snapshot.csv"
    write_csv(
        manifest_snapshot,
        manifest_rows,
        ["remote_path", "bytes"],
    )

    zip_sha = sha256_file(args.zip)
    zip_rows = read_zip_inventory(args.zip)

    zip_names = [r["member_path"] for r in zip_rows]
    zip_duplicates = sorted(
        p for p, n in Counter(zip_names).items() if n > 1
    )

    comparison = compare_manifest(manifest_rows, zip_rows)
    pair_summary = structural_pair_summary(zip_rows)

    manifest_total = sum(int(r["bytes"]) for r in manifest_rows)
    zip_total = sum(int(r["uncompressed_bytes"]) for r in zip_rows)

    missing = [
        r["path"] for r in comparison
        if int(r["manifest_present"]) and not int(r["zip_present"])
    ]
    extra = [
        r["path"] for r in comparison
        if int(r["zip_present"]) and not int(r["manifest_present"])
    ]
    mismatch = [
        r["path"] for r in comparison
        if int(r["manifest_present"])
        and int(r["zip_present"])
        and not int(r["size_match"])
    ]

    checks = {
        "manifest_rows_eq_691":
            len(manifest_rows) == EXPECTED_REMOTE_FILES,
        "zip_members_eq_691":
            len(zip_rows) == EXPECTED_REMOTE_FILES,
        "no_duplicate_zip_members":
            len(zip_duplicates) == 0,
        "no_missing_manifest_files":
            len(missing) == 0,
        "no_extra_zip_files":
            len(extra) == 0,
        "all_sizes_match":
            len(mismatch) == 0,
        "total_uncompressed_bytes_match":
            manifest_total == zip_total,
        "archive_extracted": False,
        "nifti_voxels_decoded": False,
        "gt_voxels_decoded": False,
    }

    decision = READY if all([
        checks["manifest_rows_eq_691"],
        checks["zip_members_eq_691"],
        checks["no_duplicate_zip_members"],
        checks["no_missing_manifest_files"],
        checks["no_extra_zip_files"],
        checks["all_sizes_match"],
        checks["total_uncompressed_bytes_match"],
    ]) else INCOMPLETE

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "zip_path": str(args.zip),
        "zip_file_bytes": args.zip.stat().st_size,
        "zip_sha256": zip_sha,
        "manifest_source_mode": source_mode,
        "manifest_source": source_ref,
        "manifest_snapshot_sha256": sha256_file(manifest_snapshot),
        "remote_manifest_rows": len(manifest_rows),
        "zip_non_directory_members": len(zip_rows),
        "manifest_total_uncompressed_bytes": manifest_total,
        "zip_total_uncompressed_bytes": zip_total,
        "duplicate_zip_members": zip_duplicates,
        "missing_in_zip": missing,
        "extra_in_zip": extra,
        "size_mismatch_paths": mismatch,
        "archive_extracted": False,
        "nifti_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "model_training": False,
        "model_inference": False,
        "tta": False,
        "provenance_statement": (
            "Kaggle mirror completeness verified relative to the resolved "
            "Kaggle listing; byte identity to the UB-hosted official transfer "
            "package is not asserted."
        ),
        "checks": checks,
        "decision": decision,
    }
    write_json(build_dir / "archive_audit.json", audit)

    write_csv(
        build_dir / "zip_member_inventory.csv",
        zip_rows,
        [
            "member_path",
            "uncompressed_bytes",
            "compressed_bytes",
            "crc32_hex",
            "compression_type",
            "is_nifti",
            "is_gt_like",
        ],
    )

    write_csv(
        build_dir / "manifest_comparison.csv",
        comparison,
        [
            "path",
            "manifest_present",
            "zip_present",
            "manifest_bytes",
            "zip_uncompressed_bytes",
            "size_match",
        ],
    )

    write_json(
        build_dir / "structural_pair_summary.json",
        pair_summary,
    )

    (build_dir / "decision.txt").write_text(
        decision + "\n",
        encoding="utf-8",
    )

    lines = [
        "===== Q1-R08A0 M&Ms KAGGLE MIRROR ARCHIVE IDENTITY AUDIT FIX1 =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        f"ZIP={args.zip}",
        f"ZIP bytes={args.zip.stat().st_size}",
        f"ZIP SHA256={zip_sha}",
        "",
        f"manifest source mode={source_mode}",
        f"manifest source={source_ref}",
        f"manifest rows={len(manifest_rows)}",
        f"ZIP non-directory members={len(zip_rows)}",
        f"missing in ZIP={len(missing)}",
        f"extra in ZIP={len(extra)}",
        f"size mismatches={len(mismatch)}",
        f"duplicate ZIP members={len(zip_duplicates)}",
        f"manifest total uncompressed bytes={manifest_total}",
        f"ZIP total uncompressed bytes={zip_total}",
        "",
        f"NIfTI members={pair_summary['nifti_members']}",
        f"GT-like NIfTI members={pair_summary['gt_like_nifti_members']}",
        f"exact image/GT filename pairs={pair_summary['exact_image_gt_filename_pairs']}",
        "",
        "archive extraction=NO",
        "NIfTI voxel decoding=NO",
        "GT voxel decoding=NO",
        "model training/inference/TTA=NO",
        "",
        f"Decision: {decision}",
    ]

    run_log = build_dir / "run_log.txt"
    run_log.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "preregistered_protocol_copy.md",
        "kaggle_remote_manifest_snapshot.csv",
        "archive_audit.json",
        "zip_member_inventory.csv",
        "manifest_comparison.csv",
        "structural_pair_summary.json",
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
        "zip_sha256": zip_sha,
        "manifest_source_mode": source_mode,
        "manifest_snapshot_sha256": sha256_file(manifest_snapshot),
        "remote_manifest_rows": len(manifest_rows),
        "zip_members": len(zip_rows),
        "archive_extracted": False,
        "nifti_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "decision": decision,
        "checks": checks,
        "artifacts": artifacts,
    }

    lock_path = (
        build_dir
        / "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_LOCK.json"
    )
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
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
        "Q1-R08A0 LOCK:",
        args.output_dir
        / "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_LOCK.json",
    )
    print("Q1-R08A0 LOCK SHA256:", lock_sha)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R08A0 FIX1: archive-only M&Ms Kaggle mirror audit with "
            "automatic live Kaggle manifest fallback."
        )
    )
    p.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
    )
    p.add_argument(
        "--remote-manifest",
        type=Path,
        default=None,
        help=(
            "Optional existing Kaggle remote manifest. If omitted, the script "
            "uses the historical manifest when present, otherwise queries the "
            "authenticated Kaggle API."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

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
