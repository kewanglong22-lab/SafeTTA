#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from pathlib import Path
from typing import Sequence

VERSION = "2026-08-19-Q1-R08A0-v1"
BUILD = "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_IDENTITY_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = ROOT / "docs" / "Q1_R08A0_mnms_kaggle_mirror_archive_identity_audit_preregistered_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "59a85e155bed32783d917e30f17713fee50b0501aceef856103e3dc1e5450ef5"

DEFAULT_ZIP = ROOT / "data" / "downloads" / "MnMs_kaggle" / "MnMs.zip"
DEFAULT_REMOTE_MANIFEST = ROOT / "data" / "downloads" / "MnMs_kaggle_parallel" / "_kaggle_remote_manifest.csv"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R08A0_mnms_kaggle_mirror_archive_identity_audit_v1"

EXPECTED_REMOTE_FILES = 691
READY = "MNMS_KAGGLE_MIRROR_ARCHIVE_COMPLETE"
INCOMPLETE = "MNMS_KAGGLE_MIRROR_ARCHIVE_INCOMPLETE"
GT_MARKERS = ("_gt", "-mask", "_mask", "-label", "_label", "-seg", "_seg")


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
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return actual


def normalize_member_name(name: str) -> str:
    s = str(name).replace("\\", "/").lstrip("/")
    while s.startswith("./"):
        s = s[2:]
    return s


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


def read_remote_manifest(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"remote_path", "bytes"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("Remote manifest must contain remote_path and bytes.")
        for i, row in enumerate(reader):
            rows.append({
                "remote_path": normalize_member_name(row["remote_path"]),
                "bytes": int(row["bytes"]),
            })

    if len(rows) != EXPECTED_REMOTE_FILES:
        raise RuntimeError(f"Remote manifest rows={len(rows)} expected={EXPECTED_REMOTE_FILES}")

    paths = [r["remote_path"] for r in rows]
    dup = [p for p, n in Counter(paths).items() if n > 1]
    if dup:
        raise RuntimeError(f"Remote manifest has duplicate paths: {dup[:10]}")
    return rows


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
    m = {r["remote_path"]: r["bytes"] for r in manifest_rows}
    z = {r["member_path"]: r["uncompressed_bytes"] for r in zip_rows}
    rows = []
    for p in sorted(set(m) | set(z)):
        mp = p in m
        zp = p in z
        ms = m.get(p)
        zs = z.get(p)
        rows.append({
            "path": p,
            "manifest_present": int(mp),
            "zip_present": int(zp),
            "manifest_bytes": "" if ms is None else ms,
            "zip_uncompressed_bytes": "" if zs is None else zs,
            "size_match": int(mp and zp and ms == zs),
        })
    return rows


def structural_pair_summary(zip_rows):
    nifti = [r for r in zip_rows if r["is_nifti"] == 1]
    groups = defaultdict(lambda: {"image": 0, "gt": 0})
    for r in nifti:
        k = pair_key(r["member_path"])
        groups[k]["gt" if r["is_gt_like"] else "image"] += 1

    return {
        "nifti_members": len(nifti),
        "gt_like_nifti_members": sum(r["is_gt_like"] for r in nifti),
        "non_gt_nifti_members": sum(1-r["is_gt_like"] for r in nifti),
        "normalized_pair_keys": len(groups),
        "exact_image_gt_filename_pairs": sum(
            1 for g in groups.values() if g["image"] == 1 and g["gt"] == 1
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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def self_test():
    assert normalize_member_name("./MnM/A.nii") == "MnM/A.nii"
    assert is_nifti("A_sa.nii")
    assert is_gt_like("A_sa_gt.nii")
    assert not is_gt_like("A_sa.nii")
    assert pair_key("A_sa.nii") == pair_key("A_sa_gt.nii")
    assert EXPECTED_REMOTE_FILES == 691
    print("PATH_NORMALIZATION_TEST_PASS")
    print("NIFTI_FILENAME_TEST_PASS")
    print("GT_FILENAME_TEST_PASS")
    print("PAIR_KEY_TEST_PASS")
    print("EXPECTED_KAGGLE_CARDINALITY_TEST_PASS")
    print("SELF_TEST_PASS")


def preflight(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A0 protocol")
    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)
    if not args.remote_manifest.is_file():
        raise FileNotFoundError(args.remote_manifest)

    print("===== Q1-R08A0 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"zip={args.zip}")
    print(f"zip_bytes={args.zip.stat().st_size}")
    print(f"remote_manifest={args.remote_manifest}")
    print("archive_extraction=NO")
    print("nifti_voxel_decode=NO")
    print("gt_voxel_decode=NO")
    print("model_training_inference_tta=NO")
    print("PREFLIGHT_PASS")


def run(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R08A0 protocol")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(protocol_copy, EXPECTED_PROTOCOL_SHA256, "protocol copy")

    manifest_rows = read_remote_manifest(args.remote_manifest)
    zip_sha = sha256_file(args.zip)
    zip_rows = read_zip_inventory(args.zip)

    zip_names = [r["member_path"] for r in zip_rows]
    zip_duplicates = sorted(p for p, n in Counter(zip_names).items() if n > 1)
    comparison = compare_manifest(manifest_rows, zip_rows)
    pair_summary = structural_pair_summary(zip_rows)

    manifest_total = sum(r["bytes"] for r in manifest_rows)
    zip_total = sum(r["uncompressed_bytes"] for r in zip_rows)

    missing = [r["path"] for r in comparison if r["manifest_present"] and not r["zip_present"]]
    extra = [r["path"] for r in comparison if r["zip_present"] and not r["manifest_present"]]
    mismatch = [
        r["path"] for r in comparison
        if r["manifest_present"] and r["zip_present"] and not r["size_match"]
    ]

    checks = {
        "manifest_rows_eq_691": len(manifest_rows) == EXPECTED_REMOTE_FILES,
        "zip_members_eq_691": len(zip_rows) == EXPECTED_REMOTE_FILES,
        "no_duplicate_zip_members": len(zip_duplicates) == 0,
        "no_missing_manifest_files": len(missing) == 0,
        "no_extra_zip_files": len(extra) == 0,
        "all_sizes_match": len(mismatch) == 0,
        "total_uncompressed_bytes_match": manifest_total == zip_total,
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
        "remote_manifest_path": str(args.remote_manifest),
        "remote_manifest_sha256": sha256_file(args.remote_manifest),
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
            "Kaggle mirror completeness verified relative to Kaggle listing; "
            "byte identity to the UB-hosted official transfer package is not asserted."
        ),
        "checks": checks,
        "decision": decision,
    }
    write_json(build_dir / "archive_audit.json", audit)

    write_csv(
        build_dir / "zip_member_inventory.csv",
        zip_rows,
        ["member_path","uncompressed_bytes","compressed_bytes","crc32_hex","compression_type","is_nifti","is_gt_like"],
    )
    write_csv(
        build_dir / "manifest_comparison.csv",
        comparison,
        ["path","manifest_present","zip_present","manifest_bytes","zip_uncompressed_bytes","size_match"],
    )
    write_json(build_dir / "structural_pair_summary.json", pair_summary)
    (build_dir / "decision.txt").write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R08A0 M&Ms KAGGLE MIRROR ARCHIVE IDENTITY AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        f"ZIP={args.zip}",
        f"ZIP bytes={args.zip.stat().st_size}",
        f"ZIP SHA256={zip_sha}",
        "",
        f"remote manifest rows={len(manifest_rows)}",
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
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_names = [
        "preregistered_protocol_copy.md",
        "archive_audit.json",
        "zip_member_inventory.csv",
        "manifest_comparison.csv",
        "structural_pair_summary.json",
        "decision.txt",
        "run_log.txt",
    ]
    artifacts = {
        name: {"relative_path": name, "sha256": sha256_file(build_dir / name)}
        for name in artifact_names
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "zip_sha256": zip_sha,
        "remote_manifest_sha256": sha256_file(args.remote_manifest),
        "remote_manifest_rows": len(manifest_rows),
        "zip_members": len(zip_rows),
        "archive_extracted": False,
        "nifti_voxels_decoded": False,
        "gt_voxels_decoded": False,
        "decision": decision,
        "checks": checks,
        "artifacts": artifacts,
    }
    lock_path = build_dir / "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in artifacts.items():
        if sha256_file(build_dir / meta["relative_path"]) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print("Q1-R08A0 LOCK:", args.output_dir / "Q1_R08A0_MNMS_KAGGLE_MIRROR_ARCHIVE_LOCK.json")
    print("Q1-R08A0 LOCK SHA256:", lock_sha)


def parse_args():
    p = argparse.ArgumentParser(description="Archive-only M&Ms Kaggle mirror audit.")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--remote-manifest", type=Path, default=DEFAULT_REMOTE_MANIFEST)
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
