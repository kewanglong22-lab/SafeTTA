#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S00-A1 Exact dataset parser, selective extractor, and integrity audit.

Frozen project:
    F:\MEDSEG_SAFETTA

Input archive:
    F:\MEDSEG_SAFETTA\data\downloads\PolypDataset.zip

Expected selected protocol:
    source_train:
        NewTRimage / NewTRmask = 1450 pairs

    seen_test:
        Kvasir-SEG = 100 pairs
        CVC-ClinicDB = 62 pairs

    unseen_test:
        CVC-ColonDB = 380 pairs
        CVC-300 = 60 pairs
        ETIS-LaribPolypDB = 196 pairs

Total:
    2248 image-mask pairs = 4496 selected files.

Explicitly excluded:
    NewTRimageDA
    NewTRmaskDA
    OLDTRimage
    OLDTRimageMask

No model training or target-label-based model operation occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-17-S00-A1-v1"
BUILD = "S00_A1_EXACT_PROTOCOL_SELECTIVE_EXTRACT_INTEGRITY"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_ARCHIVE = ROOT / "data" / "downloads" / "PolypDataset.zip"
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "S00_polyp_locked_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "S00_A1_exact_dataset_extract_and_integrity_v1"
DEFAULT_LOCKED_MANIFEST = ROOT / "data" / "splits" / "S00_polyp_locked_manifest_v1.csv"

EXPECTED_ARCHIVE_MD5 = "0c45c453bc16424df7a2528966ccd939"

EXPECTED_COUNTS = {
    ("source_train", "SOURCE_COMBINED"): 1450,
    ("seen_test", "Kvasir-SEG"): 100,
    ("seen_test", "CVC-ClinicDB"): 62,
    ("unseen_test", "CVC-ColonDB"): 380,
    ("unseen_test", "CVC-300"): 60,
    ("unseen_test", "ETIS-LaribPolypDB"): 196,
}

TOTAL_EXPECTED_PAIRS = sum(EXPECTED_COUNTS.values())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

EXCLUDED_TOP_LEVELS = {
    "newtrimageda",
    "newtrmaskda",
    "oldtrimage",
    "oldtrimagemask",
}

TEST_DATASET_ALIASES = {
    "Kvasir-SEG": {"kvasir", "kvasir-seg", "kvasir_seg", "kvasirseg"},
    "CVC-ClinicDB": {
        "cvc-clinicdb", "cvc_clinicdb", "cvcclinicdb",
        "clinicdb", "clinic-db", "clinic_db", "cvc-612", "cvc_612"
    },
    "CVC-ColonDB": {
        "cvc-colondb", "cvc_colondb", "cvccolondb",
        "colondb", "colon-db", "colon_db"
    },
    "CVC-300": {"cvc-300", "cvc_300", "cvc300"},
    "ETIS-LaribPolypDB": {
        "etis-laribpolypdb", "etis_laribpolypdb",
        "etislaribpolypdb", "etis", "larib"
    },
}

IMAGE_DIR_TOKENS = {"image", "images", "img", "imgs"}
MASK_DIR_TOKENS = {
    "mask", "masks", "gt", "gts", "groundtruth", "ground_truth",
    "label", "labels"
}


def norm_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def md5_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Archive MD5",
        leave=False,
    ) as bar:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def normalize_pair_key(stem: str) -> str:
    s = stem.strip().lower()
    for suffix in ("_mask", "-mask", "_gt", "-gt", "_label", "-label"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def classify_test_dataset(parts):
    partset = set(parts)
    for canonical, aliases in TEST_DATASET_ALIASES.items():
        if partset & aliases:
            return canonical
    return None


def classify_role_exact(parts):
    """
    Exact directory-token role matching.
    This intentionally fixes the S00-A0 heuristic issue where 'gt' occurring
    inside a normal filename could incorrectly classify an image as a mask.
    """
    dirs = parts[:-1]
    has_image = any(p in IMAGE_DIR_TOKENS for p in dirs)
    has_mask = any(p in MASK_DIR_TOKENS for p in dirs)

    if has_image and not has_mask:
        return "image"
    if has_mask and not has_image:
        return "mask"
    return None


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_selected_members(zf: zipfile.ZipFile):
    source_images = []
    source_masks = []
    test = defaultdict(lambda: {"image": [], "mask": []})
    ignored = Counter()
    unresolved_test = []

    for info in zf.infolist():
        if info.is_dir():
            continue

        name = norm_member(info.filename)
        p = PurePosixPath(name)
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTS:
            ignored["non_image_extension"] += 1
            continue

        parts = [x.lower() for x in p.parts]
        if not parts:
            continue

        top = parts[0]

        if top in EXCLUDED_TOP_LEVELS:
            ignored[f"excluded_top::{top}"] += 1
            continue

        if top == "newtrimage":
            source_images.append(name)
            continue

        if top == "newtrmask":
            source_masks.append(name)
            continue

        if top == "testdataset":
            dataset = classify_test_dataset(parts)
            role = classify_role_exact(parts)
            if dataset is None or role is None:
                unresolved_test.append(name)
            else:
                test[dataset][role].append(name)
            continue

        ignored[f"other_top::{top}"] += 1

    return source_images, source_masks, test, ignored, unresolved_test


def make_unique_map(members, label):
    mapping = defaultdict(list)
    for member in members:
        key = normalize_pair_key(PurePosixPath(member).stem)
        mapping[key].append(member)

    bad = {k: v for k, v in mapping.items() if len(v) != 1}
    if bad:
        raise RuntimeError(
            f"{label}: non-unique normalized pair keys. Examples={list(bad.items())[:10]}"
        )
    return {k: v[0] for k, v in mapping.items()}


def pair_members(images, masks, label):
    imap = make_unique_map(images, label + "/images")
    mmap = make_unique_map(masks, label + "/masks")

    image_only = sorted(set(imap) - set(mmap))
    mask_only = sorted(set(mmap) - set(imap))
    if image_only or mask_only:
        raise RuntimeError(
            f"{label}: image-mask pairing failed. "
            f"image_only={len(image_only)} mask_only={len(mask_only)} "
            f"image_examples={image_only[:10]} mask_examples={mask_only[:10]}"
        )

    return [(k, imap[k], mmap[k]) for k in sorted(imap)]


def build_protocol_pairs(zf):
    source_images, source_masks, test, ignored, unresolved = parse_selected_members(zf)

    if unresolved:
        raise RuntimeError(
            f"Unresolved TestDataset image files: {len(unresolved)}. "
            f"Examples={unresolved[:20]}"
        )

    pairs = []

    source_pairs = pair_members(
        source_images, source_masks, "source_train/SOURCE_COMBINED"
    )
    exp_source = EXPECTED_COUNTS[("source_train", "SOURCE_COMBINED")]
    if len(source_pairs) != exp_source:
        raise RuntimeError(
            f"Source count mismatch: expected={exp_source} actual={len(source_pairs)}"
        )

    for key, image_member, mask_member in source_pairs:
        pairs.append(
            ("source_train", "SOURCE_COMBINED", key, image_member, mask_member)
        )

    test_to_split = {
        "Kvasir-SEG": "seen_test",
        "CVC-ClinicDB": "seen_test",
        "CVC-ColonDB": "unseen_test",
        "CVC-300": "unseen_test",
        "ETIS-LaribPolypDB": "unseen_test",
    }

    for dataset, split in test_to_split.items():
        if dataset not in test:
            raise RuntimeError(f"Expected test dataset not found: {dataset}")

        image_members = test[dataset]["image"]
        mask_members = test[dataset]["mask"]
        ds_pairs = pair_members(
            image_members, mask_members, f"{split}/{dataset}"
        )

        expected = EXPECTED_COUNTS[(split, dataset)]
        if len(ds_pairs) != expected:
            raise RuntimeError(
                f"{dataset} count mismatch: expected={expected} actual={len(ds_pairs)} "
                f"images={len(image_members)} masks={len(mask_members)}"
            )

        for key, image_member, mask_member in ds_pairs:
            pairs.append((split, dataset, key, image_member, mask_member))

    if len(pairs) != TOTAL_EXPECTED_PAIRS:
        raise RuntimeError(
            f"Total pair mismatch: expected={TOTAL_EXPECTED_PAIRS} actual={len(pairs)}"
        )

    return pairs, ignored


def safe_basename(member: str) -> str:
    name = PurePosixPath(member).name
    if not name or name in {".", ".."}:
        raise RuntimeError(f"Unsafe ZIP member filename: {member}")
    return name


def extract_member(zf, member: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, dest.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def inspect_image(path: Path):
    with Image.open(path) as image:
        image.load()
        size = tuple(image.size)
        mode = image.mode
        rgb = np.asarray(image.convert("RGB"))
    pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    return size, mode, pixel_sha


def inspect_mask(path: Path):
    with Image.open(path) as mask:
        mask.load()
        size = tuple(mask.size)
        mode = mask.mode
        array = np.asarray(mask)

    unique = sorted(int(v) for v in np.unique(array).tolist())
    binary = set(unique).issubset({0, 255})

    if array.ndim == 3:
        fg = np.any(array > 0, axis=2).astype(np.uint8)
    else:
        fg = (array > 0).astype(np.uint8)

    fg_hash = hashlib.sha256(fg.tobytes()).hexdigest()
    fg_fraction = float(fg.mean())
    return size, mode, unique, binary, fg_hash, fg_fraction


def run(
    archive: Path,
    data_dir: Path,
    output_dir: Path,
    locked_manifest: Path,
    verify_md5: bool,
    overwrite: bool,
):
    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Archive        : {archive}")
    print(f"Data dir       : {data_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Locked manifest: {locked_manifest}")
    print()

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    if verify_md5:
        actual_md5 = md5_file(archive)
        print(f"MD5 expected: {EXPECTED_ARCHIVE_MD5}")
        print(f"MD5 actual  : {actual_md5}")
        if actual_md5.lower() != EXPECTED_ARCHIVE_MD5:
            raise RuntimeError("Archive MD5 mismatch.")
    else:
        actual_md5 = None
        print("Archive MD5 recheck: SKIPPED (already independently verified)")

    build_data = Path(str(data_dir) + "__building")
    build_output = Path(str(output_dir) + "__building")

    for p in (data_dir, output_dir, build_data, build_output):
        if p.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Path exists: {p}\n"
                    "Refusing to overwrite. Use --overwrite only for technical reruns."
                )
            shutil.rmtree(p)

    build_data.mkdir(parents=True)
    build_output.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC failure at: {bad_member}")

            pairs, ignored = build_protocol_pairs(zf)
            count_by_group = Counter((s, d) for s, d, _, _, _ in pairs)

            print("Exact protocol counts:")
            for key, expected in EXPECTED_COUNTS.items():
                print(
                    f"  {key[0]:12s} {key[1]:22s}: "
                    f"{count_by_group[key]} / expected {expected}"
                )
            print(f"  TOTAL: {len(pairs)} / expected {TOTAL_EXPECTED_PAIRS}")
            print()

            print("Explicit exclusions / ignored image files:")
            for key, value in ignored.most_common():
                print(f"  {key}: {value}")
            print()

            rows = []

            for split, dataset, pair_key, image_member, mask_member in tqdm(
                pairs, desc="Extract + integrity", unit="pair"
            ):
                if split == "source_train":
                    base = build_data / "source_train"
                else:
                    base = build_data / split / dataset

                image_dest = base / "images" / safe_basename(image_member)
                mask_dest = base / "masks" / safe_basename(mask_member)

                if image_dest.exists() or mask_dest.exists():
                    raise RuntimeError(
                        f"Filename collision: {image_dest} / {mask_dest}"
                    )

                extract_member(zf, image_member, image_dest)
                extract_member(zf, mask_member, mask_dest)

                image_size, image_mode, image_pixel_sha = inspect_image(image_dest)
                mask_size, mask_mode, mask_unique, mask_binary, mask_fg_sha, fg_fraction = inspect_mask(mask_dest)

                if image_size != mask_size:
                    raise RuntimeError(
                        f"Image/mask dimension mismatch for {split}/{dataset}/{pair_key}: "
                        f"{image_size} vs {mask_size}"
                    )

                if not mask_binary:
                    raise RuntimeError(
                        f"Non-binary mask for {split}/{dataset}/{pair_key}: "
                        f"values={mask_unique[:50]}"
                    )

                rows.append({
                    "sample_id": f"{split}::{dataset}::{pair_key}",
                    "split": split,
                    "dataset": dataset,
                    "pair_key": pair_key,
                    "archive_image_member": image_member,
                    "archive_mask_member": mask_member,
                    "image_relpath": str(image_dest.relative_to(build_data)).replace("\\", "/"),
                    "mask_relpath": str(mask_dest.relative_to(build_data)).replace("\\", "/"),
                    "width": image_size[0],
                    "height": image_size[1],
                    "image_mode": image_mode,
                    "mask_mode": mask_mode,
                    "mask_unique_values": "|".join(map(str, mask_unique)),
                    "mask_binary_0_255": True,
                    "mask_foreground_fraction": f"{fg_fraction:.10f}",
                    "image_file_sha256": hashlib.sha256(image_dest.read_bytes()).hexdigest(),
                    "image_pixel_sha256_rgb": image_pixel_sha,
                    "mask_file_sha256": hashlib.sha256(mask_dest.read_bytes()).hexdigest(),
                    "mask_foreground_sha256": mask_fg_sha,
                })

        # Cross-split exact duplicate audit using decoded RGB pixels.
        by_pixel_hash = defaultdict(list)
        for row in rows:
            by_pixel_hash[row["image_pixel_sha256_rgb"]].append(row)

        duplicate_rows = []
        cross_split_groups = 0

        for pixel_hash, group in by_pixel_hash.items():
            if len(group) <= 1:
                continue

            splits = sorted({r["split"] for r in group})
            datasets = sorted({r["dataset"] for r in group})
            cross_split = len(splits) > 1
            if cross_split:
                cross_split_groups += 1

            for row in group:
                duplicate_rows.append({
                    "pixel_sha256_rgb": pixel_hash,
                    "group_size": len(group),
                    "cross_split": cross_split,
                    "splits": "|".join(splits),
                    "datasets": "|".join(datasets),
                    "sample_id": row["sample_id"],
                    "image_relpath": row["image_relpath"],
                })

        if cross_split_groups:
            raise RuntimeError(
                f"Found {cross_split_groups} exact decoded-pixel duplicate groups "
                f"across protocol splits. Manual review required."
            )

        summary_rows = []
        for (split, dataset), expected in EXPECTED_COUNTS.items():
            group = [
                r for r in rows
                if r["split"] == split and r["dataset"] == dataset
            ]
            summary_rows.append({
                "split": split,
                "dataset": dataset,
                "expected_pairs": expected,
                "actual_pairs": len(group),
                "binary_masks": sum(bool(r["mask_binary_0_255"]) for r in group),
                "mean_mask_foreground_fraction": (
                    f"{np.mean([float(r['mask_foreground_fraction']) for r in group]):.10f}"
                    if group else ""
                ),
            })

        manifest_fields = [
            "sample_id", "split", "dataset", "pair_key",
            "archive_image_member", "archive_mask_member",
            "image_relpath", "mask_relpath",
            "width", "height", "image_mode", "mask_mode",
            "mask_unique_values", "mask_binary_0_255",
            "mask_foreground_fraction",
            "image_file_sha256", "image_pixel_sha256_rgb",
            "mask_file_sha256", "mask_foreground_sha256",
        ]
        write_csv(build_output / "selected_manifest.csv", rows, manifest_fields)
        write_csv(
            build_output / "dataset_summary.csv",
            summary_rows,
            [
                "split", "dataset", "expected_pairs", "actual_pairs",
                "binary_masks", "mean_mask_foreground_fraction"
            ],
        )
        write_csv(
            build_output / "exact_pixel_duplicate_groups.csv",
            duplicate_rows,
            [
                "pixel_sha256_rgb", "group_size", "cross_split", "splits",
                "datasets", "sample_id", "image_relpath"
            ],
        )

        audit = {
            "script_version": VERSION,
            "build": BUILD,
            "archive": str(archive),
            "archive_md5_expected": EXPECTED_ARCHIVE_MD5,
            "archive_md5_actual": actual_md5,
            "archive_md5_rechecked_this_run": verify_md5,
            "zip_crc": "PASS",
            "total_selected_pairs": len(rows),
            "total_selected_files": 2 * len(rows),
            "expected_total_pairs": TOTAL_EXPECTED_PAIRS,
            "cross_split_exact_pixel_duplicate_groups": cross_split_groups,
            "all_masks_binary_0_255": all(bool(r["mask_binary_0_255"]) for r in rows),
            "all_image_mask_sizes_match": True,
            "excluded_archive_folders": [
                "NewTRimageDA",
                "NewTRmaskDA",
                "OLDTRimage",
                "OLDTRimageMask",
            ],
            "decision": "S00_A1_PASS_READY_FOR_S01_SOURCE_BASELINE",
        }
        (build_output / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_lines = [
            "===== S00-A1 EXACT DATASET EXTRACT + INTEGRITY AUDIT =====",
            f"Script version: {VERSION}",
            f"Build: {BUILD}",
            f"Archive: {archive}",
            f"Frozen archive MD5: {EXPECTED_ARCHIVE_MD5}",
            f"MD5 rechecked this run: {verify_md5}",
            "ZIP CRC: PASS",
            "",
            "Exact frozen protocol:",
        ]

        for row in summary_rows:
            summary_lines.append(
                f"  {row['split']:12s} {row['dataset']:22s}: "
                f"{row['actual_pairs']} pairs "
                f"(expected {row['expected_pairs']}), "
                f"binary masks={row['binary_masks']}"
            )

        summary_lines += [
            "",
            f"Total selected pairs: {len(rows)}",
            f"Total selected files: {2 * len(rows)}",
            f"Cross-split exact decoded-pixel duplicate groups: {cross_split_groups}",
            "All masks binary (0/255 only): YES",
            "All image/mask dimensions matched: YES",
            "",
            "Explicitly excluded:",
            "  NewTRimageDA",
            "  NewTRmaskDA",
            "  OLDTRimage",
            "  OLDTRimageMask",
            "",
            "No model training performed.",
            "No target masks used for model selection/adaptation.",
            "",
            "Decision: S00_A1_PASS_READY_FOR_S01_SOURCE_BASELINE",
        ]

        (build_output / "summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        # Commit only after every audit passes.
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        build_data.rename(data_dir)
        build_output.rename(output_dir)

        locked_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / "selected_manifest.csv", locked_manifest)

        print()
        print((output_dir / "summary.txt").read_text(encoding="utf-8"))
        print(f"[OK] Data     : {data_dir}")
        print(f"[OK] Outputs  : {output_dir}")
        print(f"[OK] Manifest : {locked_manifest}")

    except Exception:
        print("", file=sys.stderr)
        print("[FAILED] Final S00-A1 directories were NOT committed.", file=sys.stderr)
        print(f"Partial data build  : {build_data}", file=sys.stderr)
        print(f"Partial output build: {build_output}", file=sys.stderr)
        raise


def self_test():
    # Exact role parser must not treat "gt" inside a filename as a mask.
    assert classify_role_exact(
        ["testdataset", "kvasir", "images", "abcgtxyz.png"]
    ) == "image"

    assert classify_role_exact(
        ["testdataset", "kvasir", "masks", "abc.png"]
    ) == "mask"

    assert classify_test_dataset(
        ["testdataset", "cvc-300", "images", "1.png"]
    ) == "CVC-300"

    assert classify_test_dataset(
        ["testdataset", "etis-laribpolypdb", "masks", "1.png"]
    ) == "ETIS-LaribPolypDB"

    assert normalize_pair_key("sample_mask") == "sample"
    assert normalize_pair_key("sample") == "sample"

    imgs = ["NewTRimage/a.png", "NewTRimage/b.jpg"]
    masks = ["NewTRmask/a.png", "NewTRmask/b.png"]
    pairs = pair_members(imgs, masks, "selftest")
    assert len(pairs) == 2
    assert {p[0] for p in pairs} == {"a", "b"}

    print("A0_GT_SUBSTRING_FALSE_ROLE_FIX_TEST_PASS")
    print("EXACT_DIRECTORY_ROLE_TEST_PASS")
    print("PAIRING_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "S00-A1 exact parser + selective extraction + integrity audit "
            "for the frozen polyp protocol."
        )
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--locked-manifest", type=Path, default=DEFAULT_LOCKED_MANIFEST
    )
    parser.add_argument(
        "--verify-md5",
        action="store_true",
        help="Recompute the 4 GB archive MD5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Use only for a technical rerun of S00-A1.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    run(
        archive=args.archive,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        locked_manifest=args.locked_manifest,
        verify_md5=args.verify_md5,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
