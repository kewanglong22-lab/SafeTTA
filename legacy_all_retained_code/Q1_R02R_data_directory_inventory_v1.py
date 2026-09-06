#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R02R — DATA directory inventory

Inventory F:\\MEDSEG_SAFETTA\\data using filenames and filesystem metadata only.

NO image decoding.
NO model loading.
NO training.
NO inference.
NO modification inside data/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02R-data-inventory-v1"
BUILD = "Q1_R02R_DATA_DIRECTORY_INVENTORY"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R02R_data_directory_inventory_v1"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".gif", ".webp",
}

MASK_SUFFIXES = (
    "_mask", "-mask", "_masks",
    "_gt", "-gt",
    "_label", "-label",
    "_seg", "-seg",
)

IMAGE_DIR_TOKENS = {
    "image", "images", "img", "imgs", "original", "originals",
}

MASK_DIR_TOKENS = {
    "mask", "masks", "gt", "gts", "label", "labels",
    "annotation", "annotations", "ground_truth", "groundtruth",
}

DATASET_HINTS = {
    "Kvasir-SEG": (
        "kvasir-seg", "kvasir_seg", "kvasirseg", "kvasir"
    ),
    "CVC-ClinicDB": (
        "cvc-clinicdb", "cvc_clinicdb", "clinicdb"
    ),
    "CVC-ColonDB": (
        "cvc-colondb", "cvc_colondb", "colondb", "colon-db"
    ),
    "CVC-300": (
        "cvc-300", "cvc_300", "cvc300"
    ),
    "ETIS": (
        "etis-larib", "etis_larib", "etis"
    ),
    "PolypGen": (
        "polypgen",
    ),
    "S00_polyp_locked_v1": (
        "s00_polyp_locked_v1",
    ),
}

MAX_SAMPLE_FILES_PER_DIR = 8


def normalize_stem(stem: str) -> str:
    s = stem.lower().strip()
    changed = True
    while changed:
        changed = False
        for suffix in MASK_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                changed = True
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rel(path: Path, base: Path) -> str:
    try:
        value = str(path.resolve().relative_to(base.resolve()))
        return value if value else "."
    except Exception:
        return str(path)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def detect_dataset_hint(path: Path) -> str:
    low = str(path).replace("\\", "/").lower()
    hits = []
    for name, tokens in DATASET_HINTS.items():
        if any(token in low for token in tokens):
            hits.append(name)
    return ";".join(hits)


def dir_role_hint(path: Path) -> str:
    name = path.name.lower().replace("-", "_")

    if name in IMAGE_DIR_TOKENS:
        return "IMAGE_DIR"
    if name in MASK_DIR_TOKENS:
        return "MASK_DIR"

    if "image" in name or name in {"img", "imgs"}:
        return "IMAGE_LIKE_DIR"
    if (
        "mask" in name
        or "label" in name
        or "annotation" in name
        or name in {"gt", "gts"}
    ):
        return "MASK_LIKE_DIR"
    if "train" in name:
        return "TRAIN_LIKE_DIR"
    if "val" in name or "valid" in name:
        return "VAL_LIKE_DIR"
    if "test" in name:
        return "TEST_LIKE_DIR"

    return ""


def scan_data_root(data_root: Path):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    walk_items = []
    for current, dirs, files in os.walk(data_root):
        walk_items.append(
            (Path(current), sorted(dirs), sorted(files))
        )

    directory_rows = []
    extension_counter = Counter()
    total_files = 0
    total_bytes = 0

    for cp, dirs, files in tqdm(
        walk_items,
        desc="Inventorying data directory",
        unit="dir",
        dynamic_ncols=True,
    ):
        direct_bytes = 0
        direct_image_files = 0
        samples = []

        for name in files:
            p = cp / name
            try:
                stat = p.stat()
            except Exception:
                continue

            total_files += 1
            total_bytes += stat.st_size
            direct_bytes += stat.st_size

            ext = p.suffix.lower() or "<no_extension>"
            extension_counter[ext] += 1

            if p.suffix.lower() in IMAGE_EXTENSIONS:
                direct_image_files += 1

            if len(samples) < MAX_SAMPLE_FILES_PER_DIR:
                samples.append(name)

        try:
            depth = len(cp.relative_to(data_root).parts)
        except Exception:
            depth = -1

        directory_rows.append({
            "relative_dir": rel(cp, data_root),
            "depth": depth,
            "role_hint": dir_role_hint(cp),
            "dataset_hint": detect_dataset_hint(cp),
            "direct_subdirs": len(dirs),
            "direct_files": len(files),
            "direct_image_like_files": direct_image_files,
            "direct_bytes": direct_bytes,
            "direct_size_human": human_size(direct_bytes),
            "sample_files": ";".join(samples),
        })

    directory_rows.sort(
        key=lambda r: (
            int(r["depth"]),
            str(r["relative_dir"]).lower(),
        )
    )

    extension_rows = [
        {
            "extension": ext,
            "file_count": count,
        }
        for ext, count in sorted(
            extension_counter.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    return (
        directory_rows,
        extension_rows,
        total_files,
        total_bytes,
    )


def find_image_mask_pairs(
    data_root: Path,
    directory_rows: Sequence[dict],
):
    path_rows = []

    for row in directory_rows:
        if row["relative_dir"] == ".":
            p = data_root
        else:
            p = data_root / row["relative_dir"]
        path_rows.append((p.resolve(), row))

    by_parent = defaultdict(list)
    for p, row in path_rows:
        by_parent[p.parent].append((p, row))

    pairs = []

    for parent, children in sorted(
        by_parent.items(),
        key=lambda kv: str(kv[0]).lower(),
    ):
        image_dirs = []
        mask_dirs = []

        for p, row in children:
            role = row["role_hint"]
            if role in {"IMAGE_DIR", "IMAGE_LIKE_DIR"}:
                image_dirs.append(p)
            if role in {"MASK_DIR", "MASK_LIKE_DIR"}:
                mask_dirs.append(p)

        for image_dir in image_dirs:
            for mask_dir in mask_dirs:
                pairs.append({
                    "parent_dir": rel(parent, data_root),
                    "image_dir": rel(image_dir, data_root),
                    "mask_dir": rel(mask_dir, data_root),
                    "dataset_hint": detect_dataset_hint(parent),
                })

    pairs.sort(
        key=lambda r: (
            r["parent_dir"].lower(),
            r["image_dir"].lower(),
            r["mask_dir"].lower(),
        )
    )
    return pairs


def build_terminal_summary(
    data_root: Path,
    directory_rows: Sequence[dict],
    extension_rows: Sequence[dict],
    pairs: Sequence[dict],
    total_files: int,
    total_bytes: int,
) -> str:
    top_level = [
        r for r in directory_rows
        if int(r["depth"]) == 1
    ]

    interesting_dirs = [
        r for r in directory_rows
        if r["role_hint"] in {
            "IMAGE_DIR",
            "IMAGE_LIKE_DIR",
            "MASK_DIR",
            "MASK_LIKE_DIR",
            "TRAIN_LIKE_DIR",
            "VAL_LIKE_DIR",
            "TEST_LIKE_DIR",
        }
    ]

    dataset_hints = sorted({
        h
        for r in directory_rows
        for h in r["dataset_hint"].split(";")
        if h
    })

    lines = [
        "===== Q1-R02R DATA DIRECTORY INVENTORY =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        f"Data root: {data_root}",
        f"Total directories: {len(directory_rows)}",
        f"Total files: {total_files}",
        f"Total size: {human_size(total_bytes)}",
        "",
        "Top-level directories:",
    ]

    if top_level:
        for r in top_level:
            lines.append(
                f"  - {r['relative_dir']} | "
                f"files={r['direct_files']} | "
                f"subdirs={r['direct_subdirs']} | "
                f"size={r['direct_size_human']} | "
                f"dataset_hint={r['dataset_hint'] or '-'}"
            )
    else:
        lines.append("  - NONE")

    lines += ["", "Detected dataset hints:"]
    if dataset_hints:
        for h in dataset_hints:
            lines.append(f"  - {h}")
    else:
        lines.append("  - NONE")

    lines += ["", "Interesting train/val/test/image/mask directories:"]
    if interesting_dirs:
        for r in interesting_dirs[:120]:
            lines.append(
                f"  - {r['relative_dir']} | "
                f"role={r['role_hint']} | "
                f"files={r['direct_files']} | "
                f"image_like={r['direct_image_like_files']} | "
                f"size={r['direct_size_human']} | "
                f"dataset={r['dataset_hint'] or '-'}"
            )
        if len(interesting_dirs) > 120:
            lines.append(
                f"  ... ({len(interesting_dirs) - 120} more; see CSV)"
            )
    else:
        lines.append("  - NONE")

    lines += ["", "Likely sibling image+mask pairs:"]
    if pairs:
        for p in pairs[:80]:
            lines.append(
                f"  - parent={p['parent_dir']} | "
                f"images={p['image_dir']} | "
                f"masks={p['mask_dir']} | "
                f"dataset={p['dataset_hint'] or '-'}"
            )
        if len(pairs) > 80:
            lines.append(
                f"  ... ({len(pairs) - 80} more; see CSV)"
            )
    else:
        lines.append("  - NONE")

    lines += ["", "Top file extensions:"]
    for r in extension_rows[:25]:
        lines.append(
            f"  - {r['extension']}: {r['file_count']}"
        )

    lines += [
        "",
        "Image content opened=NO",
        "Files modified inside data/=NO",
        "Training=NO",
        "Inference=NO",
        "",
        "[OK] DATA_DIRECTORY_INVENTORY_COMPLETE",
    ]

    return "\n".join(lines) + "\n"


def run(args):
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.exists():
        raise FileNotFoundError(data_root)

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial output exists: {build_dir}"
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    (
        directory_rows,
        extension_rows,
        total_files,
        total_bytes,
    ) = scan_data_root(data_root)

    pairs = find_image_mask_pairs(
        data_root,
        directory_rows,
    )

    dir_csv = build_dir / "directory_inventory.csv"
    write_csv(
        dir_csv,
        directory_rows,
        [
            "relative_dir",
            "depth",
            "role_hint",
            "dataset_hint",
            "direct_subdirs",
            "direct_files",
            "direct_image_like_files",
            "direct_bytes",
            "direct_size_human",
            "sample_files",
        ],
    )

    ext_csv = build_dir / "extension_counts.csv"
    write_csv(
        ext_csv,
        extension_rows,
        ["extension", "file_count"],
    )

    pair_csv = build_dir / "likely_image_mask_pairs.csv"
    write_csv(
        pair_csv,
        pairs,
        [
            "parent_dir",
            "image_dir",
            "mask_dir",
            "dataset_hint",
        ],
    )

    summary = {
        "script_version": VERSION,
        "build": BUILD,
        "data_root": str(data_root),
        "total_directories": len(directory_rows),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_size_human": human_size(total_bytes),
        "likely_image_mask_pairs": len(pairs),
        "image_content_opened": False,
        "files_modified_inside_data": False,
        "training": False,
        "inference": False,
    }
    summary_json = build_dir / "inventory_summary.json"
    write_json(summary_json, summary)

    terminal_summary = build_terminal_summary(
        data_root,
        directory_rows,
        extension_rows,
        pairs,
        total_files,
        total_bytes,
    )

    run_log = build_dir / "run_log.txt"
    run_log.write_text(
        terminal_summary,
        encoding="utf-8",
    )

    artifacts = {
        "directory_inventory": dir_csv,
        "extension_counts": ext_csv,
        "likely_image_mask_pairs": pair_csv,
        "inventory_summary": summary_json,
        "run_log": run_log,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "data_root": str(data_root),
        "image_content_opened": False,
        "files_modified_inside_data": False,
        "training": False,
        "inference": False,
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
            for name, path in artifacts.items()
        },
        "decision": "DATA_DIRECTORY_INVENTORY_COMPLETE",
    }

    lock_path = (
        build_dir
        / "Q1_R02R_DATA_DIRECTORY_INVENTORY_LOCK.json"
    )
    write_json(lock_path, lock)

    build_dir.rename(output_dir)

    print(
        (output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(f"Output directory: {output_dir}")
    print(
        "Lock: "
        f"{output_dir / 'Q1_R02R_DATA_DIRECTORY_INVENTORY_LOCK.json'}"
    )


def self_test():
    assert normalize_stem("abc_mask") == "abc"
    assert normalize_stem("abc-GT") == "abc"

    assert (
        detect_dataset_hint(
            Path(r"F:\x\Kvasir-SEG\images")
        )
        == "Kvasir-SEG"
    )
    assert (
        detect_dataset_hint(
            Path(r"F:\x\PolypGen\images")
        )
        == "PolypGen"
    )

    assert dir_role_hint(Path("images")) == "IMAGE_DIR"
    assert dir_role_hint(Path("masks")) == "MASK_DIR"
    assert dir_role_hint(Path("train")) == "TRAIN_LIKE_DIR"

    print("NORMALIZE_STEM_TEST_PASS")
    print("DATASET_HINT_TEST_PASS")
    print("DIRECTORY_ROLE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inventory F:\\MEDSEG_SAFETTA\\data without opening image "
            "contents or modifying data files."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run implementation tests only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
