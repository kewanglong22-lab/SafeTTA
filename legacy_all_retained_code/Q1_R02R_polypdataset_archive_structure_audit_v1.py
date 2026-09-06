#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R02R — PolypDataset.zip structure audit

Reads ZIP member names and metadata only.

NO extraction.
NO image decoding.
NO modification of archive/data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02R-polypdataset-structure-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
ARCHIVE = ROOT / "data" / "downloads" / "PolypDataset.zip"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R02R_polypdataset_archive_structure_audit_v1"

EXPECTED_ARCHIVE_SHA256 = (
    "27f609694d50240782673ddd1e480ea972c674d42706538c80d4a3ee637091ce"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

DATASET_TOKENS = {
    "Kvasir-SEG": ("kvasir",),
    "CVC-ClinicDB": ("clinicdb", "cvc-clinicdb", "cvc_clinicdb"),
    "CVC-ColonDB": ("colondb", "cvc-colondb", "cvc_colondb"),
    "CVC-300": ("cvc-300", "cvc300", "cvc_300"),
    "ETIS": ("etis",),
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def detect_dataset(text: str) -> str:
    low = text.replace("\\", "/").lower()
    hits = []
    for name, tokens in DATASET_TOKENS.items():
        if any(token in low for token in tokens):
            hits.append(name)
    return ";".join(hits)


def detect_role(text: str) -> str:
    low = text.replace("\\", "/").lower()

    # Strong augmentation signals only; don't use bare "da" because it
    # falsely matches "dataset".
    if (
        "newtrimage" in low
        or "newtrmask" in low
        or "augmentation" in low
        or "augmented" in low
        or re.search(r"(^|[/_.-])aug($|[/_.-])", low)
        or re.search(r"(^|[/_.-])da($|[/_.-])", low)
    ):
        return "AUGMENTED_OR_DA"

    if "testdataset" in low or re.search(r"(^|[/_.-])test($|[/_.-])", low):
        return "TEST_LIKE"

    if (
        "validation" in low
        or re.search(r"(^|[/_.-])val(id)?($|[/_.-])", low)
    ):
        return "VAL_LIKE"

    if (
        "traindataset" in low
        or "training" in low
        or re.search(r"(^|[/_.-])train($|[/_.-])", low)
    ):
        return "TRAIN_LIKE"

    return ""


def leaf_kind(text: str) -> str:
    leaf = text.replace("\\", "/").rstrip("/").split("/")[-1].lower()

    if "newtrimage" in leaf:
        return "IMAGE_LIKE"
    if "newtrmask" in leaf:
        return "MASK_LIKE"

    if leaf in {"image", "images", "img", "imgs"}:
        return "IMAGE_LIKE"
    if leaf in {"mask", "masks", "gt", "gts", "label", "labels"}:
        return "MASK_LIKE"

    if "image" in leaf and "mask" not in leaf:
        return "IMAGE_LIKE"
    if "mask" in leaf or "label" in leaf:
        return "MASK_LIKE"

    return ""


def inspect_archive(path: Path):
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()

        prefix_stats = defaultdict(
            lambda: {
                "file_count": 0,
                "image_file_count": 0,
                "bytes": 0,
                "extensions": Counter(),
            }
        )
        member_rows = []

        for info in tqdm(
            infos,
            desc="Auditing PolypDataset.zip members",
            unit="member",
            dynamic_ncols=True,
        ):
            name = info.filename.replace("\\", "/").strip("/")
            if not name or info.is_dir():
                continue

            parts = name.split("/")
            ext = Path(name).suffix.lower() or "<no_extension>"

            member_rows.append({
                "member": name,
                "top_level": parts[0],
                "second_level": "/".join(parts[:2]) if len(parts) >= 2 else "",
                "extension": ext,
                "bytes": info.file_size,
                "dataset_hint": detect_dataset(name),
                "role_hint": detect_role(name),
            })

            for depth in range(1, len(parts)):
                prefix = "/".join(parts[:depth])
                s = prefix_stats[prefix]
                s["file_count"] += 1
                s["bytes"] += info.file_size
                s["extensions"][ext] += 1
                if ext in IMAGE_EXTENSIONS:
                    s["image_file_count"] += 1

        directory_rows = []
        for prefix, s in prefix_stats.items():
            parts = prefix.split("/")
            directory_rows.append({
                "path": prefix,
                "depth": len(parts),
                "parent": "/".join(parts[:-1]),
                "leaf": parts[-1],
                "leaf_kind": leaf_kind(prefix),
                "dataset_hint": detect_dataset(prefix),
                "role_hint": detect_role(prefix),
                "file_count_recursive": s["file_count"],
                "image_files_recursive": s["image_file_count"],
                "png_count_recursive": s["extensions"].get(".png", 0),
                "jpg_count_recursive": (
                    s["extensions"].get(".jpg", 0)
                    + s["extensions"].get(".jpeg", 0)
                ),
                "uncompressed_bytes_recursive": s["bytes"],
            })

        directory_rows.sort(
            key=lambda r: (int(r["depth"]), r["path"].lower())
        )

        by_parent = defaultdict(list)
        for row in directory_rows:
            if row["leaf_kind"] in {"IMAGE_LIKE", "MASK_LIKE"}:
                by_parent[row["parent"]].append(row)

        pair_rows = []
        for parent, children in sorted(by_parent.items()):
            images = [r for r in children if r["leaf_kind"] == "IMAGE_LIKE"]
            masks = [r for r in children if r["leaf_kind"] == "MASK_LIKE"]

            for image_row in images:
                for mask_row in masks:
                    pair_rows.append({
                        "parent": parent,
                        "image_dir": image_row["path"],
                        "mask_dir": mask_row["path"],
                        "image_files": image_row["image_files_recursive"],
                        "mask_files": mask_row["image_files_recursive"],
                        "counts_equal": (
                            image_row["image_files_recursive"]
                            == mask_row["image_files_recursive"]
                        ),
                        "dataset_hint": detect_dataset(parent),
                        "role_hint": detect_role(
                            parent + "/" + image_row["leaf"] + "/" + mask_row["leaf"]
                        ),
                    })

        top_counter = Counter(r["top_level"] for r in member_rows)
        second_counter = Counter(
            r["second_level"] for r in member_rows if r["second_level"]
        )

        top_rows = [
            {
                "path": name,
                "file_count": count,
                "leaf_kind": leaf_kind(name),
                "role_hint": detect_role(name),
                "dataset_hint": detect_dataset(name),
            }
            for name, count in top_counter.most_common()
        ]

        second_rows = [
            {
                "path": name,
                "file_count": count,
                "leaf_kind": leaf_kind(name),
                "role_hint": detect_role(name),
                "dataset_hint": detect_dataset(name),
            }
            for name, count in second_counter.most_common()
        ]

        return member_rows, directory_rows, pair_rows, top_rows, second_rows


def run(args):
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()

    if not archive.exists():
        raise FileNotFoundError(archive)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    actual_sha = sha256(archive)
    if actual_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "PolypDataset.zip SHA mismatch: "
            f"expected={EXPECTED_ARCHIVE_SHA256} actual={actual_sha}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    (
        members,
        directories,
        pairs,
        top_rows,
        second_rows,
    ) = inspect_archive(archive)

    write_csv(
        output_dir / "top_level_summary.csv",
        top_rows,
        ["path", "file_count", "leaf_kind", "role_hint", "dataset_hint"],
    )
    write_csv(
        output_dir / "second_level_summary.csv",
        second_rows,
        ["path", "file_count", "leaf_kind", "role_hint", "dataset_hint"],
    )
    write_csv(
        output_dir / "directory_recursive_summary.csv",
        directories,
        [
            "path", "depth", "parent", "leaf", "leaf_kind",
            "dataset_hint", "role_hint", "file_count_recursive",
            "image_files_recursive", "png_count_recursive",
            "jpg_count_recursive", "uncompressed_bytes_recursive",
        ],
    )
    write_csv(
        output_dir / "likely_image_mask_pairs_inside_zip.csv",
        pairs,
        [
            "parent", "image_dir", "mask_dir", "image_files",
            "mask_files", "counts_equal", "dataset_hint", "role_hint",
        ],
    )

    interesting = [
        row for row in directories
        if (
            row["depth"] <= 3
            and (
                row["leaf_kind"]
                or row["dataset_hint"]
                or row["role_hint"]
            )
        )
    ]

    lines = [
        "===== Q1-R02R POLYPDATASET.ZIP STRUCTURE AUDIT =====",
        f"Script version: {VERSION}",
        "",
        f"Archive: {archive}",
        f"SHA256: {actual_sha}",
        f"Files: {len(members)}",
        "",
        "Top-level contents:",
    ]

    for row in top_rows:
        lines.append(
            f"  - {row['path']} | files={row['file_count']} | "
            f"kind={row['leaf_kind'] or '-'} | "
            f"role={row['role_hint'] or '-'} | "
            f"dataset={row['dataset_hint'] or '-'}"
        )

    lines += ["", "Interesting directories (depth <= 3):"]
    for row in interesting[:160]:
        lines.append(
            f"  - {row['path']} | depth={row['depth']} | "
            f"files={row['file_count_recursive']} | "
            f"images={row['image_files_recursive']} | "
            f"kind={row['leaf_kind'] or '-'} | "
            f"role={row['role_hint'] or '-'} | "
            f"dataset={row['dataset_hint'] or '-'}"
        )

    lines += ["", "Likely image/mask pairs inside ZIP:"]
    if pairs:
        for row in pairs[:100]:
            lines.append(
                f"  - parent={row['parent'] or '<root>'} | "
                f"images={row['image_dir']} ({row['image_files']}) | "
                f"masks={row['mask_dir']} ({row['mask_files']}) | "
                f"equal={row['counts_equal']} | "
                f"role={row['role_hint'] or '-'} | "
                f"dataset={row['dataset_hint'] or '-'}"
            )
    else:
        lines.append("  - NONE")

    lines += [
        "",
        "Extraction=NO",
        "Image content opened=NO",
        "ZIP modified=NO",
        "",
        "[OK] POLYPDATASET_ARCHIVE_STRUCTURE_AUDIT_COMPLETE",
    ]

    log = "\n".join(lines) + "\n"
    (output_dir / "run_log.txt").write_text(log, encoding="utf-8")

    write_json(
        output_dir / "audit_summary.json",
        {
            "script_version": VERSION,
            "archive": str(archive),
            "archive_sha256": actual_sha,
            "files": len(members),
            "top_level_entries": len(top_rows),
            "directory_prefixes": len(directories),
            "likely_image_mask_pairs": len(pairs),
            "extraction": False,
            "image_content_opened": False,
            "zip_modified": False,
        },
    )

    print(log)
    print(f"Output directory: {output_dir}")


def self_test():
    assert leaf_kind("NewTRimageDA") == "IMAGE_LIKE"
    assert leaf_kind("NewTRmaskDA") == "MASK_LIKE"
    assert detect_dataset("TestDataset/Kvasir/images") == "Kvasir-SEG"
    assert detect_role("NewTRimageDA") == "AUGMENTED_OR_DA"
    assert detect_role("TestDataset/Kvasir") == "TEST_LIKE"
    assert detect_role("TrainDataset/images") == "TRAIN_LIKE"
    print("NEWTR_ROLE_TEST_PASS")
    print("DATASET_HINT_TEST_PASS")
    print("ROLE_CLASSIFICATION_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit PolypDataset.zip structure without extraction or image decoding."
        )
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
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
