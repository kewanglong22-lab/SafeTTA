#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R02R — archive inventory under F:\\MEDSEG_SAFETTA\\data

Lists ZIP archive metadata and member names only.

NO extraction.
NO image decoding.
NO file modification inside data/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02R-archive-inventory-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "Q1_R02R_download_archive_inventory_v1"

DATASET_TOKENS = {
    "Kvasir-SEG": ("kvasir-seg", "kvasir_seg", "kvasirseg", "kvasir"),
    "CVC-ClinicDB": ("cvc-clinicdb", "cvc_clinicdb", "clinicdb"),
    "CVC-ColonDB": ("cvc-colondb", "cvc_colondb", "colondb", "colon-db"),
    "CVC-300": ("cvc-300", "cvc_300", "cvc300"),
    "ETIS": ("etis-larib", "etis_larib", "etis"),
    "PolypGen": ("polypgen",),
    "TrainDataset": ("traindataset", "train_dataset"),
    "TestDataset": ("testdataset", "test_dataset"),
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human_size(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or u == "TB":
            return f"{x:.2f} {u}"
        x /= 1024
    return str(n)


def detect_hints(text: str) -> str:
    low = text.replace("\\", "/").lower()
    hits = []
    for name, tokens in DATASET_TOKENS.items():
        if any(t in low for t in tokens):
            hits.append(name)
    return ";".join(hits)


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def inspect_zip(path: Path):
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()

        ext_counts = Counter()
        top_counts = Counter()
        hints = set()
        sample_members = []

        file_count = 0
        dir_count = 0
        uncompressed = 0
        compressed = 0

        for info in infos:
            name = info.filename.replace("\\", "/")

            if info.is_dir():
                dir_count += 1
                continue

            file_count += 1
            uncompressed += info.file_size
            compressed += info.compress_size

            suffix = Path(name).suffix.lower() or "<no_extension>"
            ext_counts[suffix] += 1

            top = name.split("/", 1)[0] if "/" in name else "<root>"
            top_counts[top] += 1

            h = detect_hints(name)
            for item in h.split(";"):
                if item:
                    hints.add(item)

            if len(sample_members) < 30:
                sample_members.append(name)

        top_rows = [
            {"top_level": k, "file_count": v}
            for k, v in top_counts.most_common()
        ]

        ext_rows = [
            {"extension": k, "file_count": v}
            for k, v in ext_counts.most_common()
        ]

        return {
            "archive": str(path),
            "archive_bytes": path.stat().st_size,
            "archive_size": human_size(path.stat().st_size),
            "file_members": file_count,
            "directory_members": dir_count,
            "uncompressed_bytes": uncompressed,
            "uncompressed_size": human_size(uncompressed),
            "compressed_member_bytes": compressed,
            "dataset_hints": sorted(hints),
            "sample_members": sample_members,
            "top_levels": top_rows,
            "extensions": ext_rows,
        }


def run(args):
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.exists():
        raise FileNotFoundError(data_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    archives = sorted(
        [
            p.resolve()
            for p in data_root.rglob("*.zip")
            if p.is_file()
        ],
        key=lambda p: str(p).lower(),
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    all_top = []
    all_ext = []

    for p in tqdm(
        archives,
        desc="Inspecting ZIP archives",
        unit="archive",
        dynamic_ncols=True,
    ):
        try:
            info = inspect_zip(p)
            status = "OK"
            error = ""
        except Exception as e:
            info = {
                "archive": str(p),
                "archive_bytes": p.stat().st_size,
                "archive_size": human_size(p.stat().st_size),
                "file_members": 0,
                "directory_members": 0,
                "uncompressed_bytes": 0,
                "uncompressed_size": "0 B",
                "compressed_member_bytes": 0,
                "dataset_hints": [],
                "sample_members": [],
                "top_levels": [],
                "extensions": [],
            }
            status = "ERROR"
            error = f"{type(e).__name__}: {e}"

        archive_id = f"A{len(summaries)+1:02d}"

        summaries.append({
            "archive_id": archive_id,
            "archive": info["archive"],
            "archive_size": info["archive_size"],
            "archive_bytes": info["archive_bytes"],
            "status": status,
            "error": error,
            "file_members": info["file_members"],
            "directory_members": info["directory_members"],
            "uncompressed_size": info["uncompressed_size"],
            "dataset_hints": ";".join(info["dataset_hints"]),
            "sample_members": ";".join(info["sample_members"]),
            "sha256": sha256(p),
        })

        for r in info["top_levels"]:
            all_top.append({
                "archive_id": archive_id,
                **r,
            })

        for r in info["extensions"]:
            all_ext.append({
                "archive_id": archive_id,
                **r,
            })

    write_csv(
        output_dir / "archive_summary.csv",
        summaries,
        [
            "archive_id",
            "archive",
            "archive_size",
            "archive_bytes",
            "status",
            "error",
            "file_members",
            "directory_members",
            "uncompressed_size",
            "dataset_hints",
            "sample_members",
            "sha256",
        ],
    )

    write_csv(
        output_dir / "archive_top_level_contents.csv",
        all_top,
        ["archive_id", "top_level", "file_count"],
    )

    write_csv(
        output_dir / "archive_extension_counts.csv",
        all_ext,
        ["archive_id", "extension", "file_count"],
    )

    lines = [
        "===== Q1-R02R DOWNLOAD ARCHIVE INVENTORY =====",
        f"Script version: {VERSION}",
        "",
        f"Data root: {data_root}",
        f"ZIP archives found: {len(archives)}",
        "",
    ]

    for r in summaries:
        lines += [
            f"{r['archive_id']}: {r['archive']}",
            f"  archive size={r['archive_size']}",
            f"  status={r['status']}",
            f"  files={r['file_members']}",
            f"  dirs={r['directory_members']}",
            f"  uncompressed={r['uncompressed_size']}",
            f"  dataset hints={r['dataset_hints'] or '-'}",
            f"  sha256={r['sha256']}",
            "  sample members:",
        ]
        samples = [s for s in r["sample_members"].split(";") if s]
        for s in samples[:15]:
            lines.append(f"    - {s}")
        if not samples:
            lines.append("    - NONE")
        lines.append("")

    lines += [
        "Extraction=NO",
        "Image content opened=NO",
        "Files modified inside data/=NO",
        "",
        "[OK] ARCHIVE_INVENTORY_COMPLETE",
    ]

    log = "\n".join(lines) + "\n"
    (output_dir / "run_log.txt").write_text(log, encoding="utf-8")

    write_json(
        output_dir / "inventory_summary.json",
        {
            "script_version": VERSION,
            "data_root": str(data_root),
            "zip_archives_found": len(archives),
            "extraction": False,
            "image_content_opened": False,
            "data_modified": False,
        },
    )

    print(log)
    print(f"Output directory: {output_dir}")


def self_test():
    assert detect_hints("x/Kvasir-SEG/images/a.png") == "Kvasir-SEG"
    assert "PolypGen" in detect_hints("PolypGen/test/a.png")
    assert "TrainDataset" in detect_hints("TrainDataset/images/a.png")
    print("DATASET_HINT_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="List ZIP archive contents under MEDSEG_SAFETTA/data without extraction."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
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
