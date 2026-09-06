#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from tqdm import tqdm

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA_ROOT = ROOT / "data" / "external" / "SUN_SEG"
DEFAULT_ANN = DATA_ROOT / "SUN-SEG-Annotation-v2.zip"
DEFAULT_RAW = DATA_ROOT / "SUN-SEG-FinalData-v20251212.tar.gz"
DEFAULT_OUT = ROOT / "outputs" / "Q1_R14B1_sunseg_dual_archive_asset_inventory_fix1_v1"

DECISION = "SUNSEG_DUAL_ARCHIVE_ASSET_INVENTORY_LOCKED_PRE_EXTRACTION"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_TOKENS = ("mask", "masks", "gt", "groundtruth", "ground_truth",
               "annotation", "annotations", "label", "labels")
SPLIT_TOKENS = ("train", "test", "easy", "hard", "seen", "unseen",
                "positive", "negative", "clip", "case", "video", "frame")


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=total, unit="B", unit_scale=True, dynamic_ncols=True,
        desc=f"SHA256 {path.name}"
    ) as bar:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def normalize(name: str):
    s = name.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def parts(name: str):
    return [p for p in normalize(name).split("/") if p]


def unsafe(name: str):
    s = normalize(name)
    p = PurePosixPath(s)
    traversal = any(x == ".." for x in p.parts)
    absolute = s.startswith("/") or (len(s) >= 2 and s[1] == ":")
    return traversal, absolute


def suffix(name: str):
    if name.endswith("/"):
        return "<DIR>"
    p = PurePosixPath(name)
    ss = [x.lower() for x in p.suffixes]
    if len(ss) >= 2 and ss[-2:] in ([".tar", ".gz"], [".nii", ".gz"]):
        return "".join(ss[-2:])
    return ss[-1] if ss else "<NO_EXT>"


def lev(ps, idx):
    return ps[idx] if len(ps) > idx else "<NONE>"


def token_hits(name: str):
    low = normalize(name).lower()
    mh = ",".join([t for t in MASK_TOKENS if t in low])
    sh = ",".join([t for t in SPLIT_TOKENS if t in low])
    return mh, sh


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect_common(name, size, is_dir, idx):
    n = normalize(name)
    ps = parts(n)
    tr, ab = unsafe(n)
    mh, sh = token_hits(n)
    return {
        "member_index": idx,
        "member_name": n,
        "is_directory": int(is_dir),
        "extension": suffix(n),
        "level1": lev(ps, 0),
        "level2": lev(ps, 1),
        "level3": lev(ps, 2),
        "file_size": int(size),
        "mask_token_hits": mh,
        "split_token_hits": sh,
        "unsafe_traversal": int(tr),
        "unsafe_absolute": int(ab),
    }


def inventory_zip(path: Path):
    rows = []
    ext = Counter()
    l1 = Counter()
    l2 = Counter()
    l3 = Counter()
    names = Counter()
    files = dirs = traversal = absolute = 0
    total = 0

    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        for i, info in enumerate(tqdm(infos, desc="Inventory annotation ZIP",
                                      unit="member", dynamic_ncols=True)):
            r = collect_common(info.filename, info.file_size,
                               info.is_dir() or info.filename.endswith("/"), i)
            rows.append(r)
            ext[r["extension"]] += 1
            l1[r["level1"]] += 1
            l2[(r["level1"], r["level2"])] += 1
            l3[(r["level1"], r["level2"], r["level3"])] += 1
            names[r["member_name"].lower()] += 1
            files += int(not r["is_directory"])
            dirs += int(r["is_directory"])
            traversal += r["unsafe_traversal"]
            absolute += r["unsafe_absolute"]
            total += r["file_size"]

    return {
        "rows": rows, "ext": ext, "l1": l1, "l2": l2, "l3": l3,
        "files": files, "dirs": dirs, "traversal": traversal,
        "absolute": absolute, "total": total,
        "duplicates": {k:v for k,v in names.items() if v > 1},
    }


def inventory_tar(path: Path):
    rows = []
    ext = Counter()
    l1 = Counter()
    l2 = Counter()
    l3 = Counter()
    names = Counter()
    files = dirs = traversal = absolute = 0
    total = 0

    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        for i, info in enumerate(tqdm(members, desc="Inventory SUN raw TAR.GZ",
                                      unit="member", dynamic_ncols=True)):
            r = collect_common(info.name, info.size if info.isfile() else 0,
                               info.isdir(), i)
            rows.append(r)
            ext[r["extension"]] += 1
            l1[r["level1"]] += 1
            l2[(r["level1"], r["level2"])] += 1
            l3[(r["level1"], r["level2"], r["level3"])] += 1
            names[r["member_name"].lower()] += 1
            files += int(info.isfile())
            dirs += int(info.isdir())
            traversal += r["unsafe_traversal"]
            absolute += r["unsafe_absolute"]
            total += r["file_size"]

    return {
        "rows": rows, "ext": ext, "l1": l1, "l2": l2, "l3": l3,
        "files": files, "dirs": dirs, "traversal": traversal,
        "absolute": absolute, "total": total,
        "duplicates": {k:v for k,v in names.items() if v > 1},
    }


def print_counter(title, counter, n):
    print(f"\n===== {title} =====")
    for k, v in counter.most_common(n):
        print(("/".join(k) if isinstance(k, tuple) else str(k)) + f": {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-zip", type=Path, default=DEFAULT_ANN)
    ap.add_argument("--raw-tar", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("===== Q1 R14B1 SUN-SEG DUAL-ARCHIVE ASSET INVENTORY =====")
    print("annotation:", args.annotation_zip)
    print("raw:", args.raw_tar)
    print("extract=NO")
    print("GT_PIXEL_DECODE=NO")
    print("MODEL_INFERENCE=NO")
    print("SUBSET_SELECTION=NO")
    print("CONTAMINATION_DECISION=NO")

    for p in (args.annotation_zip, args.raw_tar):
        if not p.exists():
            raise FileNotFoundError(p)
        if not p.is_file():
            raise RuntimeError(f"Not a regular file: {p}")

    ann_sha = sha256_file(args.annotation_zip)
    raw_sha = sha256_file(args.raw_tar)

    ann = inventory_zip(args.annotation_zip)
    raw = inventory_tar(args.raw_tar)

    for label, inv in (("annotation", ann), ("raw", raw)):
        if inv["traversal"] or inv["absolute"]:
            raise RuntimeError(
                f"Unsafe {label} archive paths: traversal={inv['traversal']} "
                f"absolute={inv['absolute']}"
            )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    fields = [
        "member_index", "member_name", "is_directory", "extension",
        "level1", "level2", "level3", "file_size", "mask_token_hits",
        "split_token_hits", "unsafe_traversal", "unsafe_absolute",
    ]

    ann_csv = args.output_dir / "R14B1_annotation_zip_members.csv"
    raw_csv = args.output_dir / "R14B1_raw_tar_members.csv"
    write_csv(ann_csv, ann["rows"], fields)
    write_csv(raw_csv, raw["rows"], fields)

    summaries = []
    for kind, p, sha, inv in (
        ("annotation_zip", args.annotation_zip, ann_sha, ann),
        ("raw_tar_gz", args.raw_tar, raw_sha, raw),
    ):
        summaries.append({
            "archive_kind": kind,
            "archive_name": p.name,
            "archive_path": str(p),
            "archive_sha256": sha,
            "archive_size_bytes": int(p.stat().st_size),
            "member_count": len(inv["rows"]),
            "file_count": inv["files"],
            "directory_count": inv["dirs"],
            "total_uncompressed_file_bytes": inv["total"],
            "unsafe_traversal_count": inv["traversal"],
            "unsafe_absolute_count": inv["absolute"],
            "duplicate_normalized_name_count": len(inv["duplicates"]),
            "image_extension_member_count": sum(
                v for k, v in inv["ext"].items() if k in IMAGE_EXTS
            ),
        })

    summary_csv = args.output_dir / "R14B1_archive_summary.csv"
    write_csv(summary_csv, summaries, list(summaries[0].keys()))

    print("\n===== ARCHIVE SUMMARY =====")
    for s in summaries:
        print(
            f"{s['archive_kind']}: "
            f"sizeGB={s['archive_size_bytes']/1024**3:.3f} "
            f"members={s['member_count']} files={s['file_count']} "
            f"dirs={s['directory_count']} "
            f"image_ext_members={s['image_extension_member_count']} "
            f"duplicate_names={s['duplicate_normalized_name_count']}"
        )
        print("  SHA256=", s["archive_sha256"])

    print_counter("RAW EXTENSIONS", raw["ext"], 40)
    print_counter("RAW LEVEL1", raw["l1"], 40)
    print_counter("RAW LEVEL2", raw["l2"], 80)
    print_counter("RAW LEVEL3", raw["l3"], 120)
    print_counter("ANNOTATION EXTENSIONS", ann["ext"], 40)
    print_counter("ANNOTATION LEVEL1", ann["l1"], 40)
    print_counter("ANNOTATION LEVEL2", ann["l2"], 80)
    print_counter("ANNOTATION LEVEL3", ann["l3"], 120)

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "annotation_archive": summaries[0],
        "raw_archive": summaries[1],
        "information_boundary": {
            "archive_extracted": False,
            "annotation_pixels_decoded": False,
            "gt_pixels_read": False,
            "model_inference": False,
            "subset_selection": False,
            "contamination_decision": False,
            "downstream_outcomes_read": False,
        },
        "next_stage": (
            "R14B2_CONTROLLED_EXTRACTION_THEN_CASE_CLIP_FRAME_IDENTITY_"
            "AND_CONTAMINATION_AUDIT_BEFORE_INFERENCE"
        ),
    }

    lock_path = args.output_dir / "R14B1_SUNSEG_DUAL_ARCHIVE_ASSET_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print("\nDecision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
