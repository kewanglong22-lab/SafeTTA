#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B2_sunseg_deep_schema_pairing_identity_audit_fix2.py

Supersedes R14B2 fix1 pairing counts because the official TAR contains macOS
AppleDouble/resource-fork members such as "._case..." that were incorrectly
counted as image/GT members by the first audit.

This fix:
- keeps the exact locked archive SHAs;
- performs NO extraction and NO GT pixel decoding;
- filters AppleDouble / .DS_Store / non-image metadata before pairing;
- requires exact Frame<->GT path pairing on real image members;
- cross-checks raw TAR GT keys against Annotation-v2 GT keys;
- reconstructs descriptive clip IDs and candidate physical case IDs from the
  official directory names (caseNN[_clip]);
- audits case overlap across Train / Easy-Seen / Easy-Unseen /
  Hard-Seen / Hard-Unseen;
- does NOT select a confirmatory cohort yet.

No model inference. No TTA. No outcome access. No contamination claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA = ROOT / "data" / "external" / "SUN_SEG"
RAW = DATA / "SUN-SEG-FinalData-v20251212.tar.gz"
ANN = DATA / "SUN-SEG-Annotation-v2.zip"
OUT = ROOT / "outputs" / "Q1_R14B2_sunseg_deep_schema_pairing_identity_audit_fix2_v1"

RAW_SHA = "d9a00fada04782937a144e9d1bc3c2a3b7f8e321e37ba1cf83bebdd490563016"
ANN_SHA = "4ce0324c38743aeb47188dc18eded33258b2da8c07b7392a457dc55e77a1c7e7"

DECISION = "SUNSEG_REAL_IMAGE_PAIRING_AND_CASE_IDENTITY_AUDITED_PRE_EXTRACTION"
EXPECTED_ANNOTATION_GT = 49136

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASETS = {"traindataset", "testeasydataset", "testharddataset"}
VIS = {"seen", "unseen"}
CATEGORIES = {"frame", "gt"}


def sha256_file(path: Path, chunk=16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=total, unit="B", unit_scale=True, dynamic_ncols=True,
        desc=f"SHA256 {path.name}"
    ) as bar:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def norm(name: str) -> str:
    s = name.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def parts(name: str):
    return [x for x in norm(name).split("/") if x]


def is_macos_artifact(name: str) -> bool:
    ps = parts(name)
    if not ps:
        return True
    for p in ps:
        low = p.lower()
        if p.startswith("._") or low == ".ds_store" or low == "__macosx":
            return True
    return False


def is_real_image_member(name: str) -> bool:
    if is_macos_artifact(name):
        return False
    return PurePosixPath(name).suffix.lower() in VALID_IMAGE_EXTS


def classify(name: str):
    ps = parts(name)
    low = [x.lower() for x in ps]

    dataset = next(
        (ps[i] for i, x in enumerate(low) if x in DATASETS),
        "<UNKNOWN>"
    )
    visibility = next(
        (ps[i] for i, x in enumerate(low) if x in VIS),
        "<NA>"
    )
    category = next(
        (ps[i] for i, x in enumerate(low) if x in CATEGORIES),
        "<UNKNOWN>"
    )
    return dataset, visibility, category


def relative_after_category(name: str, category: str):
    ps = parts(name)
    low = [x.lower() for x in ps]
    try:
        i = low.index(category.lower())
    except ValueError:
        return None
    return "/".join(ps[i + 1:])


def relative_pair_key(name: str, category: str):
    """
    Category-independent key rooted at official dataset/split:
      TestEasyDataset/Unseen/Frame/case3_1/foo.jpg
      TestEasyDataset/Unseen/GT/case3_1/foo.png
      -> testeasydataset/unseen/case3_1/foo
    """
    ps = parts(name)
    low = [x.lower() for x in ps]
    try:
        cidx = low.index(category.lower())
    except ValueError:
        return None

    dataset_idx = next((i for i, x in enumerate(low) if x in DATASETS), None)
    if dataset_idx is None:
        return None

    prefix = ps[dataset_idx:cidx]
    after = ps[cidx + 1:]
    if not after:
        return None
    after[-1] = str(PurePosixPath(after[-1]).with_suffix(""))
    return "/".join(prefix + after).lower()


CASE_RE = re.compile(r"^(case\d+)(?:_(\d+))?$", re.IGNORECASE)
FRAME_IDX_RE = re.compile(r"image(\d+)$", re.IGNORECASE)


def infer_identity(frame_name: str):
    rel = relative_after_category(frame_name, "frame")
    if rel is None:
        return None, None, None

    rparts = [x for x in rel.split("/") if x]
    if len(rparts) < 2:
        return None, None, None

    clip_dir = rparts[0]
    m = CASE_RE.match(clip_dir)
    if not m:
        physical_case = None
    else:
        physical_case = m.group(1).lower()

    stem = PurePosixPath(rparts[-1]).stem
    fm = FRAME_IDX_RE.search(stem)
    frame_index = int(fm.group(1)) if fm else None
    return clip_dir.lower(), physical_case, frame_index


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect_raw(path: Path):
    accepted = []
    excluded = Counter()

    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        for idx, m in enumerate(tqdm(
            members, desc="Audit real TAR image members",
            unit="member", dynamic_ncols=True
        )):
            if not m.isfile():
                excluded["not_file"] += 1
                continue

            name = norm(m.name)
            ds, vis, cat = classify(name)

            if cat.lower() not in CATEGORIES:
                excluded["not_frame_or_gt"] += 1
                continue
            if is_macos_artifact(name):
                excluded["macos_artifact"] += 1
                continue
            if PurePosixPath(name).suffix.lower() not in VALID_IMAGE_EXTS:
                excluded["non_image_extension"] += 1
                continue

            accepted.append({
                "member_index": idx,
                "member_name": name,
                "dataset": ds,
                "visibility": vis,
                "category": cat,
                "extension": PurePosixPath(name).suffix.lower(),
                "file_size": int(m.size),
            })

    return accepted, excluded


def collect_annotation_gt(path: Path):
    keys = set()
    excluded = Counter()
    names = []

    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        for info in tqdm(
            infos, desc="Audit Annotation-v2 real GT members",
            unit="member", dynamic_ncols=True
        ):
            if info.is_dir():
                excluded["directory"] += 1
                continue
            name = norm(info.filename)
            ds, vis, cat = classify(name)

            if cat.lower() != "gt":
                excluded["not_gt"] += 1
                continue
            if is_macos_artifact(name):
                excluded["macos_artifact"] += 1
                continue
            if PurePosixPath(name).suffix.lower() not in VALID_IMAGE_EXTS:
                excluded["non_image_extension"] += 1
                continue

            key = relative_pair_key(name, "gt")
            if key is None:
                excluded["unkeyable"] += 1
                continue
            keys.add(key)
            names.append(name)

    return keys, names, excluded


def pair_raw(rows):
    frames = {}
    gts = {}
    frame_dups = Counter()
    gt_dups = Counter()

    for r in rows:
        cat = r["category"].lower()
        key = relative_pair_key(r["member_name"], cat)
        if key is None:
            continue
        if cat == "frame":
            frame_dups[key] += 1
            frames.setdefault(key, r)
        elif cat == "gt":
            gt_dups[key] += 1
            gts.setdefault(key, r)

    fkeys, gkeys = set(frames), set(gts)
    common = sorted(fkeys & gkeys)

    out = []
    for key in common:
        f = frames[key]
        g = gts[key]
        clip_id, physical_case_id, frame_index = infer_identity(f["member_name"])
        out.append({
            "pair_key": key,
            "dataset": f["dataset"],
            "visibility": f["visibility"],
            "clip_id": clip_id,
            "physical_case_id": physical_case_id,
            "frame_index": frame_index if frame_index is not None else "",
            "frame_member": f["member_name"],
            "gt_member": g["member_name"],
            "frame_size": f["file_size"],
            "gt_size": g["file_size"],
        })

    return {
        "frames": frames,
        "gts": gts,
        "frame_keys": fkeys,
        "gt_keys": gkeys,
        "common": common,
        "rows": out,
        "duplicate_frame_keys": sum(v > 1 for v in frame_dups.values()),
        "duplicate_gt_keys": sum(v > 1 for v in gt_dups.values()),
    }


def split_name(dataset, visibility):
    if dataset == "TrainDataset":
        return "Train"
    return f"{dataset}-{visibility}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-tar", type=Path, default=RAW)
    ap.add_argument("--annotation-zip", type=Path, default=ANN)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()

    print("===== Q1 R14B2 FIX2 SUN-SEG REAL-IMAGE PAIRING / CASE IDENTITY AUDIT =====")
    print("Supersedes R14B2 fix1 pairing counts=YES")
    print("APPLEDOUBLE_FILTER=YES")
    print("EXTRACTION=NO")
    print("GT_PIXEL_DECODE=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("SUBSET_SELECTION=NO")
    print("CONTAMINATION_CLAIM=NO")

    for p in (args.raw_tar, args.annotation_zip):
        if not p.exists():
            raise FileNotFoundError(p)

    raw_sha = sha256_file(args.raw_tar)
    ann_sha = sha256_file(args.annotation_zip)
    if raw_sha != RAW_SHA:
        raise RuntimeError(f"RAW SHA mismatch: {raw_sha}")
    if ann_sha != ANN_SHA:
        raise RuntimeError(f"Annotation SHA mismatch: {ann_sha}")
    print("R14B1 SHA GATE=PASS")

    raw_rows, raw_excluded = collect_raw(args.raw_tar)
    ann_gt_keys, ann_gt_names, ann_excluded = collect_annotation_gt(args.annotation_zip)
    paired = pair_raw(raw_rows)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    frame_count = len(paired["frame_keys"])
    gt_count = len(paired["gt_keys"])
    pair_count = len(paired["common"])
    frame_only = len(paired["frame_keys"] - paired["gt_keys"])
    gt_only = len(paired["gt_keys"] - paired["frame_keys"])

    print("\n===== SANITIZED REAL-IMAGE PAIRING =====")
    print("raw real frame unique:", frame_count)
    print("raw real gt unique:", gt_count)
    print("raw paired:", pair_count)
    print("frame_only:", frame_only)
    print("gt_only:", gt_only)
    print("duplicate_frame_keys:", paired["duplicate_frame_keys"])
    print("duplicate_gt_keys:", paired["duplicate_gt_keys"])
    print("annotation_v2 real gt unique:", len(ann_gt_keys))

    # Hard gates: these use the independently downloaded Annotation-v2 count as
    # the reference and prevent silent continuation with AppleDouble metadata.
    if len(ann_gt_keys) != EXPECTED_ANNOTATION_GT:
        raise RuntimeError(
            f"Annotation-v2 GT count unexpected: {len(ann_gt_keys)} "
            f"!= {EXPECTED_ANNOTATION_GT}"
        )
    if frame_count != EXPECTED_ANNOTATION_GT:
        raise RuntimeError(f"Real Frame count {frame_count} != {EXPECTED_ANNOTATION_GT}")
    if gt_count != EXPECTED_ANNOTATION_GT:
        raise RuntimeError(f"Real GT count {gt_count} != {EXPECTED_ANNOTATION_GT}")
    if pair_count != EXPECTED_ANNOTATION_GT or frame_only or gt_only:
        raise RuntimeError(
            f"Pairing mismatch: paired={pair_count} frame_only={frame_only} gt_only={gt_only}"
        )
    if paired["duplicate_frame_keys"] or paired["duplicate_gt_keys"]:
        raise RuntimeError("Duplicate sanitized pair keys detected.")

    raw_gt_vs_ann_missing = paired["gt_keys"] - ann_gt_keys
    ann_vs_raw_missing = ann_gt_keys - paired["gt_keys"]
    print("\n===== RAW GT <-> ANNOTATION-V2 KEY CROSSCHECK =====")
    print("raw_gt_not_in_annotation:", len(raw_gt_vs_ann_missing))
    print("annotation_not_in_raw_gt:", len(ann_vs_raw_missing))
    if raw_gt_vs_ann_missing or ann_vs_raw_missing:
        raise RuntimeError(
            "Raw GT and Annotation-v2 GT canonical key sets are not identical."
        )

    # Split / clip / physical-case summaries.
    split_frames = Counter()
    split_clips = defaultdict(set)
    split_cases = defaultdict(set)
    bad_identity = []

    for r in paired["rows"]:
        s = split_name(r["dataset"], r["visibility"])
        split_frames[s] += 1
        if r["clip_id"]:
            split_clips[s].add(r["clip_id"])
        if r["physical_case_id"]:
            split_cases[s].add(r["physical_case_id"])
        else:
            bad_identity.append(r["pair_key"])

    print("\n===== SANITIZED COUNTS BY SPLIT =====")
    all_splits = sorted(split_frames)
    for s in all_splits:
        print(
            f"{s}: frames={split_frames[s]} "
            f"clips={len(split_clips[s])} "
            f"physical_cases={len(split_cases[s])}"
        )

    print("\n===== PHYSICAL-CASE ID PARSE =====")
    print("unparsed_pair_count:", len(bad_identity))
    if bad_identity:
        print("first_unparsed:", bad_identity[:20])
        raise RuntimeError("Some real Frame pairs lack parseable caseNN[_clip] parent identity.")

    case_union = set().union(*split_cases.values()) if split_cases else set()
    clip_union = set().union(*split_clips.values()) if split_clips else set()
    print("unique physical cases overall:", len(case_union))
    print("unique clips overall:", len(clip_union))

    overlap_rows = []
    print("\n===== PHYSICAL-CASE OVERLAP MATRIX =====")
    for i, a in enumerate(all_splits):
        for b in all_splits[i:]:
            inter = sorted(split_cases[a] & split_cases[b])
            overlap_rows.append({
                "split_a": a,
                "split_b": b,
                "case_overlap_count": len(inter),
                "case_overlap_ids": "|".join(inter),
            })
            print(f"{a} <-> {b}: {len(inter)}")

    # Specific diagnostic: official Unseen labels are not automatically assumed
    # to equal "physical patient unseen" until this overlap audit is inspected.
    train_cases = split_cases.get("Train", set())
    easy_unseen_cases = split_cases.get("TestEasyDataset-Unseen", set())
    hard_unseen_cases = split_cases.get("TestHardDataset-Unseen", set())

    print("\n===== UNSEEN-vs-TRAIN PHYSICAL CASE DIAGNOSTIC =====")
    print("Easy-Unseen ∩ Train:", len(easy_unseen_cases & train_cases))
    print("Hard-Unseen ∩ Train:", len(hard_unseen_cases & train_cases))
    print("Easy-Unseen ∩ Hard-Unseen:", len(easy_unseen_cases & hard_unseen_cases))

    pair_fields = [
        "pair_key", "dataset", "visibility", "clip_id", "physical_case_id",
        "frame_index", "frame_member", "gt_member", "frame_size", "gt_size"
    ]
    pair_csv = args.output_dir / "R14B2_fix2_real_frame_gt_pairs.csv"
    write_csv(pair_csv, paired["rows"], pair_fields)

    split_rows = []
    for s in all_splits:
        split_rows.append({
            "split": s,
            "frame_count": split_frames[s],
            "clip_count": len(split_clips[s]),
            "physical_case_count": len(split_cases[s]),
            "physical_case_ids": "|".join(sorted(split_cases[s])),
        })
    split_csv = args.output_dir / "R14B2_fix2_split_case_clip_summary.csv"
    write_csv(
        split_csv,
        split_rows,
        ["split", "frame_count", "clip_count", "physical_case_count", "physical_case_ids"]
    )

    overlap_csv = args.output_dir / "R14B2_fix2_physical_case_overlap_matrix.csv"
    write_csv(
        overlap_csv,
        overlap_rows,
        ["split_a", "split_b", "case_overlap_count", "case_overlap_ids"]
    )

    artifact_rows = [
        {"source": "raw_tar", "reason": k, "count": v}
        for k, v in raw_excluded.most_common()
    ] + [
        {"source": "annotation_zip", "reason": k, "count": v}
        for k, v in ann_excluded.most_common()
    ]
    artifact_csv = args.output_dir / "R14B2_fix2_filtered_member_summary.csv"
    write_csv(artifact_csv, artifact_rows, ["source", "reason", "count"])

    # Frame count per physical case and per clip.
    case_counts = Counter(r["physical_case_id"] for r in paired["rows"])
    clip_counts = Counter((split_name(r["dataset"], r["visibility"]), r["clip_id"])
                          for r in paired["rows"])

    case_csv = args.output_dir / "R14B2_fix2_physical_case_frame_counts.csv"
    write_csv(
        case_csv,
        [{"physical_case_id": k, "frame_count": v} for k, v in case_counts.most_common()],
        ["physical_case_id", "frame_count"]
    )

    clip_csv = args.output_dir / "R14B2_fix2_clip_frame_counts.csv"
    write_csv(
        clip_csv,
        [{"split": k[0], "clip_id": k[1], "frame_count": v}
         for k, v in clip_counts.most_common()],
        ["split", "clip_id", "frame_count"]
    )

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "supersedes": "Q1_R14B2_sunseg_deep_schema_pairing_identity_audit_fix1",
        "reason_for_fix": (
            "R14B2 fix1 counted macOS AppleDouble/resource-fork members "
            "such as ._case* as Frame/GT entries."
        ),
        "input_sha256": {
            "raw_tar_gz": raw_sha,
            "annotation_zip": ann_sha,
        },
        "sanitized_pairing": {
            "real_frame_unique": frame_count,
            "real_gt_unique": gt_count,
            "paired": pair_count,
            "frame_only": frame_only,
            "gt_only": gt_only,
            "annotation_v2_gt_unique": len(ann_gt_keys),
            "raw_gt_not_in_annotation": len(raw_gt_vs_ann_missing),
            "annotation_not_in_raw_gt": len(ann_vs_raw_missing),
        },
        "identity": {
            "physical_case_rule": (
                "parent directory caseNN[_clip] -> physical_case_id=caseNN; "
                "clip_id=full parent directory"
            ),
            "unique_physical_cases": len(case_union),
            "unique_clips": len(clip_union),
            "unparsed_pair_count": len(bad_identity),
        },
        "split_summary": split_rows,
        "information_boundary": {
            "archive_extracted": False,
            "gt_pixels_decoded": False,
            "model_inference": False,
            "tta_execution": False,
            "subset_selection": False,
            "target_outcomes_read": False,
            "contamination_claim": False,
            "confirmatory_cohort_frozen": False,
        },
        "next_stage": (
            "R14B3_FREEZE_CONFIRMATORY_COHORT_AND_PHYSICAL_CASE_CLUSTER_UNIT_"
            "AFTER_REVIEWING_CASE_OVERLAP"
        ),
    }

    lock_path = args.output_dir / "R14B2_FIX2_REAL_IMAGE_PAIRING_CASE_IDENTITY_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nDecision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
