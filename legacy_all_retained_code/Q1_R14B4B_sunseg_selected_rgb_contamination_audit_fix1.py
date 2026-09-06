#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix1.py

Final pre-inference contamination audit for the R14B3 frozen SUN-SEG candidate.

Frozen candidate:
- 980 SUN-SEG RGB frames
- 49 physical caseNN clusters
- 20 frames per case
- manifest SHA256:
  f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d

Reference datasets:
- S01 authoritative protocol manifest: 2248 rows
- NeoPolyp frozen confirmatory manifest: 1000 rows
- PolypGen unique static manifest: 1532 rows

Audit tiers:
A. exact encoded-file SHA256
B. exact decoded RGB-pixel SHA256
C. conservative re-encoding duplicate screen:
   same native width/height
   + dHash64 Hamming <= 4
   + 64x64 RGB thumbnail MAE <= 4.0 creates a review candidate
   hard probable re-encoding contamination requires:
       dHash64 Hamming <= 1
       thumbnail MAE <= 1.5
       full-resolution RGB MAE <= 2.0
       full-resolution PSNR >= 40 dB

The exact RGB-pixel hash recipe is verified against the existing S01 and
NeoPolyp manifests before any SUN comparison. Exact file hashes are also
re-verified for S01, NeoPolyp, and PolypGen.

No GT pixels are read.
No SOURCE/TTA inference is performed.
No safety scores or target outcomes are accessed.
No target tuning occurs.

If a selected SUN frame triggers exact or hard re-encoding contamination,
the entire physical case is excluded with NO replacement, as frozen in R14B3.
Soft near-duplicate candidates do not silently pass: they force
PENDING_NEAR_DUPLICATE_REVIEW.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")

SUN_TAR = (
    ROOT / "data" / "external" / "SUN_SEG"
    / "SUN-SEG-FinalData-v20251212.tar.gz"
)
R14B3_DIR = (
    ROOT / "outputs"
    / "Q1_R14B3_sunseg_unseen_case_balanced_candidate_lock_fix1_v1"
)
SUN_MANIFEST = (
    R14B3_DIR / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_MANIFEST.csv"
)
SUN_LOCK = (
    R14B3_DIR / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_LOCK.json"
)

S01_MANIFEST = ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
NEO_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)
POLYPGEN_MANIFEST = (
    ROOT / "outputs"
    / "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1"
    / "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
)

OUT = (
    ROOT / "outputs"
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix1_v1"
)

EXPECTED_SUN_TAR_SHA = (
    "d9a00fada04782937a144e9d1bc3c2a3b7f8e321e37ba1cf83bebdd490563016"
)
EXPECTED_SUN_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)
EXPECTED_S01_MANIFEST_SHA = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)
EXPECTED_NEO_MANIFEST_SHA = (
    "c944b95ec17f9c3fe26ea1c8555ca9309e490fc22a62da8fab43fa8ace1d39b9"
)
EXPECTED_POLYPGEN_MANIFEST_SHA = (
    "37058a8fcd020307e4a684011864ce9b9858f323ee895af3cfaf237f129a80c0"
)

EXPECTED_SUN_ROWS = 980
EXPECTED_SUN_CASES = 49
EXPECTED_S01_ROWS = 2248
EXPECTED_NEO_ROWS = 1000
EXPECTED_POLYPGEN_ROWS = 1532

SOFT_DHASH_MAX = 4
SOFT_THUMB_MAE_MAX = 4.0
HARD_DHASH_MAX = 1
HARD_THUMB_MAE_MAX = 1.5
HARD_FULL_MAE_MAX = 2.0
HARD_FULL_PSNR_MIN = 40.0

DECISION_PASS = (
    "SUNSEG_SELECTED_RGB_CONTAMINATION_AUDIT_PASS_"
    "FINAL_CONFIRMATORY_COHORT_LOCKED"
)
DECISION_PENDING = (
    "SUNSEG_SELECTED_RGB_CONTAMINATION_AUDIT_"
    "PENDING_NEAR_DUPLICATE_REVIEW"
)
DECISION_EXCLUDED = (
    "SUNSEG_CONTAMINATED_CASES_EXCLUDED_NO_REPLACEMENT_"
    "FINAL_CONFIRMATORY_COHORT_LOCKED"
)


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


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rel_to_path(rel: str) -> Path:
    return Path(*PurePosixPath(rel.replace("\\", "/")).parts)


def discover_s01_root(rows, user_root: Path | None):
    """
    No root guessing:
    - explicit --s01-root wins and must validate all 2248 paths;
    - otherwise discover directories named as the first relpath component under
      F:\MEDSEG_SAFETTA\data, derive parent roots, and require exactly one root
      that resolves ALL image_relpath entries.
    """
    if user_root is not None:
        candidates = [user_root]
        method = "explicit_argument"
    else:
        first_rel = str(rows[0]["image_relpath"]).strip().replace("\\", "/")
        first_component = PurePosixPath(first_rel).parts[0]
        search_base = ROOT / "data"
        found = []
        for d in search_base.rglob(first_component):
            if d.is_dir():
                found.append(d.parent)
        candidates = sorted(set(found), key=lambda p: str(p).lower())
        method = f"deterministic_search_under_{search_base}"

    valid = []
    validations = []

    for base in candidates:
        missing = 0
        for r in rows:
            p = base / rel_to_path(str(r["image_relpath"]).strip())
            if not p.is_file():
                missing += 1
        validations.append({
            "candidate_root": str(base),
            "missing_image_paths": missing,
            "valid_all": int(missing == 0),
        })
        if missing == 0:
            valid.append(base)

    print("\n===== S01 ROOT RESOLUTION =====")
    print("method:", method)
    print("candidate roots:", len(candidates))
    for v in validations:
        print(
            f"{v['candidate_root']} | "
            f"missing={v['missing_image_paths']} "
            f"valid_all={v['valid_all']}"
        )

    if len(valid) != 1:
        raise RuntimeError(
            "S01 root resolution is not unique. "
            "Pass the exact root with --s01-root after inspecting candidates."
        )

    print("resolved S01 root:", valid[0])
    return valid[0], validations


def decode_rgb_from_bytes(blob: bytes):
    with Image.open(io.BytesIO(blob)) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8).copy()
    return arr


def decode_rgb_from_path(path: Path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8).copy()
    return arr


def rgb_pixel_sha(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError("RGB array invariant failed.")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dhash64(arr: np.ndarray) -> int:
    im = Image.fromarray(arr, mode="RGB").convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    a = np.asarray(im, dtype=np.uint8)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def thumb64(arr: np.ndarray) -> np.ndarray:
    im = Image.fromarray(arr, mode="RGB").resize(
        (64, 64), Image.Resampling.LANCZOS
    )
    return np.asarray(im, dtype=np.uint8).copy()


def descriptor(arr: np.ndarray):
    h, w = arr.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "pixel_sha256": rgb_pixel_sha(arr),
        "dhash64": dhash64(arr),
        "thumb": thumb64(arr),
    }


def full_metrics(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return math.inf, -math.inf
    da = a.astype(np.float32)
    db = b.astype(np.float32)
    diff = da - db
    mae = float(np.abs(diff).mean())
    mse = float(np.square(diff).mean())
    if mse == 0:
        psnr = math.inf
    else:
        psnr = float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return mae, psnr


def verify_manifest_hash(path: Path, expected: str, label: str):
    got = sha256_file(path)
    print(f"{label} SHA256={got}")
    if got != expected:
        raise RuntimeError(f"{label} manifest SHA mismatch.")
    return got


def load_reference_s01(rows, root):
    out = []
    raw_mismatch = 0
    pixel_mismatch = 0

    for r in tqdm(rows, desc="Verify/decode S01 RGB", unit="image",
                  dynamic_ncols=True):
        p = root / rel_to_path(r["image_relpath"])
        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_file_sha256"]:
            raw_mismatch += 1
        arr = decode_rgb_from_path(p)
        d = descriptor(arr)
        if d["pixel_sha256"] != r["image_pixel_sha256_rgb"]:
            pixel_mismatch += 1

        out.append({
            "dataset": "S01",
            "ref_id": r["sample_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            **d,
        })

    print("S01 raw SHA mismatches:", raw_mismatch)
    print("S01 RGB pixel SHA mismatches:", pixel_mismatch)
    if raw_mismatch or pixel_mismatch:
        raise RuntimeError(
            "S01 canonical reference verification failed; "
            "do not continue contamination audit."
        )
    return out


def sha256_file_quiet(path: Path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_reference_neo(rows):
    out = []
    raw_mismatch = 0
    pixel_mismatch = 0

    for r in tqdm(rows, desc="Verify/decode NeoPolyp RGB", unit="image",
                  dynamic_ncols=True):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)
        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_raw_sha256"]:
            raw_mismatch += 1
        arr = decode_rgb_from_path(p)
        d = descriptor(arr)
        if d["pixel_sha256"] != r["image_rgb_pixel_sha256"]:
            pixel_mismatch += 1

        out.append({
            "dataset": "NeoPolyp",
            "ref_id": r["sample_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            **d,
        })

    print("NeoPolyp raw SHA mismatches:", raw_mismatch)
    print("NeoPolyp RGB pixel SHA mismatches:", pixel_mismatch)
    if raw_mismatch or pixel_mismatch:
        raise RuntimeError(
            "NeoPolyp canonical reference verification failed."
        )
    return out


def load_reference_polypgen(rows):
    out = []
    raw_mismatch = 0

    for r in tqdm(rows, desc="Verify/decode PolypGen RGB", unit="image",
                  dynamic_ncols=True):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)
        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_sha256"]:
            raw_mismatch += 1
        arr = decode_rgb_from_path(p)
        d = descriptor(arr)

        out.append({
            "dataset": "PolypGen",
            "ref_id": r["external_case_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            **d,
        })

    print("PolypGen raw SHA mismatches:", raw_mismatch)
    if raw_mismatch:
        raise RuntimeError("PolypGen reference verification failed.")
    return out


def load_sun_selected(rows, tar_path):
    member_names = {r["frame_member"] for r in rows}
    row_by_member = {r["frame_member"]: r for r in rows}
    if len(member_names) != len(rows):
        raise RuntimeError("SUN selected frame_member values are not unique.")

    out = []
    found = set()

    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.name.replace("\\", "/"): m for m in tf.getmembers()
                   if m.isfile()}

        for member_name in tqdm(
            sorted(member_names),
            desc="Read/decode selected SUN RGB from TAR",
            unit="frame",
            dynamic_ncols=True,
        ):
            m = members.get(member_name)
            if m is None:
                raise RuntimeError(f"SUN TAR member missing: {member_name}")
            f = tf.extractfile(m)
            if f is None:
                raise RuntimeError(f"Cannot read SUN TAR member: {member_name}")
            blob = f.read()
            arr = decode_rgb_from_bytes(blob)
            d = descriptor(arr)
            src = row_by_member[member_name]

            out.append({
                "pair_key": src["pair_key"],
                "cluster_id": src["cluster_id"],
                "physical_case_id": src["physical_case_id"],
                "clip_id": src["clip_id"],
                "source_split_label": src["source_split_label"],
                "frame_member": member_name,
                "file_sha256": sha256_bytes(blob),
                **d,
            })
            found.add(member_name)

    if found != member_names:
        raise RuntimeError("Not all selected SUN members were read.")
    return out


def exact_match_audit(sun, refs):
    file_map = defaultdict(list)
    pixel_map = defaultdict(list)
    for r in refs:
        file_map[r["file_sha256"]].append(r)
        pixel_map[r["pixel_sha256"]].append(r)

    matches = []
    contaminated_cases = set()

    for s in sun:
        for r in file_map.get(s["file_sha256"], []):
            matches.append({
                "match_type": "EXACT_ENCODED_FILE_SHA256",
                "sun_pair_key": s["pair_key"],
                "sun_cluster_id": s["cluster_id"],
                "sun_frame_member": s["frame_member"],
                "reference_dataset": r["dataset"],
                "reference_id": r["ref_id"],
                "reference_path": r["path"],
                "dhash_hamming": 0,
                "thumb_mae": 0.0,
                "full_mae": 0.0,
                "full_psnr": "inf",
                "hard_contamination": 1,
            })
            contaminated_cases.add(s["cluster_id"])

        for r in pixel_map.get(s["pixel_sha256"], []):
            matches.append({
                "match_type": "EXACT_RGB_PIXEL_SHA256",
                "sun_pair_key": s["pair_key"],
                "sun_cluster_id": s["cluster_id"],
                "sun_frame_member": s["frame_member"],
                "reference_dataset": r["dataset"],
                "reference_id": r["ref_id"],
                "reference_path": r["path"],
                "dhash_hamming": 0,
                "thumb_mae": 0.0,
                "full_mae": 0.0,
                "full_psnr": "inf",
                "hard_contamination": 1,
            })
            contaminated_cases.add(s["cluster_id"])

    return matches, contaminated_cases


def internal_sun_duplicate_audit(sun):
    by_file = defaultdict(list)
    by_pixel = defaultdict(list)
    for s in sun:
        by_file[s["file_sha256"]].append(s)
        by_pixel[s["pixel_sha256"]].append(s)

    rows = []
    for kind, mp in (
        ("SUN_INTERNAL_FILE_SHA", by_file),
        ("SUN_INTERNAL_PIXEL_SHA", by_pixel),
    ):
        for h, grp in mp.items():
            if len(grp) <= 1:
                continue
            rows.append({
                "duplicate_type": kind,
                "sha256": h,
                "count": len(grp),
                "cluster_ids": "|".join(sorted({x["cluster_id"] for x in grp})),
                "pair_keys": "|".join(x["pair_key"] for x in grp),
            })
    return rows


def read_sun_full_array(tar_path: Path, member_name: str):
    with tarfile.open(tar_path, "r:gz") as tf:
        m = tf.getmember(member_name)
        f = tf.extractfile(m)
        if f is None:
            raise RuntimeError(f"Cannot read SUN member {member_name}")
        return decode_rgb_from_bytes(f.read())


def read_ref_full_array(ref):
    return decode_rgb_from_path(Path(ref["path"]))


def near_duplicate_audit(sun, refs, tar_path):
    candidates = []
    hard_cases = set()

    # Native-size bucketing sharply reduces false candidate comparisons.
    refs_by_size = defaultdict(list)
    for r in refs:
        refs_by_size[(r["width"], r["height"])].append(r)

    sun_array_cache = {}

    for s in tqdm(
        sun,
        desc="Near-duplicate screen",
        unit="SUN frame",
        dynamic_ncols=True,
    ):
        pool = refs_by_size.get((s["width"], s["height"]), [])
        for r in pool:
            ham = (int(s["dhash64"]) ^ int(r["dhash64"])).bit_count()
            if ham > SOFT_DHASH_MAX:
                continue

            thumb_mae = float(
                np.abs(
                    s["thumb"].astype(np.int16)
                    - r["thumb"].astype(np.int16)
                ).mean()
            )
            if thumb_mae > SOFT_THUMB_MAE_MAX:
                continue

            # Full-resolution check for all soft candidates.
            if s["frame_member"] not in sun_array_cache:
                sun_array_cache[s["frame_member"]] = read_sun_full_array(
                    tar_path, s["frame_member"]
                )
            a = sun_array_cache[s["frame_member"]]
            b = read_ref_full_array(r)
            full_mae, full_psnr = full_metrics(a, b)

            hard = int(
                ham <= HARD_DHASH_MAX
                and thumb_mae <= HARD_THUMB_MAE_MAX
                and full_mae <= HARD_FULL_MAE_MAX
                and full_psnr >= HARD_FULL_PSNR_MIN
            )
            if hard:
                hard_cases.add(s["cluster_id"])

            candidates.append({
                "match_type": (
                    "HARD_PROBABLE_REENCODING"
                    if hard else "SOFT_NEAR_DUPLICATE_CANDIDATE"
                ),
                "sun_pair_key": s["pair_key"],
                "sun_cluster_id": s["cluster_id"],
                "sun_frame_member": s["frame_member"],
                "reference_dataset": r["dataset"],
                "reference_id": r["ref_id"],
                "reference_path": r["path"],
                "dhash_hamming": ham,
                "thumb_mae": thumb_mae,
                "full_mae": full_mae,
                "full_psnr": full_psnr,
                "hard_contamination": hard,
            })

    return candidates, hard_cases


def serializable_descriptor_row(s):
    return {
        "pair_key": s["pair_key"],
        "cluster_id": s["cluster_id"],
        "physical_case_id": s["physical_case_id"],
        "clip_id": s["clip_id"],
        "source_split_label": s["source_split_label"],
        "frame_member": s["frame_member"],
        "width": s["width"],
        "height": s["height"],
        "file_sha256": s["file_sha256"],
        "rgb_pixel_sha256": s["pixel_sha256"],
        "dhash64_hex": f"{int(s['dhash64']):016x}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-tar", type=Path, default=SUN_TAR)
    ap.add_argument("--sun-manifest", type=Path, default=SUN_MANIFEST)
    ap.add_argument("--sun-lock", type=Path, default=SUN_LOCK)
    ap.add_argument("--s01-manifest", type=Path, default=S01_MANIFEST)
    ap.add_argument("--neopolyp-manifest", type=Path, default=NEO_MANIFEST)
    ap.add_argument("--polypgen-manifest", type=Path, default=POLYPGEN_MANIFEST)
    ap.add_argument("--s01-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()

    print("===== Q1 R14B4B SUN-SEG SELECTED RGB CONTAMINATION AUDIT =====")
    print("SUN selected frames=980")
    print("SUN physical cases=49")
    print("GT_PIXEL_DECODE=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("SAFETY_SCORE_ACCESS=NO")
    print("TARGET_OUTCOME_ACCESS=NO")
    print("TARGET_TUNING=NO")
    print("CASE_EXCLUSION_ON_HARD_CONTAMINATION=YES")
    print("REPLACEMENT_AFTER_CASE_EXCLUSION=NO")

    required = (
        args.sun_tar,
        args.sun_manifest,
        args.sun_lock,
        args.s01_manifest,
        args.neopolyp_manifest,
        args.polypgen_manifest,
    )
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    # Exact lineage gates.
    if sha256_file(args.sun_tar) != EXPECTED_SUN_TAR_SHA:
        raise RuntimeError("SUN TAR SHA mismatch.")
    verify_manifest_hash(
        args.sun_manifest, EXPECTED_SUN_MANIFEST_SHA, "R14B3 SUN manifest"
    )
    verify_manifest_hash(
        args.s01_manifest, EXPECTED_S01_MANIFEST_SHA, "S01 manifest"
    )
    verify_manifest_hash(
        args.neopolyp_manifest, EXPECTED_NEO_MANIFEST_SHA, "NeoPolyp manifest"
    )
    verify_manifest_hash(
        args.polypgen_manifest,
        EXPECTED_POLYPGEN_MANIFEST_SHA,
        "PolypGen manifest",
    )

    sun_cols, sun_rows = read_csv(args.sun_manifest)
    s01_cols, s01_rows = read_csv(args.s01_manifest)
    neo_cols, neo_rows = read_csv(args.neopolyp_manifest)
    pg_cols, pg_rows = read_csv(args.polypgen_manifest)

    if len(sun_rows) != EXPECTED_SUN_ROWS:
        raise RuntimeError("SUN row count mismatch.")
    if len({r["cluster_id"] for r in sun_rows}) != EXPECTED_SUN_CASES:
        raise RuntimeError("SUN physical-case count mismatch.")
    if len(s01_rows) != EXPECTED_S01_ROWS:
        raise RuntimeError("S01 row count mismatch.")
    if len(neo_rows) != EXPECTED_NEO_ROWS:
        raise RuntimeError("NeoPolyp row count mismatch.")
    if len(pg_rows) != EXPECTED_POLYPGEN_ROWS:
        raise RuntimeError("PolypGen row count mismatch.")

    sun_lock = json.loads(args.sun_lock.read_text(encoding="utf-8"))
    if sun_lock.get("status") != "PASS":
        raise RuntimeError("R14B3 lock is not PASS.")

    # Resolve S01 source files without guessing.
    s01_root, s01_root_validations = discover_s01_root(
        s01_rows, args.s01_root
    )

    print("\n===== REFERENCE HASH-RECIPE VERIFICATION =====")
    refs_s01 = load_reference_s01(s01_rows, s01_root)
    refs_neo = load_reference_neo(neo_rows)
    refs_pg = load_reference_polypgen(pg_rows)
    refs = refs_s01 + refs_neo + refs_pg

    print("reference RGB images total:", len(refs))
    if len(refs) != (
        EXPECTED_S01_ROWS + EXPECTED_NEO_ROWS + EXPECTED_POLYPGEN_ROWS
    ):
        raise RuntimeError("Reference count mismatch.")

    print("\n===== SUN SELECTED RGB READ/DECODE =====")
    sun = load_sun_selected(sun_rows, args.sun_tar)
    if len(sun) != EXPECTED_SUN_ROWS:
        raise RuntimeError("SUN decoded count mismatch.")

    internal_dups = internal_sun_duplicate_audit(sun)
    print("SUN internal exact duplicate groups:", len(internal_dups))
    if internal_dups:
        print(
            "STOP CONDITION: SUN selected set contains internal exact "
            "file/pixel duplicates; do not silently rebalance."
        )

    print("\n===== EXACT CROSS-DATASET CONTAMINATION =====")
    exact_matches, exact_cases = exact_match_audit(sun, refs)
    print("exact match rows:", len(exact_matches))
    print("physical cases hit by exact matches:", len(exact_cases))

    print("\n===== CONSERVATIVE RE-ENCODING SCREEN =====")
    near_rows, hard_reencode_cases = near_duplicate_audit(
        sun, refs, args.sun_tar
    )
    hard_near_rows = [r for r in near_rows if int(r["hard_contamination"]) == 1]
    soft_near_rows = [r for r in near_rows if int(r["hard_contamination"]) == 0]

    print("soft/hard near-duplicate candidates total:", len(near_rows))
    print("hard probable re-encoding rows:", len(hard_near_rows))
    print("soft review candidates:", len(soft_near_rows))
    print(
        "physical cases hit by hard re-encoding:",
        len(hard_reencode_cases),
    )

    contaminated_cases = set(exact_cases) | set(hard_reencode_cases)

    # R14B3 frozen policy: whole-case exclusion, no replacement.
    clean_sun_rows = [
        r for r in sun_rows if r["cluster_id"] not in contaminated_cases
    ]
    clean_cases = sorted({r["cluster_id"] for r in clean_sun_rows})

    selected_counts = Counter(r["cluster_id"] for r in sun_rows)
    clean_counts = Counter(r["cluster_id"] for r in clean_sun_rows)

    print("\n===== FROZEN CASE-LEVEL CONTAMINATION POLICY =====")
    print("contaminated physical cases:", len(contaminated_cases))
    print("remaining physical cases:", len(clean_cases))
    print("remaining selected frames:", len(clean_sun_rows))
    print("replacement frames/cases added: 0")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    descriptor_csv = args.output_dir / "R14B4B_SUN_SELECTED_RGB_HASHES.csv"
    desc_rows = [serializable_descriptor_row(s) for s in sun]
    write_csv(
        descriptor_csv,
        desc_rows,
        [
            "pair_key", "cluster_id", "physical_case_id", "clip_id",
            "source_split_label", "frame_member", "width", "height",
            "file_sha256", "rgb_pixel_sha256", "dhash64_hex",
        ],
    )

    match_fields = [
        "match_type",
        "sun_pair_key",
        "sun_cluster_id",
        "sun_frame_member",
        "reference_dataset",
        "reference_id",
        "reference_path",
        "dhash_hamming",
        "thumb_mae",
        "full_mae",
        "full_psnr",
        "hard_contamination",
    ]

    exact_csv = args.output_dir / "R14B4B_EXACT_CONTAMINATION_MATCHES.csv"
    write_csv(exact_csv, exact_matches, match_fields)

    near_csv = args.output_dir / "R14B4B_NEAR_DUPLICATE_CANDIDATES.csv"
    write_csv(near_csv, near_rows, match_fields)

    internal_csv = args.output_dir / "R14B4B_SUN_INTERNAL_EXACT_DUPLICATES.csv"
    write_csv(
        internal_csv,
        internal_dups,
        ["duplicate_type", "sha256", "count", "cluster_ids", "pair_keys"],
    )

    final_manifest = args.output_dir / "R14B4B_FINAL_CONFIRMATORY_MANIFEST.csv"
    write_csv(final_manifest, clean_sun_rows, sun_cols)

    exclusion_rows = []
    for case_id in sorted(contaminated_cases):
        types = sorted({
            r["match_type"]
            for r in exact_matches + hard_near_rows
            if r["sun_cluster_id"] == case_id
        })
        exclusion_rows.append({
            "cluster_id": case_id,
            "precontamination_frame_count": selected_counts[case_id],
            "postcontamination_frame_count": clean_counts[case_id],
            "exclusion_reason": "|".join(types),
            "replacement": "NO",
        })

    exclusion_csv = args.output_dir / "R14B4B_CASE_EXCLUSIONS.csv"
    write_csv(
        exclusion_csv,
        exclusion_rows,
        [
            "cluster_id",
            "precontamination_frame_count",
            "postcontamination_frame_count",
            "exclusion_reason",
            "replacement",
        ],
    )

    root_csv = args.output_dir / "R14B4B_S01_ROOT_RESOLUTION.csv"
    write_csv(
        root_csv,
        s01_root_validations,
        ["candidate_root", "missing_image_paths", "valid_all"],
    )

    # Decision logic.
    if internal_dups:
        status = "STOP"
        decision = "SUNSEG_INTERNAL_EXACT_DUPLICATES_REQUIRE_PROTOCOL_REVIEW"
        final_locked = False
    elif soft_near_rows:
        status = "PENDING_REVIEW"
        decision = DECISION_PENDING
        final_locked = False
    else:
        status = "PASS"
        decision = DECISION_EXCLUDED if contaminated_cases else DECISION_PASS
        final_locked = True

    lock = {
        "status": status,
        "decision": decision,
        "frozen_candidate": {
            "manifest_sha256": EXPECTED_SUN_MANIFEST_SHA,
            "frames": EXPECTED_SUN_ROWS,
            "physical_cases": EXPECTED_SUN_CASES,
        },
        "reference_assets": {
            "S01": {
                "manifest_sha256": EXPECTED_S01_MANIFEST_SHA,
                "rows": EXPECTED_S01_ROWS,
                "resolved_image_root": str(s01_root),
                "raw_sha_recipe_reverified": True,
                "rgb_pixel_sha_recipe_reverified": True,
            },
            "NeoPolyp": {
                "manifest_sha256": EXPECTED_NEO_MANIFEST_SHA,
                "rows": EXPECTED_NEO_ROWS,
                "raw_sha_recipe_reverified": True,
                "rgb_pixel_sha_recipe_reverified": True,
            },
            "PolypGen": {
                "manifest_sha256": EXPECTED_POLYPGEN_MANIFEST_SHA,
                "rows": EXPECTED_POLYPGEN_ROWS,
                "raw_sha_recipe_reverified": True,
                "rgb_pixel_sha_computed_now": True,
            },
        },
        "contamination_rule": {
            "exact_file_sha256": "HARD",
            "exact_rgb_pixel_sha256": "HARD",
            "soft_candidate": {
                "same_native_size": True,
                "dhash64_hamming_max": SOFT_DHASH_MAX,
                "thumbnail_64_rgb_mae_max": SOFT_THUMB_MAE_MAX,
            },
            "hard_probable_reencoding": {
                "dhash64_hamming_max": HARD_DHASH_MAX,
                "thumbnail_64_rgb_mae_max": HARD_THUMB_MAE_MAX,
                "full_rgb_mae_max": HARD_FULL_MAE_MAX,
                "full_rgb_psnr_min_db": HARD_FULL_PSNR_MIN,
            },
            "hard_hit_exclusion_unit": "entire_physical_case",
            "replacement_after_case_exclusion": False,
        },
        "results": {
            "sun_internal_exact_duplicate_groups": len(internal_dups),
            "exact_cross_dataset_match_rows": len(exact_matches),
            "hard_probable_reencoding_rows": len(hard_near_rows),
            "soft_near_duplicate_review_rows": len(soft_near_rows),
            "contaminated_physical_cases": len(contaminated_cases),
            "remaining_physical_cases": len(clean_cases),
            "remaining_frames": len(clean_sun_rows),
        },
        "final_confirmatory_cohort_locked": final_locked,
        "final_manifest_path": str(final_manifest),
        "final_manifest_sha256": sha256_file_quiet(final_manifest),
        "information_boundary": {
            "sun_rgb_pixels_decoded_for_contamination_only": True,
            "reference_rgb_pixels_decoded_for_contamination_only": True,
            "gt_pixels_decoded": False,
            "model_inference": False,
            "tta_execution": False,
            "safety_scores_accessed": False,
            "target_outcomes_accessed": False,
            "target_tuning": False,
        },
        "next_stage": (
            "R14C_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK_PRE_GT"
            if final_locked
            else "RESOLVE_R14B4B_REVIEW_BEFORE_ANY_INFERENCE"
        ),
    }

    lock_path = args.output_dir / "R14B4B_SUNSEG_CONTAMINATION_AUDIT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== FINAL R14B4B STATUS =====")
    print("status=", status)
    print("Decision=", decision)
    print("final confirmatory cohort locked=", final_locked)
    print("final manifest=", final_manifest)
    print("final manifest SHA256=", sha256_file_quiet(final_manifest))
    print("LOCK=", lock_path)

    if status == "PASS":
        print("PASS")
    elif status == "PENDING_REVIEW":
        print("PENDING_REVIEW")
    else:
        print("STOP")


if __name__ == "__main__":
    main()
