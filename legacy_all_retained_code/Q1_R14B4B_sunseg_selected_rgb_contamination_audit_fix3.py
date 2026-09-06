#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix2.py

Supersedes R14B4B Fix1.

Reason for Fix2
---------------
R14B4B Fix1 stopped correctly before SUN-SEG RGB access because it assumed the
NeoPolyp stored `image_rgb_pixel_sha256` was SHA256 over RGB bytes only.

R14B4B2/R14B4B3 recovered and verified the exact historical NeoPolyp recipe:
    SHA256( f"{w}x{h}|RGB|".encode("ascii") + rgb.tobytes() )
and reproduced 1000/1000 stored hashes.

Fix2 separates:
1) reference-lineage verification hashes, which may be dataset-specific; from
2) a common cross-dataset RGB hash used for contamination comparison:
       SHA256(rgb.tobytes())

This prevents comparing incompatible historical hash recipes.

Frozen SUN candidate
--------------------
- 980 frames
- 49 physical caseNN clusters
- 20 frames/case
- R14B3 candidate manifest SHA256:
  f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d

References
----------
- S01: 2248 images
- NeoPolyp: 1000 images
- PolypGen: 1532 images

Audit tiers
-----------
A. exact encoded-file SHA256
B. exact COMMON decoded RGB-byte SHA256
C. conservative re-encoding screen

Soft candidate:
- same native width/height
- dHash64 Hamming <= 4
- 64x64 RGB thumbnail MAE <= 4.0

Hard probable re-encoding:
- dHash64 Hamming <= 1
- 64x64 RGB thumbnail MAE <= 1.5
- full-resolution RGB MAE <= 2.0
- full-resolution PSNR >= 40 dB

Frozen exclusion policy
-----------------------
Any hard contaminated selected SUN frame excludes the ENTIRE physical case.
No replacement frame or case is sampled.

Information boundary
--------------------
SUN RGB pixels: YES, contamination audit only.
Reference RGB pixels: YES, lineage/hash/duplicate audit only.
GT pixels: NO.
Model inference: NO.
TTA: NO.
Safety scores: NO.
Target outcomes: NO.
Target tuning: NO.
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
    R14B3_DIR
    / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_MANIFEST.csv"
)
SUN_LOCK = (
    R14B3_DIR
    / "R14B3_SUNSEG_PRECONTAMINATION_CANDIDATE_LOCK.json"
)

R14B4B3_LOCK = (
    ROOT / "outputs"
    / "Q1_R14B4B3_neopolyp_exact_historical_hash_recipe_verification_fix1_v1"
    / "R14B4B3_NEOPOLYP_HISTORICAL_HASH_RECIPE_LOCK.json"
)

S01_MANIFEST = (
    ROOT / "data" / "splits"
    / "S01_locked_protocol_manifest_v1.csv"
)

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
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
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
EXPECTED_R05D1_FIX2_SCRIPT_SHA = (
    "55ebe1b051bc250c3d5d471ee224fb5cd121501621120aaeb4729b1b3e2eee31"
)

EXPECTED_R14B4B3_DECISION = (
    "NEOPOLYP_HISTORICAL_PIXEL_HASH_RECIPE_VERIFIED_1000_OF_1000"
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


def sha256_file_quiet(path: Path, chunk=4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file_progress(path: Path, chunk=16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        desc=f"SHA256 {path.name}",
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


def decode_rgb_from_bytes(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return arr


def decode_rgb_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return arr


def common_rgb_sha256(arr: np.ndarray) -> str:
    """
    Common cross-dataset comparison hash:
        SHA256(contiguous RGB uint8 bytes)
    """
    if arr.dtype != np.uint8:
        raise RuntimeError("RGB dtype must be uint8.")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")
    return hashlib.sha256(
        np.ascontiguousarray(arr).tobytes(order="C")
    ).hexdigest()


def historical_neopolyp_pixel_sha256(arr: np.ndarray) -> str:
    """
    Exact historical R05D1 NeoPolyp recipe, recovered and verified in R14B4B3:
        SHA256( f"{w}x{h}|RGB|" + RGB bytes )
    """
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError("Invalid RGB array.")
    h, w = arr.shape[:2]
    hasher = hashlib.sha256()
    hasher.update(f"{w}x{h}|RGB|".encode("ascii"))
    hasher.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return hasher.hexdigest()


def dhash64(arr: np.ndarray) -> int:
    im = Image.fromarray(arr, mode="RGB").convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )
    a = np.asarray(im, dtype=np.uint8)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def thumb64(arr: np.ndarray) -> np.ndarray:
    im = Image.fromarray(arr, mode="RGB").resize(
        (64, 64),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(im, dtype=np.uint8).copy()


def make_descriptor(arr: np.ndarray):
    h, w = arr.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "common_rgb_sha256": common_rgb_sha256(arr),
        "dhash64": dhash64(arr),
        "thumb": thumb64(arr),
    }


def full_metrics(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return math.inf, -math.inf

    aa = a.astype(np.float32)
    bb = b.astype(np.float32)

    diff = aa - bb
    mae = float(np.abs(diff).mean())
    mse = float(np.square(diff).mean())

    if mse == 0:
        psnr = math.inf
    else:
        psnr = float(
            20.0 * math.log10(255.0 / math.sqrt(mse))
        )

    return mae, psnr


def validate_manifest_sha(path: Path, expected: str, label: str):
    got = sha256_file_progress(path)
    print(f"{label} SHA256={got}")
    if got != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected}, got={got}"
        )
    return got


def validate_r14b4b3_lock(lock_path: Path):
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if lock.get("status") != "PASS":
        raise RuntimeError("R14B4B3 lock is not PASS.")

    if lock.get("decision") != EXPECTED_R14B4B3_DECISION:
        raise RuntimeError(
            "Unexpected R14B4B3 decision: "
            f"{lock.get('decision')}"
        )

    if (
        lock.get("historical_script_sha256")
        != EXPECTED_R05D1_FIX2_SCRIPT_SHA
    ):
        raise RuntimeError(
            "R14B4B3 does not point to the expected historical "
            "R05D1 fix2 script."
        )

    ver = lock.get("verification", {})
    if int(ver.get("rows", -1)) != EXPECTED_NEO_ROWS:
        raise RuntimeError("R14B4B3 verification row count mismatch.")
    if int(ver.get("raw_sha_mismatches", -1)) != 0:
        raise RuntimeError("R14B4B3 raw SHA verification is not clean.")
    if int(ver.get("historical_pixel_sha_matches", -1)) != EXPECTED_NEO_ROWS:
        raise RuntimeError("R14B4B3 historical pixel SHA gate not 1000/1000.")
    if int(ver.get("historical_pixel_sha_mismatches", -1)) != 0:
        raise RuntimeError("R14B4B3 historical pixel mismatch count not zero.")

    recipe = lock.get("recovered_recipe", {})
    if recipe.get("sha256_prefix") != "{w}x{h}|RGB|":
        raise RuntimeError("Unexpected R14B4B3 hash prefix.")
    if recipe.get("hash_payload_order") != "prefix_then_raw_rgb_bytes":
        raise RuntimeError("Unexpected R14B4B3 hash payload order.")

    print("R14B4B3 historical NeoPolyp hash gate=PASS")
    return lock


def discover_s01_root(rows, explicit_root: Path | None):
    """
    No silent path guessing.

    If --s01-root is supplied:
        it must resolve all 2248 image_relpath values.

    Otherwise:
        search directories named as the first relpath component beneath
        F:\\MEDSEG_SAFETTA\\data and require exactly one parent root that
        resolves all 2248 image paths.
    """
    if explicit_root is not None:
        candidates = [explicit_root]
        method = "explicit_argument"
    else:
        first_rel = (
            str(rows[0]["image_relpath"])
            .strip()
            .replace("\\", "/")
        )
        first_component = PurePosixPath(first_rel).parts[0]

        search_base = ROOT / "data"
        found = []

        for d in search_base.rglob(first_component):
            if d.is_dir():
                found.append(d.parent)

        candidates = sorted(
            set(found),
            key=lambda p: str(p).lower(),
        )
        method = f"deterministic_search_under_{search_base}"

    validations = []
    valid = []

    for root in candidates:
        missing = 0
        for r in rows:
            p = root / rel_to_path(r["image_relpath"])
            if not p.is_file():
                missing += 1

        rec = {
            "candidate_root": str(root),
            "missing_image_paths": missing,
            "valid_all": int(missing == 0),
        }
        validations.append(rec)

        if missing == 0:
            valid.append(root)

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
            "Use --s01-root with the exact verified root."
        )

    print("resolved S01 root:", valid[0])
    return valid[0], validations


def load_reference_s01(rows, root: Path):
    refs = []
    raw_mismatch = 0
    stored_pixel_mismatch = 0

    for r in tqdm(
        rows,
        desc="Verify/decode S01 RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = root / rel_to_path(r["image_relpath"])
        if not p.is_file():
            raise FileNotFoundError(p)

        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_file_sha256"]:
            raw_mismatch += 1

        arr = decode_rgb_from_path(p)
        desc = make_descriptor(arr)

        # S01 historical/canonical stored recipe was already demonstrated to
        # equal common RGB bytes in Fix1, so re-check it exactly.
        if desc["common_rgb_sha256"] != r["image_pixel_sha256_rgb"]:
            stored_pixel_mismatch += 1

        refs.append({
            "dataset": "S01",
            "ref_id": r["sample_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            **desc,
        })

    print("S01 raw SHA mismatches:", raw_mismatch)
    print(
        "S01 stored RGB pixel SHA mismatches:",
        stored_pixel_mismatch,
    )

    if raw_mismatch or stored_pixel_mismatch:
        raise RuntimeError(
            "S01 canonical reference verification failed."
        )

    return refs


def load_reference_neopolyp(rows):
    refs = []
    raw_mismatch = 0
    historical_pixel_mismatch = 0

    for r in tqdm(
        rows,
        desc="Verify/decode NeoPolyp RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)

        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_raw_sha256"]:
            raw_mismatch += 1

        arr = decode_rgb_from_path(p)
        desc = make_descriptor(arr)

        historical_sha = historical_neopolyp_pixel_sha256(arr)
        if historical_sha != r["image_rgb_pixel_sha256"]:
            historical_pixel_mismatch += 1

        refs.append({
            "dataset": "NeoPolyp",
            "ref_id": r["sample_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            "historical_neopolyp_pixel_sha256": historical_sha,
            **desc,
        })

    print("NeoPolyp raw SHA mismatches:", raw_mismatch)
    print(
        "NeoPolyp historical pixel SHA mismatches:",
        historical_pixel_mismatch,
    )

    if raw_mismatch or historical_pixel_mismatch:
        raise RuntimeError(
            "NeoPolyp historical provenance verification failed."
        )

    return refs


def load_reference_polypgen(rows):
    refs = []
    raw_mismatch = 0

    for r in tqdm(
        rows,
        desc="Verify/decode PolypGen RGB",
        unit="image",
        dynamic_ncols=True,
    ):
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)

        raw_sha = sha256_file_quiet(p)
        if raw_sha != r["image_sha256"]:
            raw_mismatch += 1

        arr = decode_rgb_from_path(p)
        desc = make_descriptor(arr)

        refs.append({
            "dataset": "PolypGen",
            "ref_id": r["external_case_id"],
            "path": str(p),
            "file_sha256": raw_sha,
            **desc,
        })

    print("PolypGen raw SHA mismatches:", raw_mismatch)

    if raw_mismatch:
        raise RuntimeError(
            "PolypGen canonical reference verification failed."
        )

    return refs


def load_selected_sun(rows, tar_path: Path):
    """
    Read the frozen selected SUN RGB members in one sequential pass through
    the gzip-compressed TAR. This avoids repeated random-access decompression.
    """
    member_to_row = {r["frame_member"]: r for r in rows}
    if len(member_to_row) != len(rows):
        raise RuntimeError("SUN selected frame_member values are not unique.")

    target_names = set(member_to_row)
    out = []
    found = set()

    with tarfile.open(tar_path, "r|gz") as tf:
        with tqdm(
            total=len(target_names),
            desc="Read/decode selected SUN RGB (single TAR pass)",
            unit="frame",
            dynamic_ncols=True,
        ) as bar:
            for m in tf:
                if not m.isfile():
                    continue

                name = m.name.replace("\\", "/")
                if name not in target_names:
                    continue

                f = tf.extractfile(m)
                if f is None:
                    raise RuntimeError(f"Could not read SUN member: {name}")

                blob = f.read()
                arr = decode_rgb_from_bytes(blob)
                desc = make_descriptor(arr)
                src = member_to_row[name]

                out.append({
                    "pair_key": src["pair_key"],
                    "cluster_id": src["cluster_id"],
                    "physical_case_id": src["physical_case_id"],
                    "clip_id": src["clip_id"],
                    "source_split_label": src["source_split_label"],
                    "frame_member": name,
                    "file_sha256": sha256_bytes(blob),
                    **desc,
                })
                found.add(name)
                bar.update(1)

                if len(found) == len(target_names):
                    break

    if found != target_names:
        missing = sorted(target_names - found)
        raise RuntimeError(
            f"Not all selected SUN RGB members were read. "
            f"Missing={len(missing)} examples={missing[:10]}"
        )

    by_name = {r["frame_member"]: r for r in out}
    return [by_name[r["frame_member"]] for r in rows]


def internal_sun_exact_duplicate_audit(sun_rows):
    file_map = defaultdict(list)
    pixel_map = defaultdict(list)

    for r in sun_rows:
        file_map[r["file_sha256"]].append(r)
        pixel_map[r["common_rgb_sha256"]].append(r)

    groups = []

    for kind, mapping in (
        ("SUN_INTERNAL_FILE_SHA256", file_map),
        ("SUN_INTERNAL_COMMON_RGB_SHA256", pixel_map),
    ):
        for digest, grp in mapping.items():
            if len(grp) <= 1:
                continue

            groups.append({
                "duplicate_type": kind,
                "sha256": digest,
                "count": len(grp),
                "cluster_ids": "|".join(
                    sorted({x["cluster_id"] for x in grp})
                ),
                "pair_keys": "|".join(
                    x["pair_key"] for x in grp
                ),
            })

    return groups


def exact_cross_dataset_audit(sun_rows, refs):
    ref_file_map = defaultdict(list)
    ref_pixel_map = defaultdict(list)

    for r in refs:
        ref_file_map[r["file_sha256"]].append(r)
        ref_pixel_map[r["common_rgb_sha256"]].append(r)

    matches = []
    hard_cases = set()

    for s in sun_rows:
        for r in ref_file_map.get(s["file_sha256"], []):
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
            hard_cases.add(s["cluster_id"])

        for r in ref_pixel_map.get(
            s["common_rgb_sha256"],
            [],
        ):
            matches.append({
                "match_type": "EXACT_COMMON_RGB_SHA256",
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
            hard_cases.add(s["cluster_id"])

    return matches, hard_cases


def read_sun_full_arrays_once(
    tar_path: Path,
    member_names: set[str],
):
    """
    Load candidate SUN frames in one sequential TAR.GZ pass.
    Called only when the coarse near-duplicate screen found candidates.
    """
    if not member_names:
        return {}

    targets = set(member_names)
    out = {}

    with tarfile.open(tar_path, "r|gz") as tf:
        with tqdm(
            total=len(targets),
            desc="Load candidate SUN full RGB (single TAR pass)",
            unit="frame",
            dynamic_ncols=True,
        ) as bar:
            for m in tf:
                if not m.isfile():
                    continue

                name = m.name.replace("\\", "/")
                if name not in targets:
                    continue

                f = tf.extractfile(m)
                if f is None:
                    raise RuntimeError(
                        f"Could not read candidate SUN member: {name}"
                    )

                out[name] = decode_rgb_from_bytes(f.read())
                bar.update(1)

                if len(out) == len(targets):
                    break

    if set(out) != targets:
        missing = sorted(targets - set(out))
        raise RuntimeError(
            f"Candidate SUN frames missing from TAR: {len(missing)} "
            f"examples={missing[:10]}"
        )

    return out


def near_duplicate_audit(sun_rows, refs, tar_path: Path):
    refs_by_size = defaultdict(list)

    for r in refs:
        refs_by_size[(r["width"], r["height"])].append(r)

    soft_pairs = []

    for s in tqdm(
        sun_rows,
        desc="Near-duplicate coarse screen",
        unit="SUN frame",
        dynamic_ncols=True,
    ):
        pool = refs_by_size.get(
            (s["width"], s["height"]),
            [],
        )

        for r in pool:
            ham = (
                int(s["dhash64"]) ^ int(r["dhash64"])
            ).bit_count()

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

            soft_pairs.append({
                "sun": s,
                "ref": r,
                "dhash_hamming": ham,
                "thumb_mae": thumb_mae,
            })

    sun_candidate_members = {
        x["sun"]["frame_member"]
        for x in soft_pairs
    }
    sun_full = read_sun_full_arrays_once(
        tar_path,
        sun_candidate_members,
    )

    # Cache reference full RGB once per reference path.
    ref_full_cache = {}

    out = []
    hard_cases = set()

    for x in tqdm(
        soft_pairs,
        desc="Near-duplicate full-resolution verification",
        unit="pair",
        dynamic_ncols=True,
    ):
        s = x["sun"]
        r = x["ref"]

        a = sun_full[s["frame_member"]]

        if r["path"] not in ref_full_cache:
            ref_full_cache[r["path"]] = (
                decode_rgb_from_path(Path(r["path"]))
            )
        b = ref_full_cache[r["path"]]

        full_mae, full_psnr = full_metrics(a, b)

        hard = int(
            x["dhash_hamming"] <= HARD_DHASH_MAX
            and x["thumb_mae"] <= HARD_THUMB_MAE_MAX
            and full_mae <= HARD_FULL_MAE_MAX
            and full_psnr >= HARD_FULL_PSNR_MIN
        )

        if hard:
            hard_cases.add(s["cluster_id"])

        out.append({
            "match_type": (
                "HARD_PROBABLE_REENCODING"
                if hard
                else "SOFT_NEAR_DUPLICATE_CANDIDATE"
            ),
            "sun_pair_key": s["pair_key"],
            "sun_cluster_id": s["cluster_id"],
            "sun_frame_member": s["frame_member"],
            "reference_dataset": r["dataset"],
            "reference_id": r["ref_id"],
            "reference_path": r["path"],
            "dhash_hamming": x["dhash_hamming"],
            "thumb_mae": x["thumb_mae"],
            "full_mae": full_mae,
            "full_psnr": full_psnr,
            "hard_contamination": hard,
        })

    return out, hard_cases


def descriptor_row(s):
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
        "common_rgb_sha256": s["common_rgb_sha256"],
        "dhash64_hex": f"{int(s['dhash64']):016x}",
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sun-tar", type=Path, default=SUN_TAR)
    ap.add_argument(
        "--sun-manifest",
        type=Path,
        default=SUN_MANIFEST,
    )
    ap.add_argument("--sun-lock", type=Path, default=SUN_LOCK)
    ap.add_argument(
        "--r14b4b3-lock",
        type=Path,
        default=R14B4B3_LOCK,
    )

    ap.add_argument(
        "--s01-manifest",
        type=Path,
        default=S01_MANIFEST,
    )
    ap.add_argument(
        "--neopolyp-manifest",
        type=Path,
        default=NEO_MANIFEST,
    )
    ap.add_argument(
        "--polypgen-manifest",
        type=Path,
        default=POLYPGEN_MANIFEST,
    )
    ap.add_argument("--s01-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUT)

    args = ap.parse_args()

    print("===== Q1 R14B4B FIX3 SUN-SEG SELECTED RGB CONTAMINATION AUDIT =====")
    print("SUPERSEDES_FIX2=YES")
    print("SCIENTIFIC_PROTOCOL_CHANGE_FROM_FIX2=NO")
    print("TAR_ACCESS_OPTIMIZATION=SEQUENTIAL_SINGLE_PASS")
    print("NEOPOLYP_HASH_RECIPE_CORRECTED=YES")
    print("COMMON_CROSS_DATASET_RGB_HASH=SHA256_RGB_BYTES")
    print("SUN_SELECTED_FRAMES=980")
    print("SUN_PHYSICAL_CASES=49")
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
        args.r14b4b3_lock,
        args.s01_manifest,
        args.neopolyp_manifest,
        args.polypgen_manifest,
    )

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    # Lineage gates.
    print("\n===== LINEAGE / SHA GATES =====")

    sun_tar_sha = sha256_file_progress(args.sun_tar)
    if sun_tar_sha != EXPECTED_SUN_TAR_SHA:
        raise RuntimeError("SUN TAR SHA mismatch.")

    validate_manifest_sha(
        args.sun_manifest,
        EXPECTED_SUN_MANIFEST_SHA,
        "R14B3 SUN manifest",
    )
    validate_manifest_sha(
        args.s01_manifest,
        EXPECTED_S01_MANIFEST_SHA,
        "S01 manifest",
    )
    validate_manifest_sha(
        args.neopolyp_manifest,
        EXPECTED_NEO_MANIFEST_SHA,
        "NeoPolyp manifest",
    )
    validate_manifest_sha(
        args.polypgen_manifest,
        EXPECTED_POLYPGEN_MANIFEST_SHA,
        "PolypGen manifest",
    )

    validate_r14b4b3_lock(args.r14b4b3_lock)

    sun_cols, sun_rows = read_csv(args.sun_manifest)
    _, s01_rows = read_csv(args.s01_manifest)
    _, neo_rows = read_csv(args.neopolyp_manifest)
    _, pg_rows = read_csv(args.polypgen_manifest)

    if len(sun_rows) != EXPECTED_SUN_ROWS:
        raise RuntimeError(
            f"SUN selected rows {len(sun_rows)} "
            f"!= {EXPECTED_SUN_ROWS}"
        )

    sun_cases = {r["cluster_id"] for r in sun_rows}
    if len(sun_cases) != EXPECTED_SUN_CASES:
        raise RuntimeError(
            f"SUN selected cases {len(sun_cases)} "
            f"!= {EXPECTED_SUN_CASES}"
        )

    if len(s01_rows) != EXPECTED_S01_ROWS:
        raise RuntimeError("S01 row-count gate failed.")
    if len(neo_rows) != EXPECTED_NEO_ROWS:
        raise RuntimeError("NeoPolyp row-count gate failed.")
    if len(pg_rows) != EXPECTED_POLYPGEN_ROWS:
        raise RuntimeError("PolypGen row-count gate failed.")

    sun_lock = json.loads(
        args.sun_lock.read_text(encoding="utf-8")
    )

    if sun_lock.get("status") != "PASS":
        raise RuntimeError("R14B3 SUN lock is not PASS.")

    if (
        sun_lock
        .get("sampling", {})
        .get("selected_manifest_sha256")
        != EXPECTED_SUN_MANIFEST_SHA
    ):
        raise RuntimeError(
            "R14B3 lock selected-manifest SHA mismatch."
        )

    # Resolve S01 image root.
    s01_root, s01_root_validations = discover_s01_root(
        s01_rows,
        args.s01_root,
    )

    # Reference verification + common descriptors.
    print("\n===== REFERENCE LINEAGE / COMMON RGB HASH VERIFICATION =====")
    refs_s01 = load_reference_s01(
        s01_rows,
        s01_root,
    )
    refs_neo = load_reference_neopolyp(neo_rows)
    refs_pg = load_reference_polypgen(pg_rows)

    refs = refs_s01 + refs_neo + refs_pg

    print(
        "reference RGB images total:",
        len(refs),
    )

    expected_refs = (
        EXPECTED_S01_ROWS
        + EXPECTED_NEO_ROWS
        + EXPECTED_POLYPGEN_ROWS
    )

    if len(refs) != expected_refs:
        raise RuntimeError(
            f"Reference descriptor count {len(refs)} "
            f"!= {expected_refs}"
        )

    # Selected SUN access occurs only after all reference provenance gates pass.
    print("\n===== SUN SELECTED RGB READ / COMMON DESCRIPTORS =====")
    sun = load_selected_sun(
        sun_rows,
        args.sun_tar,
    )

    if len(sun) != EXPECTED_SUN_ROWS:
        raise RuntimeError(
            "SUN descriptor row count mismatch."
        )

    # Internal exact duplicate safeguard.
    internal_dups = internal_sun_exact_duplicate_audit(sun)

    print(
        "SUN internal exact duplicate groups:",
        len(internal_dups),
    )

    # Exact cross-dataset contamination.
    print("\n===== EXACT CROSS-DATASET CONTAMINATION =====")
    exact_matches, exact_cases = exact_cross_dataset_audit(
        sun,
        refs,
    )

    print(
        "exact match rows:",
        len(exact_matches),
    )
    print(
        "physical cases hit by exact matches:",
        len(exact_cases),
    )

    exact_dataset_counts = Counter(
        r["reference_dataset"]
        for r in exact_matches
    )
    for k, v in sorted(exact_dataset_counts.items()):
        print(f"  {k}: {v}")

    # Conservative near-duplicate screen.
    print("\n===== CONSERVATIVE RE-ENCODING SCREEN =====")
    near_rows, hard_reencode_cases = near_duplicate_audit(
        sun,
        refs,
        args.sun_tar,
    )

    hard_near_rows = [
        r for r in near_rows
        if int(r["hard_contamination"]) == 1
    ]
    soft_near_rows = [
        r for r in near_rows
        if int(r["hard_contamination"]) == 0
    ]

    print(
        "near-duplicate candidates total:",
        len(near_rows),
    )
    print(
        "hard probable re-encoding rows:",
        len(hard_near_rows),
    )
    print(
        "soft review candidates:",
        len(soft_near_rows),
    )
    print(
        "physical cases hit by hard re-encoding:",
        len(hard_reencode_cases),
    )

    contaminated_cases = (
        set(exact_cases)
        | set(hard_reencode_cases)
    )

    clean_sun_manifest_rows = [
        r for r in sun_rows
        if r["cluster_id"] not in contaminated_cases
    ]
    clean_cases = sorted({
        r["cluster_id"]
        for r in clean_sun_manifest_rows
    })

    selected_counts = Counter(
        r["cluster_id"]
        for r in sun_rows
    )
    clean_counts = Counter(
        r["cluster_id"]
        for r in clean_sun_manifest_rows
    )

    print("\n===== FROZEN CASE-LEVEL CONTAMINATION POLICY =====")
    print(
        "contaminated physical cases:",
        len(contaminated_cases),
    )
    print(
        "remaining physical cases:",
        len(clean_cases),
    )
    print(
        "remaining selected frames:",
        len(clean_sun_manifest_rows),
    )
    print(
        "replacement frames/cases added:",
        0,
    )

    # Output after all analyses complete.
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    descriptor_csv = (
        args.output_dir
        / "R14B4B_FIX3_SUN_SELECTED_RGB_HASHES.csv"
    )
    write_csv(
        descriptor_csv,
        [descriptor_row(s) for s in sun],
        [
            "pair_key",
            "cluster_id",
            "physical_case_id",
            "clip_id",
            "source_split_label",
            "frame_member",
            "width",
            "height",
            "file_sha256",
            "common_rgb_sha256",
            "dhash64_hex",
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

    exact_csv = (
        args.output_dir
        / "R14B4B_FIX3_EXACT_CONTAMINATION_MATCHES.csv"
    )
    write_csv(
        exact_csv,
        exact_matches,
        match_fields,
    )

    near_csv = (
        args.output_dir
        / "R14B4B_FIX3_NEAR_DUPLICATE_CANDIDATES.csv"
    )
    write_csv(
        near_csv,
        near_rows,
        match_fields,
    )

    internal_csv = (
        args.output_dir
        / "R14B4B_FIX3_SUN_INTERNAL_EXACT_DUPLICATES.csv"
    )
    write_csv(
        internal_csv,
        internal_dups,
        [
            "duplicate_type",
            "sha256",
            "count",
            "cluster_ids",
            "pair_keys",
        ],
    )

    final_manifest = (
        args.output_dir
        / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
    )
    write_csv(
        final_manifest,
        clean_sun_manifest_rows,
        sun_cols,
    )

    exclusion_rows = []

    for case_id in sorted(contaminated_cases):
        match_types = sorted({
            r["match_type"]
            for r in exact_matches + hard_near_rows
            if r["sun_cluster_id"] == case_id
        })

        exclusion_rows.append({
            "cluster_id": case_id,
            "precontamination_frame_count": selected_counts[case_id],
            "postcontamination_frame_count": clean_counts[case_id],
            "exclusion_reason": "|".join(match_types),
            "replacement": "NO",
        })

    exclusion_csv = (
        args.output_dir
        / "R14B4B_FIX3_CASE_EXCLUSIONS.csv"
    )
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

    root_csv = (
        args.output_dir
        / "R14B4B_FIX3_S01_ROOT_RESOLUTION.csv"
    )
    write_csv(
        root_csv,
        s01_root_validations,
        [
            "candidate_root",
            "missing_image_paths",
            "valid_all",
        ],
    )

    # Decision:
    # - any internal exact duplicate group requires protocol review;
    # - any unresolved soft candidate requires manual review;
    # - otherwise hard contaminated cases are removed whole-case/no-replacement
    #   and the remaining cohort is final.
    if internal_dups:
        status = "STOP"
        decision = (
            "SUNSEG_INTERNAL_EXACT_DUPLICATES_"
            "REQUIRE_PROTOCOL_REVIEW"
        )
        final_locked = False

    elif soft_near_rows:
        status = "PENDING_REVIEW"
        decision = DECISION_PENDING
        final_locked = False

    else:
        status = "PASS"
        decision = (
            DECISION_EXCLUDED
            if contaminated_cases
            else DECISION_PASS
        )
        final_locked = True

    final_manifest_sha = sha256_file_quiet(
        final_manifest
    )

    lock = {
        "status": status,
        "decision": decision,
        "supersedes": (
            "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix1"
        ),
        "fix_reason": (
            "Fix1 assumed NeoPolyp stored image_rgb_pixel_sha256 was "
            "SHA256(RGB bytes). R14B4B3 proved the historical recipe was "
            "SHA256(widthxheight|RGB| prefix + RGB bytes). Fix2 verifies "
            "the historical hash separately while using a common raw-RGB "
            "hash for cross-dataset exact comparison."
        ),
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
                "raw_file_sha_reverified": True,
                "stored_common_rgb_sha_reverified": True,
            },
            "NeoPolyp": {
                "manifest_sha256": EXPECTED_NEO_MANIFEST_SHA,
                "rows": EXPECTED_NEO_ROWS,
                "raw_file_sha_reverified": True,
                "historical_pixel_sha_reverified_1000_of_1000": True,
                "historical_recipe": (
                    'SHA256(f"{w}x{h}|RGB|".encode("ascii") + RGB_bytes)'
                ),
                "common_rgb_sha_computed_now_for_cross_dataset_comparison": True,
            },
            "PolypGen": {
                "manifest_sha256": EXPECTED_POLYPGEN_MANIFEST_SHA,
                "rows": EXPECTED_POLYPGEN_ROWS,
                "raw_file_sha_reverified": True,
                "common_rgb_sha_computed_now_for_cross_dataset_comparison": True,
            },
        },
        "cross_dataset_common_exact_hash": (
            "SHA256(contiguous uint8 RGB HxWx3 bytes)"
        ),
        "contamination_rule": {
            "exact_file_sha256": "HARD",
            "exact_common_rgb_sha256": "HARD",
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
            "remaining_frames": len(clean_sun_manifest_rows),
        },
        "final_confirmatory_cohort_locked": final_locked,
        "final_manifest_path": str(final_manifest),
        "final_manifest_sha256": final_manifest_sha,
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
            else "RESOLVE_R14B4B_FIX3_REVIEW_BEFORE_ANY_INFERENCE"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14B4B_FIX3_SUNSEG_CONTAMINATION_AUDIT_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== FINAL R14B4B FIX3 STATUS =====")
    print("status=", status)
    print("Decision=", decision)
    print(
        "final confirmatory cohort locked=",
        final_locked,
    )
    print(
        "remaining physical cases=",
        len(clean_cases),
    )
    print(
        "remaining selected frames=",
        len(clean_sun_manifest_rows),
    )
    print(
        "final manifest=",
        final_manifest,
    )
    print(
        "final manifest SHA256=",
        final_manifest_sha,
    )
    print(
        "LOCK=",
        lock_path,
    )

    if status == "PASS":
        print("PASS")
    elif status == "PENDING_REVIEW":
        print("PENDING_REVIEW")
    else:
        print("STOP")


if __name__ == "__main__":
    main()
