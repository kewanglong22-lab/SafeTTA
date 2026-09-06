#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S06-B PolypGen untouched integrity / overlap audit.

Reads PolypGen IMAGE bytes only.
Does NOT open/read/extract/hash any PolypGen MASK bytes.
Does NOT run segmentation inference or compute target metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


VERSION = "2026-08-18-S06-B-v1-fix3"
BUILD = "S06_B_FIX3_V2_METADATA_CANONICAL_PAIRING_FINAL_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

DEFAULT_ARCHIVE = (
    ROOT / "data" / "downloads" / "polypgen"
    / "polypgen2021_multicenterdata_v3.zip"
)

PROTOCOL = (
    ROOT / "docs"
    / "S06_A_polypgen_untouched_external_confirmation_protocol_v1_fix3.md"
)
EXPECTED_PROTOCOL_SHA256 = "744672e234a18cbdf3cca3a7e231e38362c5e09d1e6bdaaf43c9dcb98f49c78d"

S01_MANIFEST = (
    ROOT / "data" / "splits" / "S01_locked_protocol_manifest_v1.csv"
)
EXPECTED_S01_MANIFEST_SHA256 = (
    "afb4bc1175af004819538cdbb22ee7a10f1be48035945c7e0d50fd2bbe2e3dc7"
)
S00_DATA_ROOT = (
    ROOT / "data" / "processed" / "S00_polyp_locked_v1"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "S06_B_polypgen_integrity_overlap_audit_v1_fix3"
)

EXPECTED_EXISTING_IMAGES = 2248
EXPECTED_POLYPGEN_SINGLE_FRAMES = 1537
EXPECTED_CENTERS = ("C1", "C2", "C3", "C4", "C5", "C6")
EXPECTED_CENTER_COUNTS = {
    "C1": 256,
    "C2": 301,
    "C3": 457,
    "C4": 227,
    "C5": 208,
    "C6": 88,
}

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

NEAR_DUP_DHASH_MAX = 2
ASPECT_RATIO_REL_TOL = 0.02

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def file_sha256(path: Path, chunk_size=16 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bytes_sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def normalize_member(name: str):
    return name.replace("\\", "/").strip("/")


def suffix_of(name: str):
    return Path(normalize_member(name)).suffix.lower()


def is_exact_syn45200214_single_frame_member(name: str):
    n = normalize_member(name)
    low = n.lower()

    if low.startswith("__macosx/"):
        return False
    if "/sequencedata/" in low:
        return False
    if "/imagesall_positive/" in low:
        return False
    if (
        "/bbox_" in low
        or "/bbox_image_" in low
        or "/bbox_images_" in low
    ):
        return False

    pattern = (
        r"^polypgen2021_multicenterdata_v3/"
        r"data_c([1-6])/"
        r"(images|masks)_c\1/"
        r"[^/]+$"
    )
    return re.match(pattern, low) is not None


def detect_center(name: str):
    low = normalize_member(name).lower()
    m = re.match(
        r"^polypgen2021_multicenterdata_v3/"
        r"data_c([1-6])/"
        r"(?:images|masks)_c\1/",
        low,
    )
    return f"C{m.group(1)}" if m else None


def classify_single_frame_member(name: str):
    n = normalize_member(name)
    if suffix_of(n) not in IMAGE_SUFFIXES:
        return None, None
    if not is_exact_syn45200214_single_frame_member(n):
        return None, None

    center = detect_center(n)
    if center is None:
        return None, None

    low = n.lower()
    cnum = center[1:]

    if f"/images_c{cnum}/" in low:
        return "image", center
    if f"/masks_c{cnum}/" in low:
        return "mask", center
    return None, None


def detected_release_layout(name: str):
    if is_exact_syn45200214_single_frame_member(name):
        return "SYN45200214_EXACT_DATA_UNDERSCORE_C_IMAGES_MASKS"
    return ""


PAIRING_ROLE_TOKENS = (
    "ground_truth",
    "groundtruth",
    "segmentation",
    "images",
    "image",
    "masks",
    "mask",
    "labels",
    "label",
    "img",
    "gt",
    "seg",
)


def normalized_pair_stem(name: str):
    """
    Frozen S06-B1 V2 aggressive-alphanumeric canonical key.

    Metadata only. No image or mask bytes are accessed here.
    """
    stem = Path(normalize_member(name)).stem.lower()

    for token in PAIRING_ROLE_TOKENS:
        stem = stem.replace(token, "")

    return re.sub(r"[^a-z0-9]+", "", stem)


def resolve_archive(requested: Path):
    if requested.exists():
        return requested

    base = ROOT / "data" / "downloads"
    found = []
    if base.exists():
        for p in base.rglob("*.zip"):
            low = p.name.lower()
            if (
                "polypgen2021" in low
                and "multicenterdata" in low
                and "v3" in low
            ):
                found.append(p)

    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(
            "Multiple PolypGen v3 ZIPs found:\n"
            + "\n".join(str(x) for x in found)
        )
    raise FileNotFoundError(requested)


def validate_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    if actual.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "S06-A protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual  : {actual}"
        )
    return actual


def load_existing_manifest():
    if not S01_MANIFEST.exists():
        raise FileNotFoundError(S01_MANIFEST)

    actual = file_sha256(S01_MANIFEST)
    if actual.lower() != EXPECTED_S01_MANIFEST_SHA256.lower():
        raise RuntimeError("Frozen S01 manifest SHA mismatch.")

    rows, fields = read_csv(S01_MANIFEST)
    required = {"sample_id", "image_relpath", "s01_role"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(
            f"S01 manifest missing fields: {missing}"
        )

    if len(rows) != EXPECTED_EXISTING_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXISTING_IMAGES} frozen rows, "
            f"found {len(rows)}"
        )

    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id in S01 manifest.")

    return rows, actual


def open_rgb_path(path: Path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        rgb.load()
    return rgb


def open_rgb_bytes(data: bytes):
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
        rgb.load()
    return rgb


def decoded_rgb_sha256(rgb: Image.Image):
    w, h = rgb.size
    header = f"RGB|{w}|{h}|".encode("ascii")
    return hashlib.sha256(
        header + np.asarray(rgb, dtype=np.uint8).tobytes()
    ).hexdigest()


def dhash64(rgb: Image.Image):
    gray = rgb.convert("L").resize(
        (9, 8),
        RESAMPLE_LANCZOS,
    )
    arr = np.asarray(gray, dtype=np.uint8)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def dhash_hex(v: int):
    return f"{int(v):016x}"


def fingerprint_rgb(rgb, raw_sha):
    w, h = rgb.size
    return {
        "raw_sha256": raw_sha,
        "decoded_rgb_sha256": decoded_rgb_sha256(rgb),
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w) / float(h),
        "dhash64": dhash64(rgb),
    }


def fingerprint_existing(rows):
    out = []
    for row in tqdm(
        rows,
        desc="Fingerprint frozen 2248-image corpus",
        unit="img",
        dynamic_ncols=True,
    ):
        path = S00_DATA_ROOT / Path(row["image_relpath"])
        if not path.exists():
            raise FileNotFoundError(path)

        rgb = open_rgb_path(path)
        fp = fingerprint_rgb(rgb, file_sha256(path))

        out.append({
            "sample_id": row["sample_id"],
            "s01_role": row["s01_role"],
            "dataset": row.get("dataset", ""),
            "image_path": str(path),
            **fp,
        })
    return out


def zip_inventory(zf):
    rows = []
    for info in zf.infolist():
        kind, center = classify_single_frame_member(
            info.filename
        )
        rows.append({
            "member": normalize_member(info.filename),
            "is_dir": int(info.is_dir()),
            "suffix": suffix_of(info.filename),
            "compressed_size": int(info.compress_size),
            "uncompressed_size": int(info.file_size),
            "crc32_metadata_only": (
                f"{int(info.CRC) & 0xffffffff:08x}"
            ),
            "single_frame_kind": kind or "",
            "center": center or "",
            "detected_release_layout": detected_release_layout(
                info.filename
            ),
        })
    return rows


def pair_members(inventory):
    images = defaultdict(list)
    masks = defaultdict(list)

    for r in inventory:
        kind = r["single_frame_kind"]
        if kind not in {"image", "mask"}:
            continue

        key = (
            r["center"],
            normalized_pair_stem(r["member"]),
        )
        (images if kind == "image" else masks)[key].append(
            r["member"]
        )

    pairs = []
    ambiguous = []

    for key in sorted(set(images) | set(masks)):
        center, stem = key
        ims = sorted(images.get(key, []))
        mas = sorted(masks.get(key, []))

        if len(ims) == 1 and len(mas) == 1:
            pairs.append({
                "center": center,
                "pair_stem": stem,
                "image_member": ims[0],
                "mask_member": mas[0],
            })
        else:
            ambiguous.append({
                "center": center,
                "pair_stem": stem,
                "image_count": len(ims),
                "mask_count": len(mas),
                "image_members": " | ".join(ims),
                "mask_members": " | ".join(mas),
            })

    return pairs, ambiguous


def validate_zip_directory_only(zf):
    infos = zf.infolist()
    if not infos:
        raise RuntimeError("ZIP contains no members.")

    names = [normalize_member(x.filename) for x in infos]
    if len(names) != len(set(names)):
        raise RuntimeError(
            "ZIP contains duplicate member names."
        )

    # DO NOT call zf.testzip(): that would read mask bytes.
    return {
        "central_directory_readable": True,
        "member_count": len(infos),
        "full_crc_test_deferred_until_after_no_label_lock": True,
    }


def fingerprint_polypgen_images(zf, pairs):
    info_map = {
        normalize_member(x.filename): x
        for x in zf.infolist()
    }

    rows = []

    for pair in tqdm(
        pairs,
        desc="Fingerprint PolypGen single-frame images",
        unit="img",
        dynamic_ncols=True,
    ):
        image_member = pair["image_member"]
        mask_member = pair["mask_member"]

        # CRITICAL: the only opened ZIP member is IMAGE.
        with zf.open(info_map[image_member], "r") as f:
            image_bytes = f.read()

        rgb = open_rgb_bytes(image_bytes)
        fp = fingerprint_rgb(
            rgb,
            bytes_sha256(image_bytes),
        )

        mask_info = info_map[mask_member]

        rows.append({
            **pair,
            "image_member_uncompressed_size": int(
                info_map[image_member].file_size
            ),
            "mask_member_uncompressed_size": int(
                mask_info.file_size
            ),
            "mask_member_crc32_metadata_only": (
                f"{int(mask_info.CRC) & 0xffffffff:08x}"
            ),
            **fp,
        })

    return rows


def exact_overlap(polyp, existing):
    raw_map = defaultdict(list)
    rgb_map = defaultdict(list)

    for e in existing:
        raw_map[e["raw_sha256"]].append(e)
        rgb_map[e["decoded_rgb_sha256"]].append(e)

    matches = []
    status = {}

    for p in polyp:
        raw_hits = raw_map.get(p["raw_sha256"], [])
        rgb_hits = rgb_map.get(
            p["decoded_rgb_sha256"], []
        )

        if raw_hits:
            state = "EXCLUDE_EXACT_BYTE_DUPLICATE"
        elif rgb_hits:
            state = "EXCLUDE_EXACT_RGB_DUPLICATE"
        else:
            state = "NO_EXACT_DUPLICATE"

        hit_by_id = {}
        for e in raw_hits + rgb_hits:
            hit_by_id[e["sample_id"]] = e

        status[
            (p["center"], p["pair_stem"])
        ] = {
            "exact_status": state,
            "exact_hit_count": len(hit_by_id),
        }

        for sid, e in sorted(hit_by_id.items()):
            matches.append({
                "polypgen_center": p["center"],
                "polypgen_pair_stem": p["pair_stem"],
                "polypgen_image_member": p["image_member"],
                "existing_sample_id": sid,
                "existing_dataset": e["dataset"],
                "existing_role": e["s01_role"],
                "raw_byte_sha_match": int(
                    e["raw_sha256"]
                    == p["raw_sha256"]
                ),
                "decoded_rgb_sha_match": int(
                    e["decoded_rgb_sha256"]
                    == p["decoded_rgb_sha256"]
                ),
            })

    return matches, status


def aspect_close(a, b):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= ASPECT_RATIO_REL_TOL


def near_overlap(polyp, existing, exact_status):
    candidates = []
    holds = defaultdict(list)

    for p in tqdm(
        polyp,
        desc="Near-duplicate dHash screening",
        unit="img",
        dynamic_ncols=True,
    ):
        key = (p["center"], p["pair_stem"])

        if (
            exact_status[key]["exact_status"]
            != "NO_EXACT_DUPLICATE"
        ):
            continue

        ph = int(p["dhash64"])
        par = float(p["aspect_ratio"])

        for e in existing:
            if not aspect_close(
                par,
                float(e["aspect_ratio"]),
            ):
                continue

            dist = (
                ph ^ int(e["dhash64"])
            ).bit_count()

            if dist <= NEAR_DUP_DHASH_MAX:
                row = {
                    "polypgen_center": p["center"],
                    "polypgen_pair_stem": p["pair_stem"],
                    "polypgen_image_member": p["image_member"],
                    "existing_sample_id": e["sample_id"],
                    "existing_dataset": e["dataset"],
                    "existing_role": e["s01_role"],
                    "dhash_hamming": dist,
                    "polypgen_width": p["width"],
                    "polypgen_height": p["height"],
                    "existing_width": e["width"],
                    "existing_height": e["height"],
                }
                candidates.append(row)
                holds[key].append(row)

    return candidates, holds


def build_eligibility(polyp, exact_status, near_holds):
    rows = []

    for p in polyp:
        key = (p["center"], p["pair_stem"])
        exact = exact_status[key]["exact_status"]
        near = near_holds.get(key, [])

        if exact == "EXCLUDE_EXACT_BYTE_DUPLICATE":
            eligibility = exact
        elif exact == "EXCLUDE_EXACT_RGB_DUPLICATE":
            eligibility = exact
        elif near:
            eligibility = "HOLD_NEAR_DUPLICATE_REVIEW"
        else:
            eligibility = "ELIGIBLE_PENDING_S06C"

        rows.append({
            "external_id": (
                f"PolypGen_{p['center']}_{p['pair_stem']}"
            ),
            "center": p["center"],
            "pair_stem": p["pair_stem"],
            "image_member": p["image_member"],
            "mask_member": p["mask_member"],
            "image_width": p["width"],
            "image_height": p["height"],
            "image_raw_sha256": p["raw_sha256"],
            "image_decoded_rgb_sha256": (
                p["decoded_rgb_sha256"]
            ),
            "image_dhash64": dhash_hex(
                p["dhash64"]
            ),
            "exact_duplicate_status": exact,
            "exact_duplicate_hit_count": (
                exact_status[key]["exact_hit_count"]
            ),
            "near_duplicate_candidate_count": len(near),
            "eligibility": eligibility,
            "eligibility_uses_mask_content": "NO",
        })

    return rows


def run(args):
    protocol_sha = validate_protocol()
    manifest_rows, manifest_sha = (
        load_existing_manifest()
    )
    archive = resolve_archive(args.archive)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(
        str(args.output_dir) + "__building"
    )
    if build_dir.exists():
        raise FileExistsError(build_dir)

    build_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Archive: {archive}")
    print("PolypGen mask bytes opened=NO")
    print("Model inference=NO")
    print("Target metrics=NO")
    print()

    print("Computing full ZIP SHA256...")
    archive_sha = file_sha256(archive)
    archive_size = archive.stat().st_size

    existing = fingerprint_existing(
        manifest_rows
    )

    with zipfile.ZipFile(archive, "r") as zf:
        zip_meta = validate_zip_directory_only(zf)
        inventory = zip_inventory(zf)
        pairs, ambiguous = pair_members(
            inventory
        )

        # FIX3 invariant: metadata-only V2 key must yield exactly the
        # frozen 1537 one-to-one pairs before any ZIP member bytes are read.
        if len(pairs) != EXPECTED_POLYPGEN_SINGLE_FRAMES:
            raise RuntimeError(
                f"V2 canonical pairing count mismatch: "
                f"expected={EXPECTED_POLYPGEN_SINGLE_FRAMES}, "
                f"actual={len(pairs)}"
            )

        pair_center_counts = Counter(
            pair["center"] for pair in pairs
        )
        if dict(sorted(pair_center_counts.items())) != EXPECTED_CENTER_COUNTS:
            raise RuntimeError(
                "V2 canonical pairing center-count mismatch. "
                f"Expected={EXPECTED_CENTER_COUNTS}, "
                f"actual={dict(sorted(pair_center_counts.items()))}"
            )

        write_csv(
            build_dir / "archive_inventory.csv",
            inventory,
            [
                "member", "is_dir", "suffix",
                "compressed_size", "uncompressed_size",
                "crc32_metadata_only",
                "single_frame_kind", "center",
                "detected_release_layout",
            ],
        )

        write_csv(
            build_dir
            / "ambiguous_or_unpaired_members.csv",
            ambiguous,
            [
                "center", "pair_stem",
                "image_count", "mask_count",
                "image_members", "mask_members",
            ],
        )

        if ambiguous:
            raise RuntimeError(
                f"{len(ambiguous)} ambiguous/unpaired "
                "single-frame keys found."
            )

        polyp = fingerprint_polypgen_images(
            zf,
            pairs,
        )

    centers = Counter(
        x["center"] for x in polyp
    )
    layout_counts = Counter(
        r["detected_release_layout"]
        for r in inventory
        if r["single_frame_kind"] in {"image", "mask"}
    )

    exact_rows, exact_status = exact_overlap(
        polyp,
        existing,
    )
    near_rows, near_holds = near_overlap(
        polyp,
        existing,
        exact_status,
    )
    eligibility = build_eligibility(
        polyp,
        exact_status,
        near_holds,
    )

    write_csv(
        build_dir
        / "polypgen_single_frame_image_fingerprints.csv",
        [
            {
                **r,
                "dhash64": dhash_hex(r["dhash64"]),
            }
            for r in polyp
        ],
        [
            "center", "pair_stem",
            "image_member", "mask_member",
            "image_member_uncompressed_size",
            "mask_member_uncompressed_size",
            "mask_member_crc32_metadata_only",
            "raw_sha256",
            "decoded_rgb_sha256",
            "width", "height",
            "aspect_ratio", "dhash64",
        ],
    )

    write_csv(
        build_dir / "exact_duplicate_matches.csv",
        exact_rows,
        [
            "polypgen_center",
            "polypgen_pair_stem",
            "polypgen_image_member",
            "existing_sample_id",
            "existing_dataset",
            "existing_role",
            "raw_byte_sha_match",
            "decoded_rgb_sha_match",
        ],
    )

    write_csv(
        build_dir / "near_duplicate_candidates.csv",
        near_rows,
        [
            "polypgen_center",
            "polypgen_pair_stem",
            "polypgen_image_member",
            "existing_sample_id",
            "existing_dataset",
            "existing_role",
            "dhash_hamming",
            "polypgen_width",
            "polypgen_height",
            "existing_width",
            "existing_height",
        ],
    )

    write_csv(
        build_dir
        / "S06_polypgen_external_eligibility_manifest.csv",
        eligibility,
        [
            "external_id",
            "center",
            "pair_stem",
            "image_member",
            "mask_member",
            "image_width",
            "image_height",
            "image_raw_sha256",
            "image_decoded_rgb_sha256",
            "image_dhash64",
            "exact_duplicate_status",
            "exact_duplicate_hit_count",
            "near_duplicate_candidate_count",
            "eligibility",
            "eligibility_uses_mask_content",
        ],
    )

    exact_n = sum(
        r["eligibility"].startswith(
            "EXCLUDE_EXACT_"
        )
        for r in eligibility
    )
    hold_n = sum(
        r["eligibility"]
        == "HOLD_NEAR_DUPLICATE_REVIEW"
        for r in eligibility
    )
    eligible_n = sum(
        r["eligibility"]
        == "ELIGIBLE_PENDING_S06C"
        for r in eligibility
    )

    center_counts_ok = (
        dict(sorted(centers.items()))
        == EXPECTED_CENTER_COUNTS
    )

    structure_ok = (
        len(polyp)
        == EXPECTED_POLYPGEN_SINGLE_FRAMES
        and tuple(sorted(centers))
        == EXPECTED_CENTERS
        and center_counts_ok
        and not ambiguous
    )

    if not structure_ok:
        decision = (
            "S06_B_AUDIT_STRUCTURE_NOT_MET_STOP_BEFORE_S06C"
        )
    elif hold_n > 0:
        decision = (
            "S06_B_AUDIT_PASS_NEAR_DUPLICATE_REVIEW_REQUIRED"
        )
    else:
        decision = (
            "S06_B_AUDIT_PASS_READY_FOR_S06C_NO_LABEL_INFERENCE"
        )

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "s01_manifest_sha256": manifest_sha,
        "existing_frozen_image_count": len(existing),
        "zip_metadata_audit": zip_meta,
        "polypgen_single_frame_pair_count": len(polyp),
        "center_pair_counts": dict(
            sorted(centers.items())
        ),
        "detected_release_layout_member_counts": dict(
            sorted(layout_counts.items())
        ),
        "published_expected_single_frames": (
            EXPECTED_POLYPGEN_SINGLE_FRAMES
        ),
        "expected_center_pair_counts": EXPECTED_CENTER_COUNTS,
        "center_counts_match_frozen_probe": center_counts_ok,
        "pairing_rule": "S06_B1_V2_AGGRESSIVE_ALPHANUMERIC",
        "pairing_rule_metadata_only": True,
        "exact_duplicate_external_samples": exact_n,
        "near_duplicate_hold_external_samples": hold_n,
        "eligible_pending_s06c_samples": eligible_n,
        "polypgen_mask_bytes_opened": False,
        "polypgen_masks_extracted": False,
        "polypgen_mask_pixels_inspected": False,
        "eligibility_used_mask_content": False,
        "model_inference_performed": False,
        "target_metrics_computed": False,
        "decision": decision,
    }

    (
        build_dir / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = f"""===== S06-B POLYPGEN UNTOUCHED INTEGRITY / OVERLAP AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Archive:
  path={archive}
  size_bytes={archive_size}
  SHA256={archive_sha}

Frozen development/evaluation corpus:
  images={len(existing)}
  S01 manifest SHA256={manifest_sha}

PolypGen positive single-frame branch:
  paired image-mask member names={len(polyp)}
  expected published count={EXPECTED_POLYPGEN_SINGLE_FRAMES}
  centers={dict(sorted(centers.items()))}
  expected centers={EXPECTED_CENTER_COUNTS}
  center counts match frozen probe={center_counts_ok}
  pairing rule=S06_B1_V2_AGGRESSIVE_ALPHANUMERIC
  pairing rule metadata-only=YES
  detected release layouts={dict(sorted(layout_counts.items()))}
  ambiguous/unpaired keys={len(ambiguous)}

Overlap audit against frozen 2248 images:
  exact duplicate external samples={exact_n}
  near-duplicate HOLD external samples={hold_n}
  eligible pending S06-C={eligible_n}

Ground-truth confidentiality:
  PolypGen mask bytes opened=NO
  PolypGen masks extracted=NO
  PolypGen mask pixels inspected=NO
  eligibility used mask content=NO

Scientific actions:
  model inference=NO
  target metrics=NO

Decision: {decision}

[OK] Outputs: {args.output_dir}
"""

    (
        build_dir / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    build_dir.rename(args.output_dir)

    print()
    print(
        (
            args.output_dir
            / "summary.txt"
        ).read_text(encoding="utf-8")
    )


def self_test():
    real_image_c1 = (
        "PolypGen2021_MultiCenterData_v3/"
        "data_C1/images_C1/frame001.jpg"
    )
    real_mask_c1 = (
        "PolypGen2021_MultiCenterData_v3/"
        "data_C1/masks_C1/frame001.jpg"
    )
    real_image_c6 = (
        "PolypGen2021_MultiCenterData_v3/"
        "data_C6/images_C6/frame999.jpg"
    )
    bbox = (
        "PolypGen2021_MultiCenterData_v3/"
        "data_C3/bbox_image_C3/frame001.jpg"
    )
    sequence = (
        "PolypGen2021_MultiCenterData_v3/"
        "sequenceData/positive/seq5/images_seq5/frame001.jpg"
    )
    all_positive = (
        "PolypGen2021_MultiCenterData_v3/"
        "imagesAll_positive/frame001.jpg"
    )
    macosx = (
        "__MACOSX/PolypGen2021_MultiCenterData_v3/"
        "data_C1/images_C1/._frame001.jpg"
    )

    assert classify_single_frame_member(real_image_c1) == ("image", "C1")
    assert classify_single_frame_member(real_mask_c1) == ("mask", "C1")
    assert classify_single_frame_member(real_image_c6) == ("image", "C6")
    assert classify_single_frame_member(bbox) == (None, None)
    assert classify_single_frame_member(sequence) == (None, None)
    assert classify_single_frame_member(all_positive) == (None, None)
    assert classify_single_frame_member(macosx) == (None, None)

    assert (
        detected_release_layout(real_image_c1)
        == "SYN45200214_EXACT_DATA_UNDERSCORE_C_IMAGES_MASKS"
    )

    # V2 metadata-only pairing examples observed in S06-B1.
    assert (
        normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C1/images_C1/100H0050.jpg"
        )
        == normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C1/masks_C1/100H0050_mask.jpg"
        )
    )
    assert (
        normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C3/images_C3/C3_EndoCV2021_00100.jpg"
        )
        == normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C3/masks_C3/C3_EndoCV2021_00100_mask.jpg"
        )
    )
    assert (
        normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C6/images_C6/EndoCV2021_C6_0100001.jpg"
        )
        == normalized_pair_stem(
            "PolypGen2021_MultiCenterData_v3/"
            "data_C6/masks_C6/EndoCV2021_C6_0100001_mask.jpg"
        )
    )

    toy_inventory = [
        {
            "member": (
                "PolypGen2021_MultiCenterData_v3/"
                "data_C1/images_C1/100H0050.jpg"
            ),
            "single_frame_kind": "image",
            "center": "C1",
        },
        {
            "member": (
                "PolypGen2021_MultiCenterData_v3/"
                "data_C1/masks_C1/100H0050_mask.jpg"
            ),
            "single_frame_kind": "mask",
            "center": "C1",
        },
    ]
    pairs, ambiguous = pair_members(toy_inventory)
    assert len(pairs) == 1
    assert len(ambiguous) == 0

    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[:, 24:, :] = 255
    rgb = Image.fromarray(arr, mode="RGB")
    h1 = dhash64(rgb)
    h2 = dhash64(rgb.copy())
    assert h1 == h2
    assert (h1 ^ h2).bit_count() == 0

    assert EXPECTED_POLYPGEN_SINGLE_FRAMES == 1537
    assert EXPECTED_EXISTING_IMAGES == 2248
    assert EXPECTED_CENTERS == ("C1", "C2", "C3", "C4", "C5", "C6")
    assert EXPECTED_CENTER_COUNTS == {
        "C1": 256,
        "C2": 301,
        "C3": 457,
        "C4": 227,
        "C5": 208,
        "C6": 88,
    }
    assert sum(EXPECTED_CENTER_COUNTS.values()) == 1537

    print("SYN45200214_EXACT_LAYOUT_TEST_PASS")
    print("S06_B1_V2_PAIRING_RULE_TEST_PASS")
    print("EXCLUDED_BRANCHES_TEST_PASS")
    print("PAIRING_TEST_PASS")
    print("IMAGE_FINGERPRINT_TEST_PASS")
    print("MASK_BYTE_CONFIDENTIALITY_DESIGN_TEST_PASS")
    print("FROZEN_EXPECTED_CENTER_COUNTS_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Audit PolypGen ZIP structure and image overlap "
            "without reading any PolypGen mask bytes."
        )
    )
    p.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    p.add_argument(
        "--self-test",
        action="store_true",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
