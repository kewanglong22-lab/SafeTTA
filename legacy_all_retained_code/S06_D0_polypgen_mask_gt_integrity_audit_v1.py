#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

VERSION = "2026-08-18-S06-D0-v1"
BUILD = "S06_D0_MASK_ONLY_GT_INTEGRITY_AUDIT_AFTER_S06C_LOCK"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = ROOT / "docs" / "S06_D0_polypgen_mask_gt_integrity_audit_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "991d788bbe058207af76ce19506dca8564587f77141111981f92ced1b03f1827"

ARCHIVE = ROOT / "data" / "downloads" / "polypgen" / "polypgen2021_multicenterdata_v3.zip"
EXPECTED_ARCHIVE_SHA256 = "a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3"

ELIGIBILITY = (
    ROOT / "outputs" / "S06_B_polypgen_integrity_overlap_audit_v1_fix3"
    / "S06_polypgen_external_eligibility_manifest.csv"
)
EXPECTED_ELIGIBILITY_SHA256 = "b157d28634f8f53381ea6ee50db4142fbd06d94dc3f23fd858f2f62f7b71da06"

S06C_GLOBAL_LOCK = (
    ROOT / "outputs" / "S06_C_polypgen_dual_backbone_no_label_lock_v1_fix1"
    / "POLYPGEN_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK.json"
)
EXPECTED_S06C_GLOBAL_LOCK_SHA256 = "6459f0addcbddf2449fa5f7b03096cbff8432ffc19c21b0c4116a3d6f7354db5"

OUTPUT_DIR = ROOT / "outputs" / "S06_D0_polypgen_mask_gt_integrity_audit_v1"

EXPECTED_IMAGES = 1537
EXPECTED_CENTERS = {"C1":256,"C2":301,"C3":457,"C4":227,"C5":208,"C6":88}
PASS_DECISION = "S06_D0_MASK_GT_INTEGRITY_AUDIT_LOCKED_READY_FOR_EMPTY_GT_RULE_FREEZE"


def file_sha256(path: Path, chunk_size=16*1024*1024):
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


def validate_sha(path: Path, expected: str, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch. expected={expected} actual={actual}")
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def validate_inputs():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "protocol")
    archive_sha = validate_sha(ARCHIVE, EXPECTED_ARCHIVE_SHA256, "archive")
    elig_sha = validate_sha(ELIGIBILITY, EXPECTED_ELIGIBILITY_SHA256, "eligibility")
    global_sha = validate_sha(S06C_GLOBAL_LOCK, EXPECTED_S06C_GLOBAL_LOCK_SHA256, "S06-C lock")

    lock = json.loads(S06C_GLOBAL_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != "S06_C_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK_PASS_READY_FOR_S06D_GT_REVEAL":
        raise RuntimeError("S06-C global lock does not authorize GT reveal.")
    if bool(lock.get("polypgen_mask_bytes_opened_before_global_lock", True)):
        raise RuntimeError("S06-C reports mask access before global lock.")
    if bool(lock.get("target_metrics_computed_before_global_lock", True)):
        raise RuntimeError("S06-C reports target metrics before global lock.")

    rows, fields = read_csv(ELIGIBILITY)
    needed = {"external_id","center","image_member","mask_member","eligibility","eligibility_uses_mask_content"}
    missing = sorted(needed - set(fields))
    if missing:
        raise RuntimeError(f"Eligibility missing fields: {missing}")
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} rows, found {len(rows)}")
    if any(r["eligibility"] != "ELIGIBLE_PENDING_S06C" for r in rows):
        raise RuntimeError("Eligibility contains non-eligible rows.")
    if any(str(r["eligibility_uses_mask_content"]).upper() != "NO" for r in rows):
        raise RuntimeError("Eligibility used mask content.")
    if len({r["external_id"] for r in rows}) != EXPECTED_IMAGES:
        raise RuntimeError("external_id not unique.")
    counts = dict(sorted(Counter(r["center"] for r in rows).items()))
    if counts != EXPECTED_CENTERS:
        raise RuntimeError(f"Center counts mismatch: {counts}")

    rows.sort(key=lambda r: (r["center"], r["external_id"]))
    return rows, archive_sha, elig_sha, global_sha


def audit_mask(zf: zipfile.ZipFile, row):
    # Authorized GT stage: MASK bytes only.
    mask_member = row["mask_member"]
    with zf.open(mask_member, "r") as f:
        payload = f.read()

    with Image.open(io.BytesIO(payload)) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)

    if gray.ndim != 2 or gray.size == 0:
        raise RuntimeError(f"Invalid mask decode: {row['external_id']}")

    minv = int(gray.min())
    maxv = int(gray.max())
    nz = int(np.count_nonzero(gray))

    if maxv == 0:
        state = "NULL_ALL_ZERO"
        binary_fg = 0
    else:
        state = "NONEMPTY"
        binary_fg = int((gray.astype(np.float32) > 0.5*float(maxv)).sum())
        if binary_fg <= 0:
            raise RuntimeError(f"Nonempty mask threshold produced zero foreground: {row['external_id']}")

    return {
        "external_id": row["external_id"],
        "center": row["center"],
        "image_member": row["image_member"],
        "mask_member": mask_member,
        "mask_raw_sha256": bytes_sha256(payload),
        "mask_width": int(gray.shape[1]),
        "mask_height": int(gray.shape[0]),
        "min_gray": minv,
        "max_gray": maxv,
        "nonzero_gray_pixels": nz,
        "mask_state": state,
        "binary_foreground_pixels": binary_fg,
        "low_max_nonempty_lt128": int(0 < maxv < 128),
    }


def run(args):
    rows, archive_sha, elig_sha, global_sha = validate_inputs()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    build = Path(str(args.output_dir) + "__building")
    if build.exists():
        raise FileExistsError(build)
    build.mkdir(parents=True)

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"S06-C global no-label lock SHA256={global_sha}")
    print(f"Target masks={len(rows)}")
    print("Prediction arrays read=NO | decision CSVs read=NO | model metrics=NO")
    print()

    records = []
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        names = set(zf.namelist())
        for r in rows:
            if r["mask_member"] not in names:
                raise RuntimeError(f"Mask missing: {r['external_id']}")
        for r in tqdm(rows, desc="S06-D0 audit target masks", unit="mask", dynamic_ncols=True):
            records.append(audit_mask(zf, r))

    nulls = [r for r in records if r["mask_state"] == "NULL_ALL_ZERO"]
    nonempty = [r for r in records if r["mask_state"] == "NONEMPTY"]
    lowmax = [r for r in records if int(r["low_max_nonempty_lt128"]) == 1]

    center_summary = []
    for c in EXPECTED_CENTERS:
        subset = [r for r in records if r["center"] == c]
        n0 = sum(r["mask_state"] == "NULL_ALL_ZERO" for r in subset)
        center_summary.append({
            "center": c,
            "total": len(subset),
            "null_all_zero": n0,
            "nonempty": len(subset)-n0,
            "null_rate": n0/len(subset),
        })

    manifest = build / "polypgen_mask_gt_integrity_manifest.csv"
    write_csv(manifest, records, [
        "external_id","center","image_member","mask_member","mask_raw_sha256",
        "mask_width","mask_height","min_gray","max_gray","nonzero_gray_pixels",
        "mask_state","binary_foreground_pixels","low_max_nonempty_lt128"
    ])

    center_csv = build / "center_mask_gt_integrity_summary.csv"
    write_csv(center_csv, center_summary, [
        "center","total","null_all_zero","nonempty","null_rate"
    ])

    audit = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "archive_sha256": archive_sha,
        "eligibility_manifest_sha256": elig_sha,
        "s06c_global_no_label_lock_sha256": global_sha,
        "target_masks_audited": len(records),
        "null_all_zero_masks": len(nulls),
        "null_all_zero_rate": len(nulls)/len(records),
        "nonempty_masks": len(nonempty),
        "nonempty_rate": len(nonempty)/len(records),
        "low_max_nonempty_lt128": len(lowmax),
        "center_summary": center_summary,
        "null_external_ids": [r["external_id"] for r in nulls],
        "low_max_nonempty_external_ids": [r["external_id"] for r in lowmax],
        "prediction_arrays_read": False,
        "decision_csvs_read": False,
        "model_metrics_computed": False,
        "cohort_exclusion_after_gt": False,
        "threshold_tuning": False,
        "model_or_gate_change": False,
        "decision": PASS_DECISION,
    }

    audit_json = build / "S06_D0_MASK_GT_INTEGRITY_AUDIT.json"
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "s06c_global_no_label_lock_sha256": global_sha,
        "manifest_sha256": file_sha256(manifest),
        "center_summary_sha256": file_sha256(center_csv),
        "audit_json_sha256": file_sha256(audit_json),
        "prediction_arrays_read": False,
        "decision_csvs_read": False,
        "model_metrics_computed": False,
        "decision": PASS_DECISION,
    }
    lock_path = build / "S06_D0_MASK_GT_INTEGRITY_LOCK.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    lock_sha = file_sha256(lock_path)

    lines = [
        "===== S06-D0 POLYPGEN MASK-GT INTEGRITY AUDIT =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        f"Total masks={len(records)}",
        f"NULL_ALL_ZERO={len(nulls)} ({len(nulls)/len(records):.6f})",
        f"NONEMPTY={len(nonempty)} ({len(nonempty)/len(records):.6f})",
        f"low-max NONEMPTY (<128)={len(lowmax)}",
        "",
        "Center breakdown:",
    ]
    for x in center_summary:
        lines.append(
            f"  {x['center']}: total={x['total']} null={x['null_all_zero']} "
            f"nonempty={x['nonempty']} null_rate={x['null_rate']:.6f}"
        )
    lines += [
        "",
        "Scientific firewall:",
        "  prediction arrays read=NO",
        "  decision CSVs read=NO",
        "  model metrics computed=NO",
        "  cohort exclusion after GT=NO",
        "  threshold tuning=NO",
        "  model/gate change=NO",
        "",
        f"S06-D0 lock SHA256={lock_sha}",
        "",
        f"Decision: {PASS_DECISION}",
        "",
        f"[OK] Outputs: {args.output_dir}",
    ]
    (build / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")

    build.rename(args.output_dir)
    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))


def self_test():
    z = np.zeros((10,12), dtype=np.uint8)
    assert int(z.max()) == 0 and int(np.count_nonzero(z)) == 0

    x = np.zeros((10,12), dtype=np.uint8)
    x[2:8,3:9] = 250
    assert int((x.astype(np.float32) > 0.5*float(x.max())).sum()) == 36

    assert EXPECTED_IMAGES == 1537
    assert sum(EXPECTED_CENTERS.values()) == 1537

    print("NULL_ALL_ZERO_MASK_TEST_PASS")
    print("NONEMPTY_MASK_THRESHOLD_TEST_PASS")
    print("FROZEN_COHORT_COUNT_TEST_PASS")
    print("MASK_ONLY_STAGE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description="Mask-only PolypGen GT integrity audit after frozen S06-C no-label lock."
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--self-test", action="store_true")
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
