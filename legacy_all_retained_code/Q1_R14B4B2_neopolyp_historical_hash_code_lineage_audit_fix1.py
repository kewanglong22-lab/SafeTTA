#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14B4B2_neopolyp_historical_hash_code_lineage_audit_fix1.py

Recover the exact historical code that generated NeoPolyp
`image_rgb_pixel_sha256` in the authoritative R05D1 pipeline.

This stage reads source/lock text only.

NO image pixels.
NO GT pixels.
NO SUN-SEG RGB.
NO contamination comparison.
NO model/TTA inference.
NO frozen-cohort modification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_CODE_ROOT = ROOT / "code"
DEFAULT_R05D1_OUT = (
    ROOT / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
)
DEFAULT_OUT = (
    ROOT / "outputs"
    / "Q1_R14B4B2_neopolyp_historical_hash_code_lineage_audit_fix1_v1"
)

TARGET_SCRIPT_BASENAMES = (
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2.py",
    "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1.py",
)

TARGET_TOKENS = (
    "image_rgb_pixel_sha256",
    "rgb_pixel_sha256",
    "pixel_sha256",
    "sha256",
    "tobytes",
    'convert("RGB")',
    "convert('RGB')",
    "cv2.imread",
    "Image.open",
)

TEXT_SUFFIXES = {".py", ".json", ".md", ".txt"}

DECISION_FOUND = (
    "NEOPOLYP_HISTORICAL_PIXEL_HASH_CODE_LOCATED_FOR_EXACT_RECONSTRUCTION"
)
DECISION_NOT_FOUND = (
    "NEOPOLYP_HISTORICAL_PIXEL_HASH_CODE_NOT_LOCATED_STOP"
)


def sha256_file(path: Path, chunk=4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_read_text(path: Path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
        except OSError:
            return None, None
    return None, None


def find_candidate_scripts(code_root: Path):
    found = []

    for basename in TARGET_SCRIPT_BASENAMES:
        p = code_root / basename
        if p.is_file():
            found.append(p)

    if code_root.is_dir():
        for p in code_root.rglob("*.py"):
            low = p.name.lower()
            if "r05d1" in low and "neopolyp" in low:
                found.append(p)

        for p in code_root.rglob("*.py"):
            text, _ = safe_read_text(p)
            if text and "image_rgb_pixel_sha256" in text:
                found.append(p)

    return sorted(set(found), key=lambda x: str(x).lower())


def find_support_files(r05d1_out: Path):
    hits = []
    if not r05d1_out.is_dir():
        return hits

    for p in r05d1_out.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if p.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue

        text, _ = safe_read_text(p)
        if not text:
            continue

        if (
            "image_rgb_pixel_sha256" in text
            or "pixel_sha256" in text
            or "script_sha256" in text
            or "Q1_R05D1" in text
        ):
            hits.append(p)

    return sorted(set(hits), key=lambda x: str(x).lower())


def line_context(text: str, token: str, radius=18):
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if token in line:
            lo = max(0, i - radius)
            hi = min(len(lines), i + radius + 1)
            out.append((i + 1, lo + 1, hi, lines[lo:hi]))
    return out


def function_blocks_containing_token(text: str, token: str):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    blocks = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue

        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue

        lo = node.lineno - 1
        hi = node.end_lineno
        block = "\n".join(lines[lo:hi])

        if token in block:
            blocks.append({
                "kind": type(node).__name__,
                "name": getattr(node, "name", "<unnamed>"),
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "text": block,
            })

    blocks.sort(
        key=lambda x: (
            x["end_line"] - x["start_line"],
            x["start_line"],
        )
    )
    return blocks


def extract_hash_related_context(path: Path):
    text, enc = safe_read_text(path)

    if text is None:
        return {
            "path": str(path),
            "readable": False,
            "encoding": None,
            "sha256": sha256_file(path),
            "matches": [],
            "blocks": [],
        }

    matches = []
    for token in TARGET_TOKENS:
        for line_no, lo, hi, ctx in line_context(text, token):
            matches.append({
                "token": token,
                "line_no": line_no,
                "context_start": lo,
                "context_end": hi,
                "context": "\n".join(ctx),
            })

    blocks = []
    for token in (
        "image_rgb_pixel_sha256",
        "rgb_pixel_sha256",
        "pixel_sha256",
    ):
        for b in function_blocks_containing_token(text, token):
            b = dict(b)
            b["token"] = token
            blocks.append(b)

    seen = set()
    unique_blocks = []
    for b in blocks:
        key = (b["start_line"], b["end_line"], b["name"])
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return {
        "path": str(path),
        "readable": True,
        "encoding": enc,
        "sha256": sha256_file(path),
        "matches": matches,
        "blocks": unique_blocks,
    }


def exact_assignment_contexts(text: str):
    lines = text.splitlines()
    evidence = []

    rgx = re.compile(
        r"(image_rgb_pixel_sha256|rgb_pixel_sha256|pixel_sha256)"
    )

    for i, line in enumerate(lines):
        if rgx.search(line):
            lo = max(0, i - 8)
            hi = min(len(lines), i + 9)
            evidence.append({
                "line_no": i + 1,
                "context_start": lo + 1,
                "context_end": hi,
                "context": "\n".join(lines[lo:hi]),
            })

    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    ap.add_argument("--r05d1-output", type=Path, default=DEFAULT_R05D1_OUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("===== Q1 R14B4B2 NEOPOLYP HISTORICAL HASH CODE LINEAGE AUDIT =====")
    print("CODE_ROOT=", args.code_root)
    print("R05D1_OUTPUT=", args.r05d1_output)
    print("IMAGE_PIXEL_ACCESS=NO")
    print("GT_PIXEL_ACCESS=NO")
    print("SUN_RGB_ACCESS=NO")
    print("CONTAMINATION_COMPARISON=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("FROZEN_COHORT_CHANGE=NO")

    candidates = find_candidate_scripts(args.code_root)
    support = find_support_files(args.r05d1_output)

    print("\n===== HISTORICAL SCRIPT CANDIDATES =====")
    print("candidate_count:", len(candidates))
    for p in candidates:
        print(p)
        print("  SHA256:", sha256_file(p))

    print("\n===== R05D1 SUPPORT/LOCK FILES =====")
    print("support_file_count:", len(support))
    for p in support:
        print(p)
        print("  SHA256:", sha256_file(p))

    audits = [extract_hash_related_context(p) for p in candidates]

    exact_column_files = []
    for a in audits:
        text, _ = safe_read_text(Path(a["path"]))
        if text and "image_rgb_pixel_sha256" in text:
            exact_column_files.append(a["path"])

    print("\n===== EXACT COLUMN TOKEN FILES =====")
    print("files_with_image_rgb_pixel_sha256:", len(exact_column_files))
    for p in exact_column_files:
        print(p)

    for a in audits:
        print("\n" + "=" * 100)
        print("FILE:", a["path"])
        print("SHA256:", a["sha256"])
        print("encoding:", a["encoding"])

        text, _ = safe_read_text(Path(a["path"]))
        if not text:
            print("UNREADABLE")
            continue

        evidence = exact_assignment_contexts(text)
        if evidence:
            print("\n--- EXACT ASSIGNMENT/USAGE CONTEXT ---")
            for e in evidence:
                print(
                    f"\n[lines {e['context_start']}-{e['context_end']}; "
                    f"hit line {e['line_no']}]"
                )
                print(e["context"])

        if a["blocks"]:
            print("\n--- COMPLETE AST BLOCKS CONTAINING PIXEL-HASH TOKEN ---")
            for b in a["blocks"]:
                print(
                    f"\n[{b['kind']} {b['name']} "
                    f"lines {b['start_line']}-{b['end_line']}]"
                )
                print(b["text"])

        relevant_matches = [
            m
            for m in a["matches"]
            if m["token"]
            in (
                "image_rgb_pixel_sha256",
                "rgb_pixel_sha256",
                "pixel_sha256",
                "tobytes",
                'convert("RGB")',
                "convert('RGB')",
                "cv2.imread",
                "Image.open",
            )
        ]

        if relevant_matches:
            print("\n--- HASH-RELATED LINE CONTEXT ---")
            shown = set()
            count = 0

            for m in relevant_matches:
                key = (m["context_start"], m["context_end"])
                if key in shown:
                    continue

                shown.add(key)
                print(
                    f"\n[token={m['token']} "
                    f"lines {m['context_start']}-{m['context_end']}]"
                )
                print(m["context"])
                count += 1

                if count >= 20:
                    print("\n[context cap reached]")
                    break

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "candidate_scripts": [str(x) for x in candidates],
        "support_files": [str(x) for x in support],
        "exact_column_files": exact_column_files,
        "audits": audits,
    }

    report_path = args.output_dir / "R14B4B2_HISTORICAL_HASH_CODE_AUDIT.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if exact_column_files:
        status = "PASS"
        decision = DECISION_FOUND
    else:
        status = "STOP"
        decision = DECISION_NOT_FOUND

    lock = {
        "status": status,
        "decision": decision,
        "code_root": str(args.code_root),
        "r05d1_output": str(args.r05d1_output),
        "candidate_script_count": len(candidates),
        "files_with_exact_column_token": exact_column_files,
        "information_boundary": {
            "image_pixels_accessed": False,
            "gt_pixels_accessed": False,
            "sun_rgb_accessed": False,
            "contamination_comparison": False,
            "model_inference": False,
            "tta_execution": False,
            "frozen_cohort_changed": False,
        },
        "next_stage": (
            "RECONSTRUCT_EXACT_PIXEL_HASH_RECIPE_FROM_HISTORICAL_CODE"
            if status == "PASS"
            else "LOCATE_ARCHIVED_R05D1_SOURCE_OR_PROVENANCE_RECORD"
        ),
    }

    lock_path = args.output_dir / "R14B4B2_HISTORICAL_HASH_CODE_LINEAGE_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nstatus=", status)
    print("Decision=", decision)
    print("REPORT=", report_path)
    print("LOCK=", lock_path)
    print("PASS" if status == "PASS" else "STOP")


if __name__ == "__main__":
    main()
