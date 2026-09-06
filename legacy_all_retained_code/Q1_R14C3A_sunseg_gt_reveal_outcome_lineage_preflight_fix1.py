#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C3A_sunseg_gt_reveal_outcome_lineage_preflight_fix1.py

Purpose
-------
Transition from the successfully locked pre-GT SUN-SEG safety scores to the
GT-reveal stage WITHOUT yet decoding any GT pixels.

R14C3A verifies:

1) R14C2B frozen safety-score lock is PASS and SHA-locked;
2) R14C1 canonical SOURCE/TENT1/PL prediction lock is unchanged;
3) the frozen 980-frame / 49-case SUN confirmatory manifest is unchanged;
4) the confirmatory manifest contains exactly one GT member per selected RGB;
5) exact historical outcome-construction code is located from:
   - NeoPolyp TENT1 GT reveal (R05D4);
   - NeoPolyp PL GT reveal (R13C);
   - PolypGen independent external GT reveal/evaluation (R10L3C, if present);
6) source code blocks/constants defining Dice, DeltaDice, HARM/NEUTRAL/BENEFIT
   and no-retuning semantics are printed and frozen by SHA.

This stage intentionally DOES NOT:
- open/decode SUN GT pixels;
- compute Dice;
- compute DeltaDice;
- construct HARM/NEUTRAL/BENEFIT;
- evaluate AUROC/AUPRC/FPR;
- refit/recalibrate anything.

It exists so R14C3B can reuse exact historical outcome semantics rather than
reconstructing them from memory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"
SUN_TAR = (
    ROOT / "data" / "external" / "SUN_SEG"
    / "SUN-SEG-FinalData-v20251212.tar.gz"
)

# ---------------------------------------------------------------------
# Frozen pre-GT evidence.
# ---------------------------------------------------------------------
R14C2B_DIR = (
    OUT
    / "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1_v1"
)
R14C2B_LOCK = (
    R14C2B_DIR
    / "R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json"
)
EXPECTED_R14C2B_LOCK_SHA = (
    "29030a2301178575971c8f5fab2c131d7cc06ffa99dbbd98431c288a3343337d"
)
EXPECTED_R14C2B_DECISION = (
    "SUNSEG_FROZEN_SAFETY_SCORES_LOCKED_BEFORE_GT_REVEAL"
)

R14C1_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1"
)
R14C1_LOCK = (
    R14C1_DIR
    / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
)
EXPECTED_R14C1_LOCK_SHA = (
    "802e585dc075df23e0eeddf2b7d613d229dd1f7c6ae99aeaa89cf7dd7b358163"
)

SUN_FINAL_MANIFEST = (
    OUT
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
)
EXPECTED_SUN_FINAL_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)

EXPECTED_FRAMES = 980
EXPECTED_PHYSICAL_CASES = 49
EXPECTED_STATES = 9
EXPECTED_MODEL_FRAME_ROWS = 8820

# ---------------------------------------------------------------------
# Historical GT/outcome code.
# ---------------------------------------------------------------------
R05D4_EXACT = (
    CODE
    / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1.py"
)
R13C_EXACT = (
    CODE
    / "Q1_R13C_pl_conf90_gt_utility_reveal_and_outcome_lock_fix1.py"
)

# PolypGen R10L3C exact filename was not hard-coded in prior stages.
# Locate it deterministically from experiment prefix + semantics.
R10L3C_PREFIX = "Q1_R10L3C"

DECISION_PASS = (
    "SUNSEG_GT_REVEAL_OUTCOME_SEMANTICS_LOCATED_READY_FOR_"
    "LOCKED_OUTCOME_CONSTRUCTION"
)
DECISION_STOP = (
    "SUNSEG_GT_REVEAL_OUTCOME_SEMANTICS_NOT_UNIQUELY_LOCATED_STOP"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_r10l3c_candidates():
    hits = []

    for p in CODE.glob(f"{R10L3C_PREFIX}*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="utf-8-sig")

        low = txt.lower()

        score = 0
        if "polypgen" in low:
            score += 4
        if "gt" in low:
            score += 3
        if "dice" in low:
            score += 3
        if "harm" in low:
            score += 3
        if "frozen_safety" in low or "safety_score" in low:
            score += 2
        if "threshold" in low and "tuning" in low:
            score += 1

        hits.append({
            "path": str(p),
            "sha256": sha256_file(p),
            "score": score,
            "text": txt,
        })

    hits.sort(key=lambda x: (-x["score"], x["path"].lower()))
    return hits


def relevant_blocks(path: Path, text: str):
    """
    Extract historical functions/classes that define outcome semantics.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()

    tokens = (
        "dice",
        "delta",
        "harm",
        "neutral",
        "benefit",
        "outcome",
        "threshold",
        "source",
        "a1",
        "pl_",
        "gt",
    )

    blocks = []

    for n in ast.walk(tree):
        if not isinstance(
            n,
            (ast.FunctionDef, ast.ClassDef),
        ):
            continue
        if not hasattr(n, "end_lineno"):
            continue

        block = "\n".join(
            lines[n.lineno - 1:n.end_lineno]
        )
        low = block.lower()
        matched = [t for t in tokens if t in low]

        # Require strong outcome semantics.
        if (
            ("dice" in low)
            and (
                "harm" in low
                or "benefit" in low
                or "delta" in low
                or "outcome" in low
            )
        ):
            blocks.append({
                "kind": type(n).__name__,
                "name": getattr(n, "name", ""),
                "start_line": n.lineno,
                "end_line": n.end_lineno,
                "matched_tokens": matched,
                "text": block,
            })

    return blocks


def semantic_constants(text: str):
    """
    Print source lines likely to freeze exact class/threshold semantics.
    """
    keys = (
        "HARM",
        "NEUTRAL",
        "BENEFIT",
        "DELTA",
        "DICE",
        "THRESH",
        "EPS",
        "MARGIN",
        "TOL",
    )

    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()

        if any(k in upper for k in keys):
            if (
                "=" in stripped
                or "if " in stripped
                or "elif " in stripped
                or "return " in stripped
            ):
                out.append({
                    "line": i,
                    "text": stripped,
                })

    return out


def script_audit(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    blocks = relevant_blocks(path, text)
    constants = semantic_constants(text)

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "blocks": blocks,
        "constants": constants,
        "text": text,
    }


def validate_pregt_locks():
    print("===== FROZEN PRE-GT EVIDENCE GATES =====")

    for p in (
        R14C2B_LOCK,
        R14C1_LOCK,
        SUN_FINAL_MANIFEST,
        SUN_TAR,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    c2b_sha = sha256_file(R14C2B_LOCK)
    if c2b_sha != EXPECTED_R14C2B_LOCK_SHA:
        raise RuntimeError(
            f"R14C2B lock SHA changed: {c2b_sha}"
        )

    c2b = load_json(R14C2B_LOCK)
    if c2b.get("status") != "FROZEN":
        raise RuntimeError("R14C2B status is not FROZEN.")
    if c2b.get("decision") != EXPECTED_R14C2B_DECISION:
        raise RuntimeError("R14C2B decision changed.")

    info = c2b.get("information_boundary", {})
    if bool(info.get("sun_gt_pixels_decoded", True)):
        raise RuntimeError(
            "R14C2B does not preserve pre-GT boundary."
        )
    if bool(info.get("sun_outcomes_revealed", True)):
        raise RuntimeError(
            "R14C2B indicates outcomes already revealed."
        )
    if bool(info.get("target_calibration", True)):
        raise RuntimeError(
            "R14C2B target calibration unexpectedly true."
        )
    if bool(info.get("target_threshold_selection", True)):
        raise RuntimeError(
            "R14C2B target threshold selection unexpectedly true."
        )

    if sha256_file(R14C1_LOCK) != EXPECTED_R14C1_LOCK_SHA:
        raise RuntimeError("R14C1 canonical lock SHA changed.")

    manifest_sha = sha256_file(SUN_FINAL_MANIFEST)
    if manifest_sha != EXPECTED_SUN_FINAL_MANIFEST_SHA:
        raise RuntimeError(
            f"SUN final manifest SHA changed: {manifest_sha}"
        )

    print("R14C2B lock SHA:", c2b_sha)
    print("R14C1 lock SHA:", EXPECTED_R14C1_LOCK_SHA)
    print("SUN final manifest SHA:", manifest_sha)
    print("R14C2B pre-GT boundary: PASS")
    print("PASS")

    return c2b


def validate_sun_gt_schema():
    print("\n===== SUN CONFIRMATORY GT SCHEMA GATE =====")

    fields, rows = read_csv(SUN_FINAL_MANIFEST)

    required = {
        "pair_key",
        "external_case_id",
        "cluster_id",
        "frame_member",
        "gt_member",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(
            f"SUN final manifest missing GT schema columns: {missing}"
        )

    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"SUN manifest rows={len(rows)} expected={EXPECTED_FRAMES}"
        )

    if len({r["pair_key"] for r in rows}) != EXPECTED_FRAMES:
        raise RuntimeError("SUN pair_key not unique.")
    if len({r["frame_member"] for r in rows}) != EXPECTED_FRAMES:
        raise RuntimeError("SUN frame_member not unique.")
    if len({r["gt_member"] for r in rows}) != EXPECTED_FRAMES:
        raise RuntimeError("SUN gt_member not unique.")
    if len({r["cluster_id"] for r in rows}) != EXPECTED_PHYSICAL_CASES:
        raise RuntimeError("SUN physical-case count changed.")

    if any(not str(r["gt_member"]).strip() for r in rows):
        raise RuntimeError("Empty SUN gt_member found.")

    # GT member names are allowed to be inspected only now, AFTER R14C2B lock.
    # Pixel bytes are NOT extracted.
    target_gt = {str(r["gt_member"]).replace("\\", "/") for r in rows}

    print("GT member names required:", len(target_gt))
    print("GT pixel decode: NO")
    print("GT member byte extraction: NO")

    # Verify names exist in official raw TAR by one sequential metadata pass.
    found = set()
    with tarfile.open(SUN_TAR, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            if name in target_gt:
                found.add(name)
                if len(found) == len(target_gt):
                    break

    if found != target_gt:
        missing_gt = sorted(target_gt - found)
        raise RuntimeError(
            f"SUN GT TAR members missing={len(missing_gt)} "
            f"examples={missing_gt[:10]}"
        )

    print("GT member names found in official TAR:", len(found))
    print("GT pixel decode: NO")
    print("PASS")

    return rows


def print_script_report(label, audit):
    print(f"\n===== {label} =====")
    print("path:", audit["path"])
    print("SHA256:", audit["sha256"])
    print("outcome-relevant blocks:", len(audit["blocks"]))
    print("semantic constant/branch lines:", len(audit["constants"]))

    for row in audit["constants"][:80]:
        print(
            f"CONST/BRANCH L{row['line']}: "
            f"{row['text']}"
        )

    for b in audit["blocks"][:20]:
        print(
            f"\n[{b['kind']} {b['name']} "
            f"lines {b['start_line']}-{b['end_line']}]"
        )
        print(b["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUT
            / "Q1_R14C3A_sunseg_gt_reveal_outcome_lineage_preflight_fix1_v1"
        ),
    )
    args = ap.parse_args()

    print(
        "===== Q1 R14C3A SUN-SEG GT-REVEAL OUTCOME "
        "LINEAGE PREFLIGHT ====="
    )
    print("R14C2B_SCORE_LOCK_REQUIRED=YES")
    print("GT_MEMBER_NAME_ACCESS=YES")
    print("GT_PIXEL_DECODE=NO")
    print("DICE_COMPUTATION=NO")
    print("DELTA_DICE_COMPUTATION=NO")
    print("HARM_NEUTRAL_BENEFIT_CONSTRUCTION=NO")
    print("SAFETY_SCORE_RECALIBRATION=NO")
    print("TARGET_THRESHOLD_RETUNING=NO")
    print("MODEL_TTA_RERUN=NO")

    c2b = validate_pregt_locks()
    sun_rows = validate_sun_gt_schema()

    # Exact NeoPolyp historical outcome implementations.
    r05 = script_audit(R05D4_EXACT)
    r13 = script_audit(R13C_EXACT)

    print_script_report(
        "HISTORICAL TENT1 GT/OUTCOME IMPLEMENTATION R05D4",
        r05,
    )
    print_script_report(
        "HISTORICAL PL GT/OUTCOME IMPLEMENTATION R13C",
        r13,
    )

    # Locate historical independent external-cohort GT reveal implementation.
    print("\n===== HISTORICAL POLYPGEN R10L3C SEARCH =====")
    l3c_candidates = find_r10l3c_candidates()
    print("R10L3C candidates:", len(l3c_candidates))

    for i, c in enumerate(l3c_candidates, start=1):
        print(
            f"[{i}] score={c['score']} "
            f"SHA={c['sha256']}\n    {c['path']}"
        )

    # Require a unique top semantic candidate.
    l3c = None
    if l3c_candidates:
        top_score = l3c_candidates[0]["score"]
        top = [
            c for c in l3c_candidates
            if c["score"] == top_score
        ]
        if len(top) == 1 and top_score >= 8:
            p = Path(top[0]["path"])
            l3c = script_audit(p)
            print_script_report(
                "HISTORICAL POLYPGEN GT/OUTCOME IMPLEMENTATION R10L3C",
                l3c,
            )

    # Hard outcome-semantic evidence requirement.
    if not r05["blocks"]:
        raise RuntimeError(
            "R05D4 has no outcome-relevant Dice block."
        )
    if not r13["blocks"]:
        raise RuntimeError(
            "R13C has no outcome-relevant Dice block."
        )

    # We can proceed even if no unique R10L3C exists only if both exact
    # action-specific NeoPolyp implementations are strongly located.
    status = "PASS"
    decision = DECISION_PASS

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    audit = {
        "status": status,
        "decision": decision,
        "r14c2b": {
            "lock_path": str(R14C2B_LOCK),
            "lock_sha256": EXPECTED_R14C2B_LOCK_SHA,
            "decision": EXPECTED_R14C2B_DECISION,
            "frozen_probability_rows": int(
                c2b.get("estimator", {}).get(
                    "probability_rows",
                    -1,
                )
            ),
            "frozen_threshold": float(
                c2b.get("estimator", {}).get(
                    "frozen_threshold"
                )
            ),
        },
        "sun_gt_schema": {
            "manifest_path": str(SUN_FINAL_MANIFEST),
            "manifest_sha256": EXPECTED_SUN_FINAL_MANIFEST_SHA,
            "frames": len(sun_rows),
            "physical_cases": len(
                {r["cluster_id"] for r in sun_rows}
            ),
            "gt_members": len(
                {r["gt_member"] for r in sun_rows}
            ),
            "gt_member_names_verified_in_tar": True,
            "gt_pixel_decode": False,
        },
        "historical_tent1_outcome_code": {
            "path": r05["path"],
            "sha256": r05["sha256"],
            "blocks": r05["blocks"],
            "constants": r05["constants"],
        },
        "historical_pl_outcome_code": {
            "path": r13["path"],
            "sha256": r13["sha256"],
            "blocks": r13["blocks"],
            "constants": r13["constants"],
        },
        "historical_polypgen_r10l3c": (
            None
            if l3c is None
            else {
                "path": l3c["path"],
                "sha256": l3c["sha256"],
                "blocks": l3c["blocks"],
                "constants": l3c["constants"],
            }
        ),
        "information_boundary": {
            "r14c2b_score_lock_completed": True,
            "gt_member_names_accessed": True,
            "gt_pixel_bytes_extracted": False,
            "gt_pixels_decoded": False,
            "dice_computed": False,
            "delta_dice_computed": False,
            "outcomes_constructed": False,
            "safety_scores_recomputed": False,
            "safety_scores_recalibrated": False,
            "threshold_retuned": False,
            "tta_rerun": False,
        },
        "next_stage": (
            "R14C3B_SUNSEG_GT_REVEAL_DUAL_ACTION_OUTCOME_LOCK_"
            "THEN_FROZEN_SCORE_EVALUATION"
        ),
    }

    audit_path = (
        args.output_dir
        / "R14C3A_SUNSEG_GT_REVEAL_OUTCOME_LINEAGE_AUDIT.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== R14C3A FINAL =====")
    print("status=", status)
    print("Decision=", decision)
    print("AUDIT=", audit_path)
    print("GT pixels decoded=NO")
    print("Dice computed=NO")
    print("PASS")


if __name__ == "__main__":
    main()
