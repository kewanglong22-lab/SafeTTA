#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C2A_frozen_safety_estimator_lineage_schema_preflight_fix1.py

Purpose
-------
Pre-GT lineage/schema preflight before applying the frozen R10L0 source-safety
estimator to the canonical SUN-SEG R14C1 predictions.

This stage does NOT score SUN-SEG.

It verifies:
1) canonical R14C1 prediction lock is PASS and still pre-GT/pre-score;
2) frozen R10L0 PCA/head/threshold artifacts exist and match their lock;
3) exact frozen threshold is 0.300584763193734;
4) serialized estimator schemas can be loaded and inspected;
5) historical code implementing the CondDINO image+source-mask representation
   is located from the project source tree, rather than reconstructed from
   memory or guessed;
6) candidate historical code containing the DINOv2 base / source-mask /
   foreground-background pooling / PCA-input dtype lock is printed for the
   next scoring stage.

Forbidden:
- SUN RGB decoding
- SUN SOURCE-mask decoding
- DINO inference
- target feature extraction
- safety scoring
- GT decoding
- Dice/DeltaDice
- HARM/BENEFIT
- target tuning
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import joblib


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

R14C1_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1"
)
R14C1_LOCK = (
    R14C1_DIR
    / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
)
R14C1_GLOBAL = (
    R14C1_DIR
    / "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"
)

R10L0_DIR = (
    OUT
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
PCA_PATH = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
HEAD_PATH = R10L0_DIR / "R10L0_final_safety_head.joblib"
THRESH_PATH = R10L0_DIR / "R10L0_frozen_operating_threshold.json"
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"

EXPECTED_R14C1_DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"
)
EXPECTED_R14C1_CANONICAL_DECISION = (
    "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_CANONICALIZED_BEFORE_GT_REVEAL"
)
EXPECTED_R10L0_DECISION = (
    "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
)
EXPECTED_THRESHOLD = 0.300584763193734
EXPECTED_R14C1_ROWS = 8820
EXPECTED_R14C1_FRAMES = 980
EXPECTED_R14C1_STATES = 9

# Historical implementation evidence required before R14C2 scoring is built.
SEARCH_GROUPS = {
    "dinov2_model": (
        "facebook/dinov2-base",
        "dinov2-base",
    ),
    "source_mask_input": (
        "source_masks_packed",
        "source_mask",
    ),
    "fg_bg_semantics": (
        "foreground",
        "background",
        "fg",
        "bg",
    ),
    "pca64_or_condDINO": (
        "PCA64",
        "condDINO",
        "CondDINO",
        "pca",
    ),
}

DECISION_PASS = (
    "R10L0_FROZEN_SAFETY_ESTIMATOR_LINEAGE_AND_SCHEMA_"
    "LOCATED_READY_FOR_SUN_PREGT_SCORING"
)
DECISION_STOP = (
    "R10L0_FROZEN_SAFETY_REPRESENTATION_CODE_NOT_UNIQUELY_"
    "LOCATED_STOP_BEFORE_SCORING"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_strings(obj):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(flatten_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_strings(v))
    elif obj is not None:
        out.append(str(obj))
    return out


def find_threshold_value(obj):
    """
    Collect numeric values approximately equal to the exact frozen threshold.
    """
    hits = []

    def rec(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")
        elif isinstance(x, (int, float)):
            if abs(float(x) - EXPECTED_THRESHOLD) <= 1e-15:
                hits.append((path, float(x)))

    rec(obj)
    return hits


def artifact_reference_matches(lock_obj, artifact_path: Path, digest: str):
    """
    Locate path/name/SHA evidence inside the frozen R10L0 lock without
    assuming one particular JSON schema.
    """
    strings = flatten_strings(lock_obj)
    name_hit = any(artifact_path.name in s for s in strings)
    sha_hit = any(digest.lower() == s.strip().lower() for s in strings)
    return name_hit, sha_hit


def describe_estimator(obj):
    info = {
        "python_type": f"{type(obj).__module__}.{type(obj).__name__}",
        "n_features_in_": getattr(obj, "n_features_in_", None),
        "n_components_": getattr(obj, "n_components_", None),
        "classes_": (
            getattr(obj, "classes_", None).tolist()
            if hasattr(getattr(obj, "classes_", None), "tolist")
            else getattr(obj, "classes_", None)
        ),
        "steps": None,
    }

    if hasattr(obj, "steps"):
        info["steps"] = [
            {
                "name": name,
                "type": f"{type(step).__module__}.{type(step).__name__}",
                "n_features_in_": getattr(step, "n_features_in_", None),
                "classes_": (
                    getattr(step, "classes_", None).tolist()
                    if hasattr(getattr(step, "classes_", None), "tolist")
                    else getattr(step, "classes_", None)
                ),
            }
            for name, step in obj.steps
        ]
    return info


def scan_historical_representation_code():
    """
    Search only Python source code. We prefer scripts carrying R10K2/R10L0
    lineage markers and require DINOv2 + source-mask + representation evidence.
    """
    candidates = []

    for p in CODE.glob("*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8-sig")
        except OSError:
            continue

        low = text.lower()
        score = 0
        evidence = {}

        # Strong experiment-lineage preference.
        if "r10k2" in p.name.lower() or "r10l0" in p.name.lower():
            score += 8
        if "facebook/dinov2-base" in low:
            score += 8
        elif "dinov2-base" in low:
            score += 5

        if "source_masks_packed" in low:
            score += 5
        elif "source_mask" in low:
            score += 2

        # FG/BG image-conditioned pooling evidence.
        fg_tokens = [
            t for t in (
                "fg_mean",
                "bg_mean",
                "fg_feat",
                "bg_feat",
                "foreground",
                "background",
            )
            if t in low
        ]
        if fg_tokens:
            score += min(6, len(fg_tokens) * 2)

        if "float16" in low or "astype(np.float16)" in low:
            score += 3
        if "pca64" in low or "condDINO".lower() in low:
            score += 3
        if "224" in text and ("bicubic" in low or "interpolation" in low):
            score += 2
        if "16" in text and "22" in text:
            score += 1

        evidence["dinov2"] = [
            t for t in ("facebook/dinov2-base", "dinov2-base")
            if t in low
        ]
        evidence["source_mask"] = [
            t for t in ("source_masks_packed", "source_mask")
            if t in low
        ]
        evidence["fg_bg"] = fg_tokens
        evidence["float16"] = (
            "float16" in low or "astype(np.float16)" in low
        )
        evidence["pca64_or_conddino"] = (
            "pca64" in low or "conddino" in low
        )

        if score >= 10:
            candidates.append({
                "path": str(p),
                "sha256": sha256_file(p),
                "score": score,
                "evidence": evidence,
                "text": text,
            })

    candidates.sort(
        key=lambda x: (-x["score"], x["path"].lower())
    )
    return candidates


def ast_relevant_blocks(path: Path, text: str):
    """
    Print function/class blocks containing DINO/mask/FG/BG/PCA/float16
    evidence. This does not execute historical code.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    tokens = (
        "dinov2",
        "source_mask",
        "source_masks_packed",
        "foreground",
        "background",
        "fg_",
        "bg_",
        "float16",
        "pca",
        "224",
    )
    blocks = []

    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            continue
        if not hasattr(n, "end_lineno"):
            continue

        block = "\n".join(
            lines[n.lineno - 1:n.end_lineno]
        )
        low = block.lower()
        matched = [t for t in tokens if t in low]

        if len(matched) >= 2:
            blocks.append({
                "kind": type(n).__name__,
                "name": getattr(n, "name", ""),
                "start_line": n.lineno,
                "end_line": n.end_lineno,
                "matched_tokens": matched,
                "text": block,
            })

    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUT
            / "Q1_R14C2A_frozen_safety_estimator_lineage_schema_preflight_fix1_v1"
        ),
    )
    args = ap.parse_args()

    print(
        "===== Q1 R14C2A FROZEN SAFETY ESTIMATOR "
        "LINEAGE/SCHEMA PREFLIGHT ====="
    )
    print("SUN_RGB_DECODE=NO")
    print("SUN_SOURCE_MASK_DECODE=NO")
    print("DINO_INFERENCE=NO")
    print("TARGET_FEATURE_EXTRACTION=NO")
    print("SAFETY_SCORING=NO")
    print("GT_PIXEL_DECODE=NO")
    print("DICE_DELTADICE=NO")
    print("HARM_BENEFIT_ACCESS=NO")
    print("TARGET_TUNING=NO")

    required = (
        R14C1_LOCK,
        R14C1_GLOBAL,
        R10L0_LOCK,
        PCA_PATH,
        HEAD_PATH,
        THRESH_PATH,
    )
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    print("\n===== R14C1 CANONICAL PRE-GT GATE =====")
    r14 = load_json(R14C1_LOCK)

    if r14.get("status") != "PASS":
        raise RuntimeError("R14C1 canonical lock is not PASS.")
    if r14.get("decision") != EXPECTED_R14C1_DECISION:
        raise RuntimeError(
            f"Unexpected R14C1 decision: {r14.get('decision')}"
        )
    if (
        r14.get("canonicalization_decision")
        != EXPECTED_R14C1_CANONICAL_DECISION
    ):
        raise RuntimeError(
            "R14C1 canonicalization decision mismatch."
        )

    cohort = r14.get("cohort", {})
    if int(cohort.get("frames", -1)) != EXPECTED_R14C1_FRAMES:
        raise RuntimeError("R14C1 frame count mismatch.")

    panel = r14.get("model_panel", {})
    if int(panel.get("states", -1)) != EXPECTED_R14C1_STATES:
        raise RuntimeError("R14C1 state count mismatch.")
    if int(panel.get("model_frame_rows", -1)) != EXPECTED_R14C1_ROWS:
        raise RuntimeError("R14C1 row count mismatch.")

    info = r14.get("information_boundary", {})
    for k in (
        "sun_gt_pixels_decoded",
        "dice_computed",
        "delta_dice_computed",
        "harm_benefit_accessed",
        "frozen_safety_scores_accessed",
        "target_outcome_based_selection",
        "target_tuning",
    ):
        if bool(info.get(k, True)):
            raise RuntimeError(
                f"R14C1 pre-GT boundary not clean: {k}"
            )

    global_meta = r14.get("artifacts", {}).get(
        "global_manifest", {}
    )
    global_sha = sha256_file(R14C1_GLOBAL)
    if global_meta.get("sha256") != global_sha:
        raise RuntimeError(
            "R14C1 canonical global-manifest SHA mismatch."
        )
    if int(global_meta.get("rows", -1)) != EXPECTED_R14C1_ROWS:
        raise RuntimeError(
            "R14C1 canonical global-manifest row mismatch."
        )

    print("R14C1 canonical lock SHA256:", sha256_file(R14C1_LOCK))
    print("R14C1 global manifest SHA256:", global_sha)
    print("frames:", EXPECTED_R14C1_FRAMES)
    print("states:", EXPECTED_R14C1_STATES)
    print("rows:", EXPECTED_R14C1_ROWS)
    print("PASS")

    print("\n===== R10L0 FROZEN ESTIMATOR ARTIFACT GATE =====")
    r10 = load_json(R10L0_LOCK)

    if r10.get("decision") != EXPECTED_R10L0_DECISION:
        raise RuntimeError(
            f"Unexpected R10L0 decision: {r10.get('decision')}"
        )

    pca_sha = sha256_file(PCA_PATH)
    head_sha = sha256_file(HEAD_PATH)
    thresh_sha = sha256_file(THRESH_PATH)

    print("R10L0 lock SHA256:", sha256_file(R10L0_LOCK))
    print("PCA:", PCA_PATH)
    print("PCA SHA256:", pca_sha)
    print("HEAD:", HEAD_PATH)
    print("HEAD SHA256:", head_sha)
    print("THRESHOLD:", THRESH_PATH)
    print("THRESHOLD SHA256:", thresh_sha)

    for name, path, digest in (
        ("PCA", PCA_PATH, pca_sha),
        ("HEAD", HEAD_PATH, head_sha),
        ("THRESH", THRESH_PATH, thresh_sha),
    ):
        name_hit, sha_hit = artifact_reference_matches(
            r10, path, digest
        )
        print(
            f"{name} referenced by R10L0 lock: "
            f"name={name_hit} sha={sha_hit}"
        )
        if not name_hit:
            raise RuntimeError(
                f"R10L0 lock does not reference {name} artifact name."
            )
        if not sha_hit:
            raise RuntimeError(
                f"R10L0 lock does not contain exact {name} SHA."
            )

    threshold_obj = load_json(THRESH_PATH)
    threshold_hits = find_threshold_value(threshold_obj)
    lock_threshold_hits = find_threshold_value(r10)

    print("threshold JSON exact-value hits:", threshold_hits)
    print("R10L0 lock exact-value hits:", lock_threshold_hits)

    if not threshold_hits:
        raise RuntimeError(
            "Exact frozen threshold not found in threshold artifact."
        )
    if not lock_threshold_hits:
        raise RuntimeError(
            "Exact frozen threshold not found in R10L0 lock."
        )

    print("\n===== SERIALIZED SCHEMA INSPECTION =====")
    pca_obj = joblib.load(PCA_PATH)
    head_obj = joblib.load(HEAD_PATH)

    pca_info = describe_estimator(pca_obj)
    head_info = describe_estimator(head_obj)

    print("PCA schema:")
    print(json.dumps(pca_info, indent=2, ensure_ascii=False))
    print("HEAD schema:")
    print(json.dumps(head_info, indent=2, ensure_ascii=False))

    # Frozen representation should be 1536 DINO dimensions entering PCA64.
    pca_nin = pca_info.get("n_features_in_")
    pca_ncomp = pca_info.get("n_components_")
    if pca_nin not in (None, 1536):
        raise RuntimeError(
            f"Unexpected PCA n_features_in_={pca_nin}; expected 1536."
        )
    if pca_ncomp not in (None, 64):
        raise RuntimeError(
            f"Unexpected PCA n_components_={pca_ncomp}; expected 64."
        )

    # Final safety head should consume PCA64 + M2 = 66 dimensions.
    head_nin = head_info.get("n_features_in_")
    if head_nin not in (None, 66):
        raise RuntimeError(
            f"Unexpected safety-head n_features_in_={head_nin}; expected 66."
        )

    print("serialized schema gate: PASS")

    print("\n===== HISTORICAL CONDDINO REPRESENTATION CODE SEARCH =====")
    candidates = scan_historical_representation_code()

    print("candidate_count:", len(candidates))
    for i, c in enumerate(candidates[:12], start=1):
        print(
            f"[{i}] score={c['score']} "
            f"SHA={c['sha256']}\n    {c['path']}\n"
            f"    evidence={c['evidence']}"
        )

    # Strong candidates must explicitly include DINOv2-base and source-mask
    # semantics. We do not choose by filename alone.
    strong = []
    for c in candidates:
        low = c["text"].lower()
        if (
            ("facebook/dinov2-base" in low or "dinov2-base" in low)
            and "source_mask" in low
            and (
                "foreground" in low
                or "background" in low
                or "fg_" in low
                or "bg_" in low
            )
        ):
            strong.append(c)

    print("strong_representation_candidates:", len(strong))

    block_report = []
    for c in strong[:5]:
        blocks = ast_relevant_blocks(
            Path(c["path"]),
            c["text"],
        )
        print(
            f"\n--- RELEVANT BLOCKS: {c['path']} "
            f"({len(blocks)}) ---"
        )
        for b in blocks[:12]:
            print(
                f"\n[{b['kind']} {b['name']} "
                f"lines {b['start_line']}-{b['end_line']}]"
            )
            print(b["text"])

        block_report.append({
            "path": c["path"],
            "sha256": c["sha256"],
            "score": c["score"],
            "evidence": c["evidence"],
            "blocks": blocks,
        })

    status = "PASS" if strong else "STOP"
    decision = DECISION_PASS if strong else DECISION_STOP

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True, exist_ok=False
    )

    audit = {
        "status": status,
        "decision": decision,
        "r14c1": {
            "lock_path": str(R14C1_LOCK),
            "lock_sha256": sha256_file(R14C1_LOCK),
            "global_manifest_path": str(R14C1_GLOBAL),
            "global_manifest_sha256": global_sha,
            "frames": EXPECTED_R14C1_FRAMES,
            "states": EXPECTED_R14C1_STATES,
            "rows": EXPECTED_R14C1_ROWS,
        },
        "r10l0": {
            "lock_path": str(R10L0_LOCK),
            "lock_sha256": sha256_file(R10L0_LOCK),
            "pca_path": str(PCA_PATH),
            "pca_sha256": pca_sha,
            "head_path": str(HEAD_PATH),
            "head_sha256": head_sha,
            "threshold_path": str(THRESH_PATH),
            "threshold_sha256": thresh_sha,
            "frozen_threshold": EXPECTED_THRESHOLD,
            "pca_schema": pca_info,
            "head_schema": head_info,
        },
        "strong_representation_candidates": [
            {
                "path": c["path"],
                "sha256": c["sha256"],
                "score": c["score"],
                "evidence": c["evidence"],
            }
            for c in strong
        ],
        "block_report": block_report,
        "information_boundary": {
            "sun_rgb_decoded": False,
            "sun_source_masks_decoded": False,
            "dino_inference": False,
            "target_features_extracted": False,
            "safety_scores_computed": False,
            "gt_pixels_decoded": False,
            "dice_computed": False,
            "harm_benefit_accessed": False,
            "target_tuning": False,
        },
        "next_stage": (
            "R14C2B_BUILD_EXACT_SUN_PREGT_SCORE_LOCK_FROM_LOCATED_CODE"
            if status == "PASS"
            else "STOP_AND_RECOVER_EXACT_R10K2A_REPRESENTATION_CODE"
        ),
    }

    audit_path = (
        args.output_dir
        / "R14C2A_FROZEN_SAFETY_ESTIMATOR_LINEAGE_SCHEMA_AUDIT.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== R14C2A FINAL =====")
    print("status=", status)
    print("Decision=", decision)
    print("AUDIT=", audit_path)
    print(status)


if __name__ == "__main__":
    main()
