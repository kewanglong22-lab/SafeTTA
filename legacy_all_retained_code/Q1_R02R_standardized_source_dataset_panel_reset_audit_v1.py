#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R02R — Standardized Source Dataset & Panel Reset Audit

Filesystem / filename audit only:
- NO image decoding
- NO checkpoint loading
- NO training
- NO inference
- NO random source split

Goal:
Find clean original non-target polyp segmentation image/mask datasets that
can support a newly standardized 3×3 model panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R02R-v1"
BUILD = "Q1_R02R_STANDARDIZED_SOURCE_DATASET_PANEL_RESET_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "Q1_R02R_standardized_source_dataset_panel_reset_audit_v1"
)

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_R02R_standardized_source_dataset_panel_reset_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "2868adef05434a30cf8ccd8d1fc681deb4c559f42df282db00cb95a68b9d717d"
)

FIX4_DIR = (
    ROOT
    / "outputs"
    / "Q1_R02_original_source_provenance_audit_fix4"
)
FIX4_LOCK = (
    FIX4_DIR
    / "Q1_R02_ORIGINAL_SOURCE_PROVENANCE_AUDIT_FIX4_LOCK.json"
)
EXPECTED_FIX4_LOCK_SHA256 = (
    "1c35fc011cc88d0b1c0d0d589c78d9253c62e1baf99c6a990ab8282f64dad3f0"
)
EXPECTED_FIX4_DECISION = (
    "ORIGINAL_SOURCE_TRAINING_PROVENANCE_UNRESOLVED"
)
FIX4_S00_ROLE = FIX4_DIR / "s00_asset_role_audit.json"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

IMAGE_DIR_NAMES = {
    "image", "images", "img", "imgs",
    "original", "originals",
}
MASK_DIR_NAMES = {
    "mask", "masks", "gt", "gts",
    "label", "labels",
    "ground_truth", "groundtruth",
}

MASK_SUFFIXES = (
    "_mask", "-mask", "_masks",
    "_gt", "-gt",
    "_label", "-label",
    "_seg", "-seg",
    "_annotation", "-annotation",
)

DATASET_TOKENS = {
    "Kvasir-SEG": (
        "kvasir-seg", "kvasir_seg", "kvasirseg", "kvasir"
    ),
    "CVC-ClinicDB": (
        "cvc-clinicdb", "cvc_clinicdb", "clinicdb"
    ),
    "CVC-ColonDB": (
        "cvc-colondb", "cvc_colondb", "colondb", "colon-db"
    ),
    "CVC-300": (
        "cvc-300", "cvc_300", "cvc300"
    ),
    "ETIS": (
        "etis-larib", "etis_larib", "etis"
    ),
    "PolypGen": (
        "polypgen",
    ),
}

UTILITY_TOKENS = (
    "s00_polyp_locked_v1",
    "q1_s00",
    "q1-s00",
    "s07_b",
    "s07-b",
    "counterfactual_utility",
    "gradient_probe",
    "source_side_counterfactual",
)

TARGET_TOKENS = (
    "polypgen",
    "target",
    "external_target",
    "untouched_external",
)

GENERATED_TOKENS = (
    "prediction",
    "predictions",
    "logit",
    "logits",
    "probability",
    "probabilities",
    "pseudo",
    "adapted",
    "inference",
    "oof",
)

SKIP_DIR_NAMES = {
    ".git",
    ".conda",
    "env",
    "venv",
    "__pycache__",
    "outputs",
    "code",
    "docs",
    "configs",
    "checkpoints",
    "models",
    "weights",
    "cache",
}

MIN_SOURCE_PAIRS = 100

DECISION_MISSING = "SOURCE_DATASET_ASSET_MISSING"
DECISION_PAIRING = "SOURCE_DATASET_PAIRING_INVALID"
DECISION_MULTIPLE = "MULTIPLE_SOURCE_CANDIDATES_REQUIRE_FREEZE"
DECISION_READY = "STANDARDIZED_SOURCE_DATASET_READY"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[dict],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
        )
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path, root: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(root.resolve())
        )
    except Exception:
        return str(path)


def normalize_stem(stem: str) -> str:
    s = stem.lower().strip()
    changed = True

    while changed:
        changed = False
        for suffix in MASK_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                changed = True

    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def detect_dataset_family(path: Path) -> str:
    low = str(path).replace("\\", "/").lower()

    for family, tokens in DATASET_TOKENS.items():
        if any(token in low for token in tokens):
            return family

    return "Unknown"


def path_has_any(path: Path, tokens: Sequence[str]) -> bool:
    low = str(path).replace("\\", "/").lower()
    return any(token in low for token in tokens)


def validate_upstream():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)

    protocol_sha = file_sha256(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Q1-R02R protocol SHA mismatch: "
            f"expected={EXPECTED_PROTOCOL_SHA256} "
            f"actual={protocol_sha}"
        )

    if not FIX4_LOCK.exists():
        raise FileNotFoundError(FIX4_LOCK)

    fix4_sha = file_sha256(FIX4_LOCK)
    if fix4_sha != EXPECTED_FIX4_LOCK_SHA256:
        raise RuntimeError(
            "Q1-R02 FIX4 lock SHA mismatch: "
            f"expected={EXPECTED_FIX4_LOCK_SHA256} "
            f"actual={fix4_sha}"
        )

    fix4_lock = json.loads(
        FIX4_LOCK.read_text(encoding="utf-8")
    )
    if fix4_lock.get("decision") != EXPECTED_FIX4_DECISION:
        raise RuntimeError(
            "Unexpected Q1-R02 FIX4 decision: "
            f"{fix4_lock.get('decision')}"
        )

    if not FIX4_S00_ROLE.exists():
        raise FileNotFoundError(FIX4_S00_ROLE)

    s00 = json.loads(
        FIX4_S00_ROLE.read_text(encoding="utf-8")
    )

    if s00.get("matches_145x10_cardinality") is not True:
        raise RuntimeError(
            "Frozen S00 145x10 role finding no longer reproduces."
        )

    return {
        "protocol_sha256": protocol_sha,
        "fix4_lock_sha256": fix4_sha,
        "fix4_decision": fix4_lock["decision"],
        "s00_matches_145x10": True,
        "s00_physical_image_count": s00.get(
            "physical_image_count"
        ),
        "s00_source_group_unique": s00.get(
            "q1_s00a_source_group_unique"
        ),
        "s00_perturbation_unique": s00.get(
            "q1_s00a_perturbation_unique"
        ),
    }


def iter_project_dirs(root: Path):
    for current, dirs, files in os.walk(
        root,
        topdown=True,
    ):
        cp = Path(current)

        dirs[:] = sorted(
            d
            for d in dirs
            if d.lower() not in SKIP_DIR_NAMES
            and not (cp / d).is_symlink()
        )

        yield cp, list(dirs), list(files)


def count_image_files(directory: Path) -> int:
    count = 0
    for p in directory.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            count += 1
    return count


def list_image_files(directory: Path) -> List[Path]:
    files = [
        p.resolve()
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]
    files.sort(key=lambda p: str(p).lower())
    return files


def discover_named_dataset_dirs(root: Path):
    rows = []

    for cp, dirs, files in tqdm(
        list(iter_project_dirs(root)),
        desc="Q1-R02R discovering dataset directories",
        unit="dir",
        dynamic_ncols=True,
    ):
        low = str(cp).replace("\\", "/").lower()
        family = detect_dataset_family(cp)

        image_children = []
        mask_children = []

        for d in dirs:
            p = cp / d
            n = d.lower().replace("-", "_")

            if n in IMAGE_DIR_NAMES:
                image_children.append(p.resolve())
            if n in MASK_DIR_NAMES:
                mask_children.append(p.resolve())

        if (
            family != "Unknown"
            or image_children
            or mask_children
            or any(
                tok in low
                for tok in (
                    "dataset", "data", "polyp",
                    "kvasir", "clinic", "colon", "etis"
                )
            )
        ):
            rows.append({
                "relative_dir": rel(cp, root),
                "dataset_family": family,
                "direct_subdir_count": len(dirs),
                "direct_file_count": len(files),
                "image_child_dirs": ";".join(
                    str(p) for p in image_children
                ),
                "mask_child_dirs": ";".join(
                    str(p) for p in mask_children
                ),
                "utility_token": path_has_any(
                    cp,
                    UTILITY_TOKENS,
                ),
                "target_token": path_has_any(
                    cp,
                    TARGET_TOKENS,
                ),
                "generated_token": path_has_any(
                    cp,
                    GENERATED_TOKENS,
                ),
            })

    rows.sort(
        key=lambda r: (
            r["dataset_family"],
            r["relative_dir"].lower(),
        )
    )
    return rows


def discover_pair_candidates(root: Path):
    pairs = []
    seen = set()

    for cp, dirs, _ in iter_project_dirs(root):
        children = {
            d.lower().replace("-", "_"): (
                cp / d
            ).resolve()
            for d in dirs
        }

        images = []
        masks = []

        for name, p in children.items():
            if name in IMAGE_DIR_NAMES:
                images.append(p)
            if name in MASK_DIR_NAMES:
                masks.append(p)

        for image_dir in images:
            for mask_dir in masks:
                key = (
                    str(image_dir).lower(),
                    str(mask_dir).lower(),
                )
                if key in seen:
                    continue
                seen.add(key)

                pairs.append({
                    "candidate_root": cp.resolve(),
                    "image_dir": image_dir,
                    "mask_dir": mask_dir,
                    "discovery_mode": "SIBLING_IMAGE_MASK_DIRS",
                })

    # Also identify common nested split layouts:
    # dataset/train/images + dataset/train/masks
    # already captured because train is walked as cp.

    pairs.sort(
        key=lambda r: (
            str(r["candidate_root"]).lower(),
            str(r["image_dir"]).lower(),
            str(r["mask_dir"]).lower(),
        )
    )
    return pairs


def audit_pair(image_dir: Path, mask_dir: Path):
    images = list_image_files(image_dir)
    masks = list_image_files(mask_dir)

    exact_mask_map = defaultdict(list)
    norm_mask_map = defaultdict(list)

    for m in masks:
        exact_mask_map[m.stem.lower()].append(m)
        norm_mask_map[normalize_stem(m.stem)].append(m)

    duplicate_norm = {
        k: [str(p) for p in v]
        for k, v in norm_mask_map.items()
        if len(v) > 1
    }

    matched = []
    unmatched_images = []
    used_masks = set()

    for img in images:
        exact = exact_mask_map.get(
            img.stem.lower(),
            [],
        )

        chosen = None

        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            unmatched_images.append(str(img))
            continue
        else:
            norm = norm_mask_map.get(
                normalize_stem(img.stem),
                [],
            )
            if len(norm) == 1:
                chosen = norm[0]
            else:
                unmatched_images.append(str(img))
                continue

        if chosen in used_masks:
            unmatched_images.append(str(img))
            continue

        used_masks.add(chosen)
        matched.append((img, chosen))

    unmatched_masks = [
        str(m)
        for m in masks
        if m not in used_masks
    ]

    image_count = len(images)
    mask_count = len(masks)
    matched_count = len(matched)

    pairing_rate = (
        matched_count / image_count
        if image_count
        else 0.0
    )

    return {
        "image_count": image_count,
        "mask_count": mask_count,
        "matched_count": matched_count,
        "pairing_rate": pairing_rate,
        "unmatched_image_count": len(
            unmatched_images
        ),
        "unmatched_mask_count": len(
            unmatched_masks
        ),
        "duplicate_normalized_mask_keys": len(
            duplicate_norm
        ),
        "unmatched_image_examples": unmatched_images[:10],
        "unmatched_mask_examples": unmatched_masks[:10],
        "duplicate_mask_examples": dict(
            list(duplicate_norm.items())[:10]
        ),
    }


def classify_candidate(
    candidate_root: Path,
    image_dir: Path,
    mask_dir: Path,
    audit: dict,
):
    all_paths = (
        str(candidate_root)
        + " "
        + str(image_dir)
        + " "
        + str(mask_dir)
    )
    pseudo_path = Path(all_paths)

    family = detect_dataset_family(
        candidate_root
    )
    if family == "Unknown":
        family = detect_dataset_family(
            image_dir
        )

    utility = path_has_any(
        pseudo_path,
        UTILITY_TOKENS,
    )
    target = path_has_any(
        pseudo_path,
        TARGET_TOKENS,
    )
    generated = path_has_any(
        pseudo_path,
        GENERATED_TOKENS,
    )

    perfect_pairing = (
        audit["image_count"] > 0
        and audit["matched_count"]
        == audit["image_count"]
        and audit["unmatched_image_count"] == 0
        and audit["unmatched_mask_count"] == 0
        and audit["duplicate_normalized_mask_keys"] == 0
    )

    if utility:
        classification = "EXCLUDED_Q1_UTILITY_ASSET"
        reason = "S00/Q1/S07 utility experiment asset."

    elif family == "PolypGen" or target:
        classification = "EXCLUDED_TARGET_OR_EVALUATION"
        reason = (
            "PolypGen/target-associated asset cannot enter "
            "source training."
        )

    elif generated:
        classification = "EXCLUDED_GENERATED_OUTPUT"
        reason = (
            "Path suggests predictions/logits/pseudo/adapted outputs."
        )

    elif not perfect_pairing:
        classification = "PAIRING_INCOMPLETE"
        reason = "Image/mask pairing is not 100% clean."

    elif audit["matched_count"] < MIN_SOURCE_PAIRS:
        classification = "TOO_SMALL_FOR_PRIMARY_SOURCE"
        reason = (
            f"Only {audit['matched_count']} paired cases; "
            f"minimum is {MIN_SOURCE_PAIRS}."
        )

    else:
        classification = "ELIGIBLE_CLEAN_SOURCE_CANDIDATE"
        reason = (
            "Original-looking image/GT pair structure, complete pairing, "
            "not target/utility/generated."
        )

    return family, classification, reason


def run(args):
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    try:
        output_dir.relative_to(root)
    except Exception:
        raise RuntimeError(
            f"Output must remain under project root: {output_dir}"
        )

    if str(output_dir).lower().startswith("c:\\"):
        raise RuntimeError("C-drive output is forbidden.")

    if output_dir.exists():
        raise FileExistsError(
            f"Q1-R02R output already exists: {output_dir}"
        )

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-R02R output exists: {build_dir}"
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print(
        "===== Q1-R02R STANDARDIZED SOURCE DATASET & PANEL RESET AUDIT ====="
    )
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Root: {root}")
    print("Image content opening=NO")
    print("Checkpoint loading=NO")
    print("Training=NO")
    print("Inference=NO")
    print("Random source split=NO")
    print()

    provenance = validate_upstream()

    discovered = discover_named_dataset_dirs(
        root
    )
    discovered_path = (
        build_dir
        / "discovered_dataset_directories.csv"
    )
    write_csv(
        discovered_path,
        discovered,
        [
            "relative_dir",
            "dataset_family",
            "direct_subdir_count",
            "direct_file_count",
            "image_child_dirs",
            "mask_child_dirs",
            "utility_token",
            "target_token",
            "generated_token",
        ],
    )

    candidates = discover_pair_candidates(
        root
    )

    candidate_rows = []
    pairing_rows = []
    classification_rows = []
    excluded_rows = []

    for idx, c in enumerate(
        tqdm(
            candidates,
            desc="Q1-R02R auditing image-mask candidates",
            unit="candidate",
            dynamic_ncols=True,
        ),
        1,
    ):
        audit = audit_pair(
            c["image_dir"],
            c["mask_dir"],
        )

        family, classification, reason = (
            classify_candidate(
                c["candidate_root"],
                c["image_dir"],
                c["mask_dir"],
                audit,
            )
        )

        candidate_id = f"C{idx:03d}"

        candidate_rows.append({
            "candidate_id": candidate_id,
            "candidate_root": str(
                c["candidate_root"]
            ),
            "image_dir": str(c["image_dir"]),
            "mask_dir": str(c["mask_dir"]),
            "dataset_family": family,
            "discovery_mode": c[
                "discovery_mode"
            ],
        })

        pairing_rows.append({
            "candidate_id": candidate_id,
            "dataset_family": family,
            "image_count": audit[
                "image_count"
            ],
            "mask_count": audit[
                "mask_count"
            ],
            "matched_count": audit[
                "matched_count"
            ],
            "pairing_rate": audit[
                "pairing_rate"
            ],
            "unmatched_image_count": audit[
                "unmatched_image_count"
            ],
            "unmatched_mask_count": audit[
                "unmatched_mask_count"
            ],
            "duplicate_normalized_mask_keys": audit[
                "duplicate_normalized_mask_keys"
            ],
            "unmatched_image_examples": ";".join(
                audit[
                    "unmatched_image_examples"
                ]
            ),
            "unmatched_mask_examples": ";".join(
                audit[
                    "unmatched_mask_examples"
                ]
            ),
        })

        row = {
            "candidate_id": candidate_id,
            "dataset_family": family,
            "candidate_root": str(
                c["candidate_root"]
            ),
            "image_dir": str(c["image_dir"]),
            "mask_dir": str(c["mask_dir"]),
            "matched_count": audit[
                "matched_count"
            ],
            "classification": classification,
            "reason": reason,
        }
        classification_rows.append(row)

        if classification != (
            "ELIGIBLE_CLEAN_SOURCE_CANDIDATE"
        ):
            excluded_rows.append(row)

    candidates_path = (
        build_dir
        / "image_mask_pair_candidates.csv"
    )
    write_csv(
        candidates_path,
        candidate_rows,
        [
            "candidate_id",
            "candidate_root",
            "image_dir",
            "mask_dir",
            "dataset_family",
            "discovery_mode",
        ],
    )

    pairing_path = build_dir / "pairing_audit.csv"
    write_csv(
        pairing_path,
        pairing_rows,
        [
            "candidate_id",
            "dataset_family",
            "image_count",
            "mask_count",
            "matched_count",
            "pairing_rate",
            "unmatched_image_count",
            "unmatched_mask_count",
            "duplicate_normalized_mask_keys",
            "unmatched_image_examples",
            "unmatched_mask_examples",
        ],
    )

    classification_path = (
        build_dir
        / "dataset_role_classification.csv"
    )
    write_csv(
        classification_path,
        classification_rows,
        [
            "candidate_id",
            "dataset_family",
            "candidate_root",
            "image_dir",
            "mask_dir",
            "matched_count",
            "classification",
            "reason",
        ],
    )

    excluded_path = (
        build_dir
        / "excluded_asset_audit.csv"
    )
    write_csv(
        excluded_path,
        excluded_rows,
        [
            "candidate_id",
            "dataset_family",
            "candidate_root",
            "image_dir",
            "mask_dir",
            "matched_count",
            "classification",
            "reason",
        ],
    )

    eligible = [
        r
        for r in classification_rows
        if r["classification"]
        == "ELIGIBLE_CLEAN_SOURCE_CANDIDATE"
    ]

    pairing_invalid = [
        r
        for r in classification_rows
        if r["classification"]
        == "PAIRING_INCOMPLETE"
    ]

    if not candidates:
        decision = DECISION_MISSING
    elif not eligible:
        if pairing_invalid:
            decision = DECISION_PAIRING
        else:
            decision = DECISION_MISSING
    elif len(eligible) > 1:
        decision = DECISION_MULTIPLE
    else:
        decision = DECISION_READY

    source_summary = {
        "candidate_pair_directories": len(
            candidates
        ),
        "eligible_clean_source_candidates": len(
            eligible
        ),
        "eligible_candidates": eligible,
        "pairing_invalid_candidates": len(
            pairing_invalid
        ),
        "minimum_source_pairs": MIN_SOURCE_PAIRS,
        "decision": decision,
    }
    source_summary_path = (
        build_dir
        / "source_candidate_summary.json"
    )
    write_json(
        source_summary_path,
        source_summary,
    )

    reset_panel = {
        "legacy_main_panel_retired": True,
        "legacy_checkpoint_use": [
            "historical mechanism analysis",
            "supplementary comparison",
        ],
        "new_main_panel": {
            "PraNet": 3,
            "DeepLabV3-R50": 3,
            "SegFormer-B0": 3,
            "total_states": 9,
        },
        "common_training_data_required": True,
        "common_validation_membership_required": True,
        "common_checkpoint_selection_rule_required": True,
        "target_data_in_training": False,
        "random_split_created_in_q1_r02r": False,
    }
    reset_path = (
        build_dir
        / "reset_panel_plan.json"
    )
    write_json(
        reset_path,
        reset_panel,
    )

    provenance.update({
        "root": str(root),
        "dataset_dirs_reported": len(
            discovered
        ),
        "pair_candidates": len(
            candidates
        ),
        "eligible_candidates": len(
            eligible
        ),
        "image_content_opened": False,
        "checkpoint_loaded": False,
        "training": False,
        "inference": False,
        "random_source_split": False,
        "decision": decision,
    })
    provenance_path = (
        build_dir
        / "provenance_audit.json"
    )
    write_json(
        provenance_path,
        provenance,
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(
        decision + "\n",
        encoding="utf-8",
    )

    log = f"""===== Q1-R02R STANDARDIZED SOURCE DATASET & PANEL RESET AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Upstream:
  Q1-R02 FIX4 lock verified=YES
  FIX4 decision={provenance['fix4_decision']}
  S00 exact 145x10 utility cardinality=YES

Reset-panel principle:
  legacy PraNet/DeepLab main panel=RETIRED
  new main panel=3 PraNet + 3 DeepLabV3-R50 + 3 SegFormer-B0
  common source train/val protocol required=YES

Asset discovery:
  dataset-like directories={len(discovered)}
  image-mask candidate pairs={len(candidates)}
  eligible clean source candidates={len(eligible)}
  pairing-invalid candidates={len(pairing_invalid)}

Eligible candidates:
"""
    if eligible:
        for r in eligible:
            log += (
                f"  - {r['candidate_id']} "
                f"family={r['dataset_family']} "
                f"pairs={r['matched_count']} "
                f"root={r['candidate_root']}\n"
            )
    else:
        log += "  - NONE\n"

    log += f"""
Image content opening=NO
Checkpoint loading=NO
Training=NO
Inference=NO
Random source split=NO

Decision:
  {decision}

Decision mapping:
  {DECISION_READY}
    -> Q1-R02S Standardized Source Split Freeze

  {DECISION_MULTIPLE}
    -> Q1-R02S Multi-Dataset Source Composition & Split Freeze

  {DECISION_MISSING}
    -> restore/download a clean public source segmentation dataset

  {DECISION_PAIRING}
    -> resolve original image/mask pairing before source split

[OK] Outputs:
  {output_dir}
"""

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        log,
        encoding="utf-8",
    )

    artifact_paths = {
        "protocol_copy": (
            build_dir
            / "preregistered_protocol_copy.md"
        ),
        "discovered_dataset_directories": discovered_path,
        "image_mask_pair_candidates": candidates_path,
        "pairing_audit": pairing_path,
        "dataset_role_classification": classification_path,
        "excluded_asset_audit": excluded_path,
        "source_candidate_summary": source_summary_path,
        "reset_panel_plan": reset_path,
        "provenance_audit": provenance_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    shutil.copy2(
        PROTOCOL,
        artifact_paths["protocol_copy"],
    )

    # Strong negative-classification assertions.
    for r in classification_rows:
        low = (
            r["candidate_root"]
            + " "
            + r["image_dir"]
            + " "
            + r["mask_dir"]
        ).lower()

        if (
            r["classification"]
            == "ELIGIBLE_CLEAN_SOURCE_CANDIDATE"
        ):
            if "polypgen" in low:
                raise RuntimeError(
                    "Integrity failure: PolypGen was marked source-eligible."
                )
            if any(
                token in low
                for token in UTILITY_TOKENS
            ):
                raise RuntimeError(
                    "Integrity failure: Q1/S00 utility asset was marked "
                    "source-eligible."
                )

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "fix4_lock_sha256": EXPECTED_FIX4_LOCK_SHA256,
        "fix4_decision": EXPECTED_FIX4_DECISION,
        "s00_145x10_preserved": True,
        "image_content_opened": False,
        "checkpoint_loaded": False,
        "training": False,
        "inference": False,
        "random_source_split": False,
        "eligible_source_candidates": len(
            eligible
        ),
        "decision": decision,
        "artifacts": {
            name: {
                "filename": p.name,
                "sha256": file_sha256(p),
            }
            for name, p in artifact_paths.items()
        },
    }

    lock_path = (
        build_dir
        / "Q1_R02R_STANDARDIZED_SOURCE_AUDIT_LOCK.json"
    )
    write_json(
        lock_path,
        lock,
    )
    lock_sha = file_sha256(lock_path)

    for name, p in artifact_paths.items():
        if (
            file_sha256(p)
            != lock["artifacts"][name]["sha256"]
        ):
            raise RuntimeError(
                f"Artifact changed before commit: {name}"
            )

    build_dir.rename(output_dir)

    print()
    print(
        (output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R02R LOCK:",
        output_dir
        / "Q1_R02R_STANDARDIZED_SOURCE_AUDIT_LOCK.json",
    )
    print("Q1-R02R LOCK SHA256:", lock_sha)


def self_test():
    assert normalize_stem("case001_mask") == "case001"
    assert normalize_stem("case001-GT") == "case001"

    assert (
        detect_dataset_family(
            Path(r"F:\x\Kvasir-SEG\images")
        )
        == "Kvasir-SEG"
    )
    assert (
        detect_dataset_family(
            Path(r"F:\x\CVC-ClinicDB\images")
        )
        == "CVC-ClinicDB"
    )
    assert (
        detect_dataset_family(
            Path(r"F:\x\PolypGen\images")
        )
        == "PolypGen"
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds = root / "Kvasir-SEG"
        img = ds / "images"
        msk = ds / "masks"
        img.mkdir(parents=True)
        msk.mkdir(parents=True)

        for i in range(120):
            (img / f"case{i:03d}.png").write_bytes(b"x")
            (msk / f"case{i:03d}_mask.png").write_bytes(b"y")

        audit = audit_pair(img, msk)
        assert audit["matched_count"] == 120
        assert audit["unmatched_image_count"] == 0
        assert audit["unmatched_mask_count"] == 0

        family, classification, _ = classify_candidate(
            ds,
            img,
            msk,
            audit,
        )
        assert family == "Kvasir-SEG"
        assert (
            classification
            == "ELIGIBLE_CLEAN_SOURCE_CANDIDATE"
        )

        target = root / "PolypGen"
        ti = target / "images"
        tm = target / "masks"
        ti.mkdir(parents=True)
        tm.mkdir(parents=True)

        for i in range(120):
            (ti / f"p{i}.png").write_bytes(b"x")
            (tm / f"p{i}_mask.png").write_bytes(b"y")

        ta = audit_pair(ti, tm)
        _, tc, _ = classify_candidate(
            target,
            ti,
            tm,
            ta,
        )
        assert tc == "EXCLUDED_TARGET_OR_EVALUATION"

        utility = root / "S00_polyp_locked_v1" / "source_train"
        ui = utility / "images"
        um = utility / "masks"
        ui.mkdir(parents=True)
        um.mkdir(parents=True)

        for i in range(120):
            (ui / f"u{i}.png").write_bytes(b"x")
            (um / f"u{i}_mask.png").write_bytes(b"y")

        ua = audit_pair(ui, um)
        _, uc, _ = classify_candidate(
            utility,
            ui,
            um,
            ua,
        )
        assert uc == "EXCLUDED_Q1_UTILITY_ASSET"

    print("STEM_NORMALIZATION_TEST_PASS")
    print("DATASET_TOKEN_TEST_PASS")
    print("PAIRING_TEST_PASS")
    print("TARGET_EXCLUSION_TEST_PASS")
    print("UTILITY_EXCLUSION_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Q1-R02R: audit clean raw polyp segmentation datasets for a "
            "new standardized 3×3 model panel."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run implementation tests only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
