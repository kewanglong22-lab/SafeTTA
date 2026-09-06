#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R00 — Heterogeneous Model Panel Feasibility & Asset Audit

Filesystem / metadata audit only:
- NO segmentation inference
- NO TTA inference
- NO training
- NO PyTorch checkpoint loading
- NO image opening

Expected project root:
    F:\\MEDSEG_SAFETTA
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


VERSION = "2026-08-19-Q1-R00-v1"
BUILD = "Q1_R00_HETEROGENEOUS_MODEL_PANEL_ASSET_AUDIT"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUTPUT_DIR = ROOT / "outputs" / "Q1_R00_heterogeneous_model_panel_asset_audit_v1"

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_R00_heterogeneous_model_panel_asset_audit_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "bbca3838965200978e77cd8377861ce3e7b58aec2312ec840b6501a0db9d5ded"
)

Q1_S03A_DIR = (
    ROOT
    / "outputs"
    / "Q1_S03A_prospective_segmentation_quality_proxy_v1"
)
Q1_S03A_LOCK = Q1_S03A_DIR / "Q1_S03A_QUALITY_PROXY_LOCK.json"
EXPECTED_Q1_S03A_LOCK_SHA256 = (
    "20e92a01507510b3738c6299d03ebd1a69ebe576788259f12146c03307bd9cfe"
)
EXPECTED_Q1_S03A_DECISION = "STOP_QUALITY_PROXY_CHECKPOINT_GENERALIZATION_FAILURE"

PREFERRED_SCAN_DIRS = (
    "code",
    "outputs",
    "checkpoints",
    "models",
    "weights",
    "docs",
    "configs",
)

EXCLUDED_DIR_NAMES = {
    "data",
    "dataset",
    "datasets",
    "raw",
    "images",
    "masks",
    "external",
    "external_data",
    "external_datasets",
    ".git",
    ".conda",
    "env",
    "venv",
    "__pycache__",
}

CHECKPOINT_SUFFIXES = (
    ".pth",
    ".pt",
    ".ckpt",
    ".tar",
    ".pth.tar",
    ".bin",
    ".safetensors",
)

REUSABLE_SUFFIXES = {
    ".csv",
    ".json",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".parquet",
}

TEXT_SUFFIXES = {
    ".py",
    ".ps1",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
}

REUSABLE_KEYWORDS = (
    "pred",
    "prediction",
    "infer",
    "inference",
    "oof",
    "feature",
    "utility",
    "dice",
    "metric",
    "lock",
    "manifest",
    "provenance",
    "checkpoint",
    "model",
    "score",
)

ARCH_PATTERNS = [
    ("PraNet", [
        r"\bpranet\b",
        r"pra[_\-\s]?net",
    ]),
    ("DeepLabV3-R50", [
        r"deeplabv3[^a-z0-9]{0,6}(r50|resnet50)",
        r"deeplab[^a-z0-9]{0,6}(r50|resnet50)",
        r"dlv3[^a-z0-9]{0,6}(r50|resnet50)",
    ]),
    ("DeepLabV3", [
        r"\bdeeplabv3\b",
        r"\bdeeplab\b",
        r"\bdlv3\b",
    ]),
    ("SegFormer", [
        r"\bsegformer\b",
    ]),
    ("TransUNet", [
        r"\btransunet\b",
        r"trans[_\-\s]?unet",
    ]),
    ("SwinUNet", [
        r"\bswinunet\b",
        r"swin[_\-\s]?unet",
    ]),
    ("nnUNet", [
        r"\bnnunet\b",
        r"nn[_\-\s]?unet",
    ]),
    ("UNet", [
        r"\bu[_\-\s]?net\b",
        r"\bunet\b",
    ]),
]

REFERENCE_TERMS = {
    "PraNet": (r"\bpranet\b", r"pra[_\-\s]?net"),
    "DeepLabV3-R50": (
        r"deeplabv3[^a-z0-9]{0,8}(r50|resnet50)",
        r"deeplab[^a-z0-9]{0,8}(r50|resnet50)",
        r"dlv3[^a-z0-9]{0,8}(r50|resnet50)",
    ),
    "DeepLabV3": (r"\bdeeplabv3\b", r"\bdeeplab\b", r"\bdlv3\b"),
    "UNet": (r"\bu[_\-\s]?net\b", r"\bunet\b"),
    "SegFormer": (r"\bsegformer\b",),
    "TransUNet": (r"\btransunet\b", r"trans[_\-\s]?unet"),
    "SwinUNet": (r"\bswinunet\b", r"swin[_\-\s]?unet"),
    "nnUNet": (r"\bnnunet\b", r"nn[_\-\s]?unet"),
}

MIN_ARCH_FAMILIES = 3
MIN_UNIQUE_STATES = 8
MIN_STATES_PER_ARCH = 2

DECISION_INSUFFICIENT = "PANEL_ASSET_AUDIT_INSUFFICIENT"
DECISION_ARCH = "PANEL_NEEDS_ADDITIONAL_ARCHITECTURES"
DECISION_CKPT = "PANEL_NEEDS_ADDITIONAL_CHECKPOINTS"
DECISION_PARTIAL = "PANEL_PARTIAL_REUSE_PLUS_NEW_INFERENCE"
DECISION_READY = "PANEL_READY_FOR_REUSE_STUDY"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in CHECKPOINT_SUFFIXES)


def is_excluded_dir(path: Path) -> bool:
    return path.name.lower() in EXCLUDED_DIR_NAMES


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def detect_architecture(text: str) -> Tuple[str, str, str]:
    low = text.lower()
    for arch, patterns in ARCH_PATTERNS:
        for pat in patterns:
            if re.search(pat, low, flags=re.IGNORECASE):
                return arch, "HEURISTIC_PATH", pat
    return "Unknown", "UNKNOWN", ""


def detect_seed(text: str) -> str:
    patterns = [
        r"(?:seed|sd)[_\-\s:=]*(20\d{6}|\d{1,6})",
        r"(2026081[789])",
    ]
    low = text.lower()
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return m.group(1)
    return ""


def detect_fold(text: str) -> str:
    m = re.search(r"(?:fold|f)[_\-\s:=]*(\d{1,2})", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def detect_epoch(text: str) -> str:
    m = re.search(
        r"(?:epoch|ep)[_\-\s:=]*(\d{1,5})",
        text,
        flags=re.IGNORECASE,
    )
    return m.group(1) if m else ""


def detect_training_state(text: str) -> str:
    low = text.lower()
    for token, label in [
        ("best", "best"),
        ("final", "final"),
        ("last", "last"),
        ("early", "early"),
        ("pretrain", "pretrained"),
        ("source", "source"),
        ("baseline", "baseline"),
    ]:
        if token in low:
            return label
    return ""


def candidate_reusable_asset(path: Path) -> bool:
    if path.suffix.lower() not in REUSABLE_SUFFIXES:
        return False
    low = str(path).lower()
    return any(k in low for k in REUSABLE_KEYWORDS)


def safe_read_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    try:
        if path.stat().st_size > max_bytes:
            with path.open("rb") as f:
                raw = f.read(max_bytes)
            return raw.decode("utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def iter_files(scan_roots: Sequence[Path]):
    seen = set()
    for root in scan_roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, topdown=True):
            current_path = Path(current)
            dirs[:] = sorted(
                d for d in dirs
                if d.lower() not in EXCLUDED_DIR_NAMES
                and not (current_path / d).is_symlink()
            )
            for name in sorted(files):
                p = current_path / name
                try:
                    key = str(p.resolve()).lower()
                except Exception:
                    key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield p


def choose_scan_roots(root: Path) -> List[Path]:
    roots = [root / name for name in PREFERRED_SCAN_DIRS if (root / name).exists()]
    if roots:
        return roots
    return [root]


def architecture_from_adjacent_metadata(
    checkpoint: Path,
    base_arch: str,
    base_status: str,
) -> Tuple[str, str, str]:
    if base_arch != "Unknown":
        return base_arch, base_status, "path_or_filename"

    parent = checkpoint.parent
    candidates = []
    for p in sorted(parent.glob("*")):
        if p == checkpoint or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        candidates.append(p)

    for p in candidates[:30]:
        text = safe_read_text(p, max_bytes=512 * 1024)
        arch, _, evidence = detect_architecture(text)
        if arch != "Unknown":
            return arch, "VERIFIED_EXPLICIT", f"adjacent:{p.name}:{evidence}"

    return "Unknown", "UNKNOWN", ""


# ---------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------

def scan_all_files(scan_roots: Sequence[Path]) -> List[Path]:
    return list(iter_files(scan_roots))


def checkpoint_inventory(files: Sequence[Path], root: Path):
    ckpts = [p for p in files if is_checkpoint(p)]

    rows = []
    for p in tqdm(
        ckpts,
        desc="Q1-R00 hashing checkpoints",
        unit="checkpoint",
        dynamic_ncols=True,
    ):
        stat = p.stat()
        path_text = str(p)
        arch, status, evidence = detect_architecture(path_text)
        arch, status, evidence2 = architecture_from_adjacent_metadata(
            p, arch, status
        )
        if evidence2:
            evidence = evidence2

        rows.append({
            "relative_path": rel(p, root),
            "filename": p.name,
            "bytes": stat.st_size,
            "gib": stat.st_size / (1024 ** 3),
            "sha256": file_sha256(p),
            "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "architecture": arch,
            "architecture_status": status,
            "architecture_evidence": evidence,
            "seed_hint": detect_seed(path_text),
            "fold_hint": detect_fold(path_text),
            "epoch_hint": detect_epoch(path_text),
            "training_state_hint": detect_training_state(path_text),
        })

    rows.sort(key=lambda r: (r["architecture"], r["sha256"], r["relative_path"]))

    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r)

    duplicate_rows = []
    for sha, rr in sorted(by_sha.items()):
        if len(rr) <= 1:
            continue
        canonical = rr[0]["relative_path"]
        for r in rr[1:]:
            duplicate_rows.append({
                "sha256": sha,
                "canonical_path": canonical,
                "duplicate_path": r["relative_path"],
                "architecture": r["architecture"],
            })

    return rows, duplicate_rows


def reusable_asset_inventory(files: Sequence[Path], root: Path):
    rows = []
    for p in files:
        if not candidate_reusable_asset(p):
            continue
        stat = p.stat()
        arch, status, evidence = detect_architecture(str(p))
        rows.append({
            "relative_path": rel(p, root),
            "filename": p.name,
            "extension": p.suffix.lower(),
            "bytes": stat.st_size,
            "architecture_hint": arch,
            "architecture_status": status,
            "seed_hint": detect_seed(str(p)),
            "fold_hint": detect_fold(str(p)),
            "asset_keyword_hits": ";".join(
                k for k in REUSABLE_KEYWORDS if k in str(p).lower()
            ),
        })
    rows.sort(key=lambda r: r["relative_path"])
    return rows


def code_model_references(files: Sequence[Path], root: Path):
    rows = []

    text_files = [
        p for p in files
        if p.suffix.lower() in TEXT_SUFFIXES
        and p.stat().st_size <= 8 * 1024 * 1024
    ]

    for p in tqdm(
        text_files,
        desc="Q1-R00 scanning code/config references",
        unit="file",
        dynamic_ncols=True,
    ):
        text = safe_read_text(p)
        if not text:
            continue
        low = text.lower()

        matched_arches = []
        for arch, patterns in REFERENCE_TERMS.items():
            if any(re.search(pat, low, flags=re.IGNORECASE) for pat in patterns):
                matched_arches.append(arch)

        path_refs = sorted(set(re.findall(
            r"""(?i)([A-Z]:\\[^"'`\r\n]+\.(?:pth|pt|ckpt|tar|bin|safetensors))""",
            text,
        )))[:20]

        if not matched_arches and not path_refs:
            continue

        rows.append({
            "relative_path": rel(p, root),
            "architecture_references": ";".join(sorted(set(matched_arches))),
            "checkpoint_path_references": ";".join(path_refs),
            "seed_hint": detect_seed(text[:500000]),
            "fold_hint": detect_fold(text[:500000]),
        })

    rows.sort(key=lambda r: r["relative_path"])
    return rows


def build_architecture_evidence(checkpoints, refs):
    explicit_ref_arches = set()
    for r in refs:
        for a in r["architecture_references"].split(";"):
            if a:
                explicit_ref_arches.add(a)

    for r in checkpoints:
        if r["architecture"] != "Unknown" and r["architecture"] in explicit_ref_arches:
            if r["architecture_status"] == "HEURISTIC_PATH":
                r["architecture_status"] = "VERIFIED_EXPLICIT"
                r["architecture_evidence"] = (
                    r["architecture_evidence"] + ";code_or_config_reference"
                ).strip(";")

    return checkpoints, explicit_ref_arches


def link_reusability(checkpoints, assets):
    asset_text = [
        (
            r,
            (
                r["relative_path"]
                + " "
                + r["filename"]
                + " "
                + r["architecture_hint"]
                + " "
                + r["seed_hint"]
            ).lower(),
        )
        for r in assets
    ]

    canonical_by_sha = {}
    states = []

    for ck in checkpoints:
        sha = ck["sha256"]
        if sha in canonical_by_sha:
            states.append({
                **ck,
                "reusability": "DUPLICATE_CHECKPOINT_COPY",
                "linked_asset_count": 0,
                "linked_asset_examples": "",
                "canonical_checkpoint_path": canonical_by_sha[sha],
            })
            continue

        canonical_by_sha[sha] = ck["relative_path"]

        arch = ck["architecture"]
        seed = ck["seed_hint"]
        hits = []

        if arch != "Unknown":
            for asset, text in asset_text:
                arch_hit = arch.lower() in text
                # tolerate DeepLabV3-R50 assets recorded as DeepLabV3
                if arch == "DeepLabV3-R50":
                    arch_hit = arch_hit or "deeplabv3" in text or "deeplab" in text
                seed_hit = (not seed) or (seed in text)
                if arch_hit and seed_hit:
                    hits.append(asset["relative_path"])

        if arch == "Unknown":
            label = "UNVERIFIED_ARCHITECTURE"
        elif hits:
            label = "READY_WITH_FROZEN_PREDICTIONS"
        else:
            label = "READY_CHECKPOINT_ONLY"

        states.append({
            **ck,
            "reusability": label,
            "linked_asset_count": len(hits),
            "linked_asset_examples": ";".join(hits[:8]),
            "canonical_checkpoint_path": ck["relative_path"],
        })

    states.sort(
        key=lambda r: (
            r["reusability"],
            r["architecture"],
            r["sha256"],
            r["relative_path"],
        )
    )
    return states


def architecture_summary(states, explicit_ref_arches):
    unique = {}
    for r in states:
        if r["reusability"] == "DUPLICATE_CHECKPOINT_COPY":
            continue
        unique[r["sha256"]] = r

    by_arch = defaultdict(list)
    for r in unique.values():
        if r["architecture"] != "Unknown":
            by_arch[r["architecture"]].append(r)

    all_arches = sorted(set(by_arch.keys()) | set(explicit_ref_arches))
    rows = []

    for arch in all_arches:
        rr = by_arch.get(arch, [])
        rows.append({
            "architecture": arch,
            "checkpoint_backed": bool(rr),
            "unique_checkpoint_states": len(rr),
            "ready_with_frozen_predictions": sum(
                r["reusability"] == "READY_WITH_FROZEN_PREDICTIONS"
                for r in rr
            ),
            "ready_checkpoint_only": sum(
                r["reusability"] == "READY_CHECKPOINT_ONLY"
                for r in rr
            ),
            "explicit_code_or_config_reference": arch in explicit_ref_arches,
            "minimum_2_states_pass": len(rr) >= MIN_STATES_PER_ARCH,
        })

    return rows


def panel_decision(states, arch_summary_rows):
    unique_states = {}
    for r in states:
        if r["reusability"] == "DUPLICATE_CHECKPOINT_COPY":
            continue
        unique_states[r["sha256"]] = r

    known_states = [
        r for r in unique_states.values()
        if r["architecture"] != "Unknown"
    ]

    checkpoint_backed_arches = sorted({
        r["architecture"]
        for r in known_states
    })

    counts = defaultdict(int)
    for r in known_states:
        counts[r["architecture"]] += 1

    deficits = []

    if not unique_states:
        decision = DECISION_INSUFFICIENT
        deficits.append("No checkpoint files were located.")
    elif len(checkpoint_backed_arches) < MIN_ARCH_FAMILIES:
        decision = DECISION_ARCH
        deficits.append(
            f"Need {MIN_ARCH_FAMILIES} checkpoint-backed architecture families; "
            f"found {len(checkpoint_backed_arches)}."
        )
    elif (
        len(known_states) < MIN_UNIQUE_STATES
        or any(counts[a] < MIN_STATES_PER_ARCH for a in checkpoint_backed_arches)
    ):
        decision = DECISION_CKPT
        if len(known_states) < MIN_UNIQUE_STATES:
            deficits.append(
                f"Need {MIN_UNIQUE_STATES} unique known-architecture states; "
                f"found {len(known_states)}."
            )
        for arch in checkpoint_backed_arches:
            if counts[arch] < MIN_STATES_PER_ARCH:
                deficits.append(
                    f"{arch} has {counts[arch]} unique state(s); "
                    f"need >= {MIN_STATES_PER_ARCH}."
                )
    else:
        all_have_predictions = all(
            r["reusability"] == "READY_WITH_FROZEN_PREDICTIONS"
            for r in known_states
        )
        decision = DECISION_READY if all_have_predictions else DECISION_PARTIAL
        if not all_have_predictions:
            missing = sum(
                r["reusability"] != "READY_WITH_FROZEN_PREDICTIONS"
                for r in known_states
            )
            deficits.append(
                f"{missing} checkpoint-backed state(s) lack confidently linked "
                f"frozen predictions and will require new inference."
            )

    # Record secondary deficits even if architecture deficit wins priority.
    if len(checkpoint_backed_arches) < MIN_ARCH_FAMILIES:
        needed = MIN_ARCH_FAMILIES - len(checkpoint_backed_arches)
        deficits.append(f"Additional architecture families required: {needed}.")
    if len(known_states) < MIN_UNIQUE_STATES:
        deficits.append(
            f"Additional unique checkpoint states required: "
            f"{MIN_UNIQUE_STATES - len(known_states)}."
        )

    for arch in checkpoint_backed_arches:
        if counts[arch] < MIN_STATES_PER_ARCH:
            deficits.append(
                f"Additional {arch} states required: "
                f"{MIN_STATES_PER_ARCH - counts[arch]}."
            )

    gap = {
        "decision": decision,
        "checkpoint_files_total": len(states),
        "unique_checkpoint_sha256_states_total": len(unique_states),
        "unique_known_architecture_states": len(known_states),
        "checkpoint_backed_architecture_families": checkpoint_backed_arches,
        "checkpoint_backed_architecture_count": len(checkpoint_backed_arches),
        "states_per_architecture": dict(sorted(counts.items())),
        "minimum_architecture_families": MIN_ARCH_FAMILIES,
        "minimum_unique_states": MIN_UNIQUE_STATES,
        "minimum_states_per_architecture": MIN_STATES_PER_ARCH,
        "deficits": sorted(set(deficits)),
    }
    return decision, gap, known_states


def recommended_plan(known_states, arch_summary_rows, decision):
    by_arch = defaultdict(list)
    for r in known_states:
        by_arch[r["architecture"]].append(r)

    priority = {
        "READY_WITH_FROZEN_PREDICTIONS": 0,
        "READY_CHECKPOINT_ONLY": 1,
        "UNVERIFIED_ARCHITECTURE": 2,
    }

    rows = []
    for arch in sorted(by_arch):
        rr = sorted(
            by_arch[arch],
            key=lambda r: (
                priority.get(r["reusability"], 99),
                r["seed_hint"],
                r["sha256"],
            ),
        )
        for i, r in enumerate(rr[:3], 1):
            rows.append({
                "plan_order": 0,
                "architecture": arch,
                "state_slot": i,
                "action": "REUSE_EXISTING_STATE",
                "checkpoint_path": r["relative_path"],
                "sha256": r["sha256"],
                "reusability": r["reusability"],
                "reason": "Existing unique checkpoint state.",
            })

        missing_to_two = max(0, MIN_STATES_PER_ARCH - len(rr))
        for j in range(missing_to_two):
            rows.append({
                "plan_order": 0,
                "architecture": arch,
                "state_slot": len(rr) + j + 1,
                "action": "ADD_INDEPENDENT_CHECKPOINT_STATE",
                "checkpoint_path": "",
                "sha256": "",
                "reusability": "",
                "reason": "Needed to reach >=2 states for this architecture.",
            })

    existing_arches = sorted(by_arch.keys())
    missing_arches = max(0, MIN_ARCH_FAMILIES - len(existing_arches))
    for i in range(missing_arches):
        generic_num = len(existing_arches) + i + 1
        arch = f"NEW_ARCHITECTURE_FAMILY_{generic_num}"
        for slot in (1, 2):
            rows.append({
                "plan_order": 0,
                "architecture": arch,
                "state_slot": slot,
                "action": "ADD_NEW_ARCHITECTURE_CHECKPOINT",
                "checkpoint_path": "",
                "sha256": "",
                "reusability": "",
                "reason": "Needed for heterogeneous >=3-family LOMO panel.",
            })

    # Ensure at least 8 total planned states.
    planned_existing = sum(
        1 for r in rows if r["action"] == "REUSE_EXISTING_STATE"
    )
    planned_new = sum(
        1 for r in rows if r["action"] != "REUSE_EXISTING_STATE"
    )
    total = planned_existing + planned_new

    if total < MIN_UNIQUE_STATES:
        target_arches = sorted({
            r["architecture"] for r in rows
        })
        idx = 0
        while total < MIN_UNIQUE_STATES and target_arches:
            arch = target_arches[idx % len(target_arches)]
            current_slots = [
                int(r["state_slot"])
                for r in rows
                if r["architecture"] == arch
            ]
            next_slot = max(current_slots, default=0) + 1
            rows.append({
                "plan_order": 0,
                "architecture": arch,
                "state_slot": next_slot,
                "action": "ADD_INDEPENDENT_CHECKPOINT_STATE",
                "checkpoint_path": "",
                "sha256": "",
                "reusability": "",
                "reason": "Needed to reach >=8 total unique model states.",
            })
            total += 1
            idx += 1

    action_priority = {
        "REUSE_EXISTING_STATE": 0,
        "ADD_INDEPENDENT_CHECKPOINT_STATE": 1,
        "ADD_NEW_ARCHITECTURE_CHECKPOINT": 2,
    }
    rows.sort(
        key=lambda r: (
            action_priority.get(r["action"], 9),
            r["architecture"],
            int(r["state_slot"]),
            r["checkpoint_path"],
        )
    )
    for i, r in enumerate(rows, 1):
        r["plan_order"] = i

    return rows


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def self_test():
    assert is_checkpoint(Path("a.pth"))
    assert is_checkpoint(Path("a.pth.tar"))
    assert is_checkpoint(Path("a.safetensors"))
    assert not is_checkpoint(Path("a.csv"))

    arch, status, _ = detect_architecture(r"F:\x\PraNet\seed_20260817\best.pth")
    assert arch == "PraNet"
    assert status == "HEURISTIC_PATH"

    arch, _, _ = detect_architecture("deeplabv3_resnet50_seed3.ckpt")
    assert arch == "DeepLabV3-R50"

    assert detect_seed("model_seed_20260819_best.pth") == "20260819"
    assert detect_fold("fold_3/model.pth") == "3"
    assert detect_epoch("epoch_80.pth") == "80"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "code").mkdir()
        (root / "outputs").mkdir()
        (root / "checkpoints" / "PraNet" / "seed_1").mkdir(parents=True)
        (root / "checkpoints" / "DeepLabV3_R50" / "seed_2").mkdir(parents=True)
        (root / "datasets" / "big").mkdir(parents=True)

        p1 = root / "checkpoints" / "PraNet" / "seed_1" / "best.pth"
        p2 = root / "checkpoints" / "DeepLabV3_R50" / "seed_2" / "best.pth"
        p3 = root / "checkpoints" / "PraNet" / "seed_1" / "copy.pth"
        p1.write_bytes(b"pranet-state")
        p2.write_bytes(b"deeplab-state")
        p3.write_bytes(b"pranet-state")

        (root / "outputs" / "pranet_seed_1_predictions.csv").write_text(
            "a,b\n1,2\n", encoding="utf-8"
        )
        (root / "code" / "models.py").write_text(
            "MODEL='PraNet'\nOTHER='DeepLabV3 ResNet50'\n",
            encoding="utf-8",
        )
        (root / "datasets" / "big" / "do_not_scan.pth").write_bytes(b"x")

        scan_roots = choose_scan_roots(root)
        files = scan_all_files(scan_roots)
        assert all("do_not_scan.pth" not in str(p) for p in files)

        ck, dup = checkpoint_inventory(files, root)
        assert len(ck) == 3
        assert len(dup) == 1

        assets = reusable_asset_inventory(files, root)
        refs = code_model_references(files, root)
        ck, explicit = build_architecture_evidence(ck, refs)
        states = link_reusability(ck, assets)
        summary = architecture_summary(states, explicit)
        decision, gap, known = panel_decision(states, summary)
        plan = recommended_plan(known, summary, decision)

        assert decision == DECISION_ARCH
        assert len(plan) >= MIN_UNIQUE_STATES

    print("FILE_CLASSIFICATION_TEST_PASS")
    print("ARCHITECTURE_HINT_TEST_PASS")
    print("DUPLICATE_SHA_TEST_PASS")
    print("EXCLUSION_TEST_PASS")
    print("PANEL_DECISION_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    root = args.root

    if not root.exists():
        raise FileNotFoundError(root)

    output_dir = args.output_dir

    # Enforce project-local output in production.
    root_resolved = root.resolve()
    output_resolved = output_dir.resolve()
    try:
        output_resolved.relative_to(root_resolved)
    except Exception:
        raise RuntimeError(
            f"Output must stay under project root: {root}; got {output_dir}"
        )

    if str(output_dir).lower().startswith("c:\\"):
        raise RuntimeError("C-drive output is forbidden.")

    if output_dir.exists():
        raise FileExistsError(f"Q1-R00 output already exists: {output_dir}")

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-R00 build exists: {build_dir}. "
            "Remove only this __building directory if the previous audit "
            "was technically interrupted."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-R00 HETEROGENEOUS MODEL PANEL ASSET AUDIT =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Root: {root}")
    print("Segmentation inference=NO")
    print("TTA inference=NO")
    print("Training=NO")
    print("Checkpoint loading into PyTorch=NO")
    print("Image opening=NO")
    print()

    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    protocol_sha = file_sha256(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Q1-R00 protocol SHA mismatch: "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={protocol_sha}"
        )

    if not Q1_S03A_LOCK.exists():
        raise FileNotFoundError(Q1_S03A_LOCK)
    q1_s03a_lock_sha = file_sha256(Q1_S03A_LOCK)
    if q1_s03a_lock_sha != EXPECTED_Q1_S03A_LOCK_SHA256:
        raise RuntimeError(
            "Q1-S03A lock SHA mismatch: "
            f"expected={EXPECTED_Q1_S03A_LOCK_SHA256} actual={q1_s03a_lock_sha}"
        )
    q1_s03a_lock = json.loads(Q1_S03A_LOCK.read_text(encoding="utf-8"))
    if q1_s03a_lock.get("decision") != EXPECTED_Q1_S03A_DECISION:
        raise RuntimeError(
            "Unexpected Q1-S03A decision: "
            f"{q1_s03a_lock.get('decision')}"
        )

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)

    scan_roots = choose_scan_roots(root)
    scan_roots_path = build_dir / "scan_roots.json"
    write_json(
        scan_roots_path,
        {
            "project_root": str(root),
            "scan_roots": [str(p) for p in scan_roots],
            "excluded_directory_names": sorted(EXCLUDED_DIR_NAMES),
        },
    )

    print("Scanning project assets...")
    files = scan_all_files(scan_roots)
    print(f"Files discovered after exclusions: {len(files)}")

    checkpoints, duplicates = checkpoint_inventory(files, root)
    assets = reusable_asset_inventory(files, root)
    refs = code_model_references(files, root)
    checkpoints, explicit_ref_arches = build_architecture_evidence(
        checkpoints,
        refs,
    )
    states = link_reusability(checkpoints, assets)
    arch_rows = architecture_summary(states, explicit_ref_arches)
    decision, gap, known_states = panel_decision(states, arch_rows)
    plan = recommended_plan(known_states, arch_rows, decision)

    checkpoint_path = build_dir / "checkpoint_inventory.csv"
    write_csv(
        checkpoint_path,
        checkpoints,
        [
            "relative_path",
            "filename",
            "bytes",
            "gib",
            "sha256",
            "modified_iso",
            "architecture",
            "architecture_status",
            "architecture_evidence",
            "seed_hint",
            "fold_hint",
            "epoch_hint",
            "training_state_hint",
        ],
    )

    duplicate_path = build_dir / "checkpoint_duplicates.csv"
    write_csv(
        duplicate_path,
        duplicates,
        [
            "sha256",
            "canonical_path",
            "duplicate_path",
            "architecture",
        ],
    )

    asset_path = build_dir / "prediction_feature_asset_inventory.csv"
    write_csv(
        asset_path,
        assets,
        [
            "relative_path",
            "filename",
            "extension",
            "bytes",
            "architecture_hint",
            "architecture_status",
            "seed_hint",
            "fold_hint",
            "asset_keyword_hits",
        ],
    )

    refs_path = build_dir / "code_model_references.csv"
    write_csv(
        refs_path,
        refs,
        [
            "relative_path",
            "architecture_references",
            "checkpoint_path_references",
            "seed_hint",
            "fold_hint",
        ],
    )

    arch_path = build_dir / "architecture_summary.csv"
    write_csv(
        arch_path,
        arch_rows,
        [
            "architecture",
            "checkpoint_backed",
            "unique_checkpoint_states",
            "ready_with_frozen_predictions",
            "ready_checkpoint_only",
            "explicit_code_or_config_reference",
            "minimum_2_states_pass",
        ],
    )

    states_path = build_dir / "model_state_reusability.csv"
    write_csv(
        states_path,
        states,
        [
            "relative_path",
            "filename",
            "bytes",
            "gib",
            "sha256",
            "modified_iso",
            "architecture",
            "architecture_status",
            "architecture_evidence",
            "seed_hint",
            "fold_hint",
            "epoch_hint",
            "training_state_hint",
            "reusability",
            "linked_asset_count",
            "linked_asset_examples",
            "canonical_checkpoint_path",
        ],
    )

    gap_path = build_dir / "panel_gap_analysis.json"
    write_json(gap_path, gap)

    plan_path = build_dir / "recommended_panel_plan.csv"
    write_csv(
        plan_path,
        plan,
        [
            "plan_order",
            "architecture",
            "state_slot",
            "action",
            "checkpoint_path",
            "sha256",
            "reusability",
            "reason",
        ],
    )

    provenance = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "q1_s03a_lock_sha256": q1_s03a_lock_sha,
        "q1_s03a_decision": q1_s03a_lock["decision"],
        "root": str(root),
        "scan_roots": [str(p) for p in scan_roots],
        "files_discovered": len(files),
        "checkpoint_files": len(checkpoints),
        "duplicate_checkpoint_copies": len(duplicates),
        "reusable_asset_candidates": len(assets),
        "code_config_reference_files": len(refs),
        "project_code_imported": False,
        "checkpoint_loaded_into_pytorch": False,
        "image_opened": False,
        "segmentation_inference": False,
        "tta_inference": False,
        "training": False,
        "decision": decision,
    }
    provenance_path = build_dir / "provenance_audit.json"
    write_json(provenance_path, provenance)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Integrity assertions.
    listed_checkpoint_paths = [
        root / r["relative_path"]
        for r in checkpoints
    ]
    listed_asset_paths = [
        root / r["relative_path"]
        for r in assets
    ]
    listed_ref_paths = [
        root / r["relative_path"]
        for r in refs
    ]
    for p in listed_checkpoint_paths + listed_asset_paths + listed_ref_paths:
        if not p.exists():
            raise RuntimeError(f"Listed file disappeared during audit: {p}")

    integrity = {
        "q1_s03a_lock_verified": q1_s03a_lock_sha == EXPECTED_Q1_S03A_LOCK_SHA256,
        "q1_s03a_decision_verified": q1_s03a_lock["decision"] == EXPECTED_Q1_S03A_DECISION,
        "project_root_exists": root.exists(),
        "output_under_project_root": True,
        "c_drive_output_not_used": not str(output_dir).lower().startswith("c:\\"),
        "project_python_code_not_imported": True,
        "checkpoint_not_loaded_into_pytorch": True,
        "images_not_opened": True,
        "segmentation_inference_not_run": True,
        "training_not_run": True,
        "checkpoint_sha256_computed_from_raw_bytes": True,
        "duplicate_detection_uses_sha256": True,
        "excluded_data_dirs_not_recursively_traversed": True,
        "all_listed_files_exist_at_commit": True,
        "architecture_evidence_status_recorded": all(
            r["architecture_status"] in {
                "VERIFIED_EXPLICIT",
                "HEURISTIC_PATH",
                "UNKNOWN",
            }
            for r in checkpoints
        ),
        "deterministic_sorting_used": True,
    }
    if not all(integrity.values()):
        raise RuntimeError(
            f"Q1-R00 integrity failure: "
            f"{[k for k,v in integrity.items() if not v]}"
        )

    # Human-readable log.
    print_arches = gap["checkpoint_backed_architecture_families"]
    summary = f"""===== Q1-R00 HETEROGENEOUS MODEL PANEL ASSET AUDIT =====
Script version: {VERSION}
Build: {BUILD}

Project root:
  {root}

Protocol SHA256:
  {protocol_sha}

Scan:
  scan roots={len(scan_roots)}
  discovered files={len(files)}
  checkpoint files={len(checkpoints)}
  duplicate checkpoint copies={len(duplicates)}
  reusable prediction/feature candidates={len(assets)}
  code/config reference files={len(refs)}

Checkpoint-backed model panel:
  architecture families={len(print_arches)}
  families={print_arches}
  unique known-architecture checkpoint states={gap['unique_known_architecture_states']}
  unique checkpoint SHA256 states total={gap['unique_checkpoint_sha256_states_total']}
  states per architecture={gap['states_per_architecture']}

Frozen minimum:
  architecture families >= {MIN_ARCH_FAMILIES}
  unique states >= {MIN_UNIQUE_STATES}
  states per included architecture >= {MIN_STATES_PER_ARCH}

Deficits:
"""
    if gap["deficits"]:
        for d in gap["deficits"]:
            summary += f"  - {d}\n"
    else:
        summary += "  - NONE\n"

    summary += f"""
Decision: {decision}

Next mapping:
  {DECISION_READY}
    -> Q1-R01 Heterogeneous Model Utility Dataset Audit

  {DECISION_PARTIAL}
    -> Q1-R01A Frozen-Inference Completion Plan

  {DECISION_CKPT}
    -> Q1-R01B Minimal Checkpoint Expansion Plan

  {DECISION_ARCH}
    -> Q1-R01C Third-Architecture Selection and Training Plan

  {DECISION_INSUFFICIENT}
    -> manually resolve project asset locations first.

No segmentation inference, TTA inference, checkpoint loading, or training was performed.

[OK] Outputs: {output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "scan_roots": scan_roots_path,
        "checkpoint_inventory": checkpoint_path,
        "checkpoint_duplicates": duplicate_path,
        "prediction_feature_asset_inventory": asset_path,
        "code_model_references": refs_path,
        "architecture_summary": arch_path,
        "model_state_reusability": states_path,
        "panel_gap_analysis": gap_path,
        "recommended_panel_plan": plan_path,
        "provenance_audit": provenance_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "q1_s03a_lock_sha256": EXPECTED_Q1_S03A_LOCK_SHA256,
        "q1_s03a_decision": EXPECTED_Q1_S03A_DECISION,
        "artifacts": {
            name: {
                "filename": p.name,
                "sha256": file_sha256(p),
            }
            for name, p in artifact_paths.items()
        },
        "integrity": integrity,
        "decision": decision,
    }

    lock_path = build_dir / "Q1_R00_MODEL_PANEL_ASSET_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    # Verify artifact immutability before commit.
    for name, p in artifact_paths.items():
        if file_sha256(p) != lock["artifacts"][name]["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(output_dir)

    print()
    print((output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R00 LOCK:",
        output_dir / "Q1_R00_MODEL_PANEL_ASSET_LOCK.json",
    )
    print("Q1-R00 LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Q1-R00: audit existing SafeTTA model/checkpoint/prediction assets "
            "for a heterogeneous leave-one-model-out model panel."
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
