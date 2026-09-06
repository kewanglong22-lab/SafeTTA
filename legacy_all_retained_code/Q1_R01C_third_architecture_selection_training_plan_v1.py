#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R01C — Third-Architecture Selection and Training Plan

Frozen architecture:
    SegFormer-B0 / MiT-B0

This stage DOES NOT train anything.
It verifies the Q1-R00 panel, deterministically selects a balanced
PraNet/DeepLabV3-R50 subset, audits source-training interface evidence,
and checks whether the current Python environment has a practical
SegFormer implementation path.

Expected root:
    F:\\MEDSEG_SAFETTA
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


VERSION = "2026-08-19-Q1-R01C-v1"
BUILD = "Q1_R01C_THIRD_ARCHITECTURE_SELECTION_TRAINING_PLAN"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "Q1_R01C_third_architecture_selection_training_plan_v1"
)

PROTOCOL = (
    ROOT
    / "docs"
    / "Q1_R01C_third_architecture_selection_training_plan_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b99df8e67c6af0db93a1891683a95a7ed361fb3c254eb0afd84ea51b5702b25d"
)

Q1_R00_DIR = (
    ROOT
    / "outputs"
    / "Q1_R00_heterogeneous_model_panel_asset_audit_v1"
)
Q1_R00_LOCK = Q1_R00_DIR / "Q1_R00_MODEL_PANEL_ASSET_LOCK.json"
EXPECTED_Q1_R00_LOCK_SHA256 = (
    "052d03456324fef03430cd090f2c2df28b3a0330c7c8467f9351189d4394a6ee"
)
EXPECTED_Q1_R00_DECISION = "PANEL_NEEDS_ADDITIONAL_ARCHITECTURES"

R00_STATES = Q1_R00_DIR / "model_state_reusability.csv"
R00_ARCH = Q1_R00_DIR / "architecture_summary.csv"
R00_GAP = Q1_R00_DIR / "panel_gap_analysis.json"

THIRD_ARCHITECTURE = "SegFormer"
THIRD_VARIANT = "SegFormer-B0"
THIRD_ENCODER = "MiT-B0"
NEW_SEEDS = (20260820, 20260821, 20260822)

EXISTING_ARCHES = ("PraNet", "DeepLabV3-R50")
BALANCED_STATES_PER_ARCH = 3
BALANCED_TOTAL_STATES = 9

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
    ".ini",
    ".cfg",
}

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

SOURCE_TRAIN_HINTS = (
    "train_image",
    "train_images",
    "train_img",
    "train_root",
    "train_dir",
    "training_images",
    "source_train",
    "source_images",
)

SOURCE_MASK_HINTS = (
    "train_mask",
    "train_masks",
    "mask_root",
    "mask_dir",
    "training_masks",
    "source_mask",
    "gt_root",
    "label_root",
)

VAL_HINTS = (
    "val_image",
    "val_images",
    "val_img",
    "val_root",
    "val_dir",
    "validation",
    "source_val",
)

MASK_GENERIC_HINTS = (
    "mask",
    "masks",
    "gt",
    "ground_truth",
    "label",
)

TRAINING_SETTING_HINTS = (
    "batch_size",
    "epochs",
    "epoch",
    "optimizer",
    "learning_rate",
    "lr",
    "resize",
    "image_size",
    "img_size",
    "augmentation",
    "normalize",
    "loss",
    "dice",
    "bce",
)

SEGFORMER_PATTERNS = (
    r"\bsegformer\b",
    r"\bmit[_\-]?b0\b",
    r"\bmix.?transformer\b",
)

DECISION_INVALID = "INVALID_Q1_R01C_UPSTREAM_PANEL"
DECISION_DATA = "NEEDS_SOURCE_DATA_INTERFACE_RESOLUTION"
DECISION_SETUP = "NEEDS_SEGFORMER_IMPLEMENTATION_SETUP"
DECISION_READY = "READY_TO_IMPLEMENT_SEGFORMER_B0_TRAINING"


# ---------------------------------------------------------------------
# IO helpers
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


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_read_text(path: Path, max_bytes: int = 3 * 1024 * 1024) -> str:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            with path.open("rb") as f:
                raw = f.read(max_bytes)
            return raw.decode("utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def split_semicolon(value: str) -> List[str]:
    return [x for x in (value or "").split(";") if x]


# ---------------------------------------------------------------------
# Upstream reproduction
# ---------------------------------------------------------------------

def validate_upstream(root: Path):
    if not Q1_R00_LOCK.exists():
        raise FileNotFoundError(Q1_R00_LOCK)

    lock_sha = file_sha256(Q1_R00_LOCK)
    if lock_sha != EXPECTED_Q1_R00_LOCK_SHA256:
        raise RuntimeError(
            "Q1-R00 lock SHA mismatch: "
            f"expected={EXPECTED_Q1_R00_LOCK_SHA256} actual={lock_sha}"
        )

    lock = json.loads(Q1_R00_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_Q1_R00_DECISION:
        raise RuntimeError(
            f"Unexpected Q1-R00 decision: {lock.get('decision')}"
        )

    for p in (R00_STATES, R00_ARCH, R00_GAP):
        if not p.exists():
            raise FileNotFoundError(p)

    states, _ = read_csv(R00_STATES)
    arch_rows, _ = read_csv(R00_ARCH)
    gap = json.loads(R00_GAP.read_text(encoding="utf-8"))

    backed_arches = sorted(
        r["architecture"]
        for r in arch_rows
        if str(r.get("checkpoint_backed", "")).lower() in {"true", "1"}
    )

    if "PraNet" not in backed_arches:
        raise RuntimeError("Q1-R00 no longer reproduces checkpoint-backed PraNet.")
    if "DeepLabV3-R50" not in backed_arches:
        raise RuntimeError(
            "Q1-R00 no longer reproduces checkpoint-backed DeepLabV3-R50."
        )

    unique_counts = defaultdict(set)
    for r in states:
        if r["reusability"] == "DUPLICATE_CHECKPOINT_COPY":
            continue
        if r["architecture"] in EXISTING_ARCHES:
            unique_counts[r["architecture"]].add(r["sha256"])

        # Checkpoint file must still exist.
        if r["architecture"] in EXISTING_ARCHES:
            p = root / r["relative_path"]
            if not p.exists():
                raise RuntimeError(f"Upstream checkpoint disappeared: {p}")
            if file_sha256(p) != r["sha256"]:
                raise RuntimeError(f"Upstream checkpoint SHA changed: {p}")

    for arch in EXISTING_ARCHES:
        if len(unique_counts[arch]) < 2:
            raise RuntimeError(
                f"Upstream panel invalid: {arch} has <2 unique states."
            )

    return states, arch_rows, gap, {
        "q1_r00_lock_sha256": lock_sha,
        "q1_r00_decision": lock["decision"],
        "checkpoint_backed_architectures": backed_arches,
        "pranet_unique_states": len(unique_counts["PraNet"]),
        "deeplabv3_r50_unique_states": len(unique_counts["DeepLabV3-R50"]),
        "upstream_valid": True,
    }


# ---------------------------------------------------------------------
# Balanced existing-panel selection
# ---------------------------------------------------------------------

def select_balanced_states(states):
    priority = {
        "READY_WITH_FROZEN_PREDICTIONS": 0,
        "READY_CHECKPOINT_ONLY": 1,
        "UNVERIFIED_ARCHITECTURE": 2,
    }

    selected = []

    for arch in EXISTING_ARCHES:
        candidates = []
        seen_sha = set()

        for r in states:
            if r["architecture"] != arch:
                continue
            if r["reusability"] == "DUPLICATE_CHECKPOINT_COPY":
                continue
            if r["sha256"] in seen_sha:
                continue
            seen_sha.add(r["sha256"])
            candidates.append(r)

        candidates.sort(
            key=lambda r: (
                priority.get(r["reusability"], 99),
                0 if r["seed_hint"] else 1,
                r["seed_hint"],
                0 if r["architecture_status"] == "VERIFIED_EXPLICIT" else 1,
                r["sha256"],
                r["relative_path"],
            )
        )

        # Prefer distinct seed hints before taking repeated/unknown seed hints.
        chosen = []
        used_seed_hints = set()

        for r in candidates:
            seed = r["seed_hint"]
            if seed and seed not in used_seed_hints:
                chosen.append(r)
                used_seed_hints.add(seed)
            if len(chosen) == BALANCED_STATES_PER_ARCH:
                break

        if len(chosen) < BALANCED_STATES_PER_ARCH:
            chosen_sha = {r["sha256"] for r in chosen}
            for r in candidates:
                if r["sha256"] in chosen_sha:
                    continue
                chosen.append(r)
                chosen_sha.add(r["sha256"])
                if len(chosen) == BALANCED_STATES_PER_ARCH:
                    break

        for slot, r in enumerate(chosen, 1):
            selected.append({
                "architecture": arch,
                "panel_slot": slot,
                "relative_path": r["relative_path"],
                "sha256": r["sha256"],
                "seed_hint": r["seed_hint"],
                "fold_hint": r["fold_hint"],
                "epoch_hint": r["epoch_hint"],
                "training_state_hint": r["training_state_hint"],
                "architecture_status": r["architecture_status"],
                "reusability": r["reusability"],
                "linked_asset_count": r["linked_asset_count"],
            })

    return selected


# ---------------------------------------------------------------------
# Environment readiness
# ---------------------------------------------------------------------

def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def environment_readiness():
    modules = {
        name: module_available(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "timm",
            "mmseg",
            "mmengine",
        )
    }

    segformer_impl_paths = []
    if modules["transformers"]:
        segformer_impl_paths.append("transformers")
    if modules["mmseg"]:
        segformer_impl_paths.append("mmseg")

    return {
        "modules": modules,
        "pytorch_available": modules["torch"],
        "installed_segformer_implementation_paths": segformer_impl_paths,
    }


# ---------------------------------------------------------------------
# Static project text scanning
# ---------------------------------------------------------------------

def iter_text_files(root: Path):
    preferred = [
        root / "code",
        root / "configs",
        root / "docs",
        root / "outputs",
    ]
    scan_roots = [p for p in preferred if p.exists()]
    if not scan_roots:
        scan_roots = [root]

    seen = set()
    for sr in scan_roots:
        for current, dirs, files in os.walk(sr, topdown=True):
            cp = Path(current)
            dirs[:] = sorted(
                d for d in dirs
                if d.lower() not in EXCLUDED_DIR_NAMES
                and not (cp / d).is_symlink()
            )
            for name in sorted(files):
                p = cp / name
                if p.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                try:
                    if p.stat().st_size > 8 * 1024 * 1024:
                        continue
                except Exception:
                    continue

                key = str(p.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield p


def extract_path_like_strings(text: str):
    # Windows paths and quoted relative/project paths.
    patterns = [
        r"""[A-Za-z]:\\[^"'`\r\n]+""",
        r"""(?:\.\.?[\\/])?[\w.\-\\/ ]+(?:train|val|mask|image|dataset|manifest)[\w.\-\\/ ]*""",
    ]
    hits = []
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            s = m.strip().strip("'\"` ,;)")
            if s and len(s) <= 500:
                hits.append(s)
    return sorted(set(hits))


def scan_source_interface(root: Path):
    data_candidates = []
    code_candidates = []
    local_segformer_refs = []

    files = list(iter_text_files(root))

    for p in tqdm(
        files,
        desc="Q1-R01C scanning source-training interface",
        unit="file",
        dynamic_ncols=True,
    ):
        text = safe_read_text(p)
        if not text:
            continue

        low = text.lower()
        path_hits = extract_path_like_strings(text)

        source_train_hits = [
            h for h in SOURCE_TRAIN_HINTS if h in low
        ]
        source_mask_hits = [
            h for h in SOURCE_MASK_HINTS if h in low
        ]
        val_hits = [
            h for h in VAL_HINTS if h in low
        ]
        generic_mask_hits = [
            h for h in MASK_GENERIC_HINTS if h in low
        ]
        setting_hits = [
            h for h in TRAINING_SETTING_HINTS if h in low
        ]

        architecture_hits = []
        if re.search(r"\bpranet\b", low):
            architecture_hits.append("PraNet")
        if re.search(r"\bdeeplab(v3)?\b", low):
            architecture_hits.append("DeepLabV3")
        if any(re.search(pat, low) for pat in SEGFORMER_PATTERNS):
            architecture_hits.append("SegFormer")
            local_segformer_refs.append(rel(p, root))

        if (
            source_train_hits
            or source_mask_hits
            or val_hits
            or path_hits
        ):
            data_candidates.append({
                "relative_path": rel(p, root),
                "source_train_hits": ";".join(source_train_hits),
                "source_mask_hits": ";".join(source_mask_hits),
                "validation_hits": ";".join(val_hits),
                "generic_mask_hits": ";".join(generic_mask_hits),
                "path_like_evidence": ";".join(path_hits[:15]),
            })

        if architecture_hits and setting_hits:
            code_candidates.append({
                "relative_path": rel(p, root),
                "architecture_hits": ";".join(sorted(set(architecture_hits))),
                "training_setting_hits": ";".join(setting_hits),
                "path_like_evidence": ";".join(path_hits[:15]),
            })

    # Conservative interface resolution:
    # need explicit training image/root evidence, mask/label evidence, AND validation
    # evidence somewhere in project text.
    has_train = any(r["source_train_hits"] for r in data_candidates)
    has_mask = any(
        r["source_mask_hits"] or r["generic_mask_hits"]
        for r in data_candidates
    )
    has_val = any(r["validation_hits"] for r in data_candidates)

    # Existing training entrypoint evidence helps avoid accepting random docs only.
    has_existing_train_code = any(
        ("PraNet" in r["architecture_hits"] or "DeepLabV3" in r["architecture_hits"])
        for r in code_candidates
    )

    interface_sufficient = (
        has_train
        and has_mask
        and has_val
        and has_existing_train_code
    )

    protocol_evidence = {
        "text_files_scanned": len(files),
        "has_explicit_source_train_evidence": has_train,
        "has_mask_or_label_evidence": has_mask,
        "has_validation_evidence": has_val,
        "has_existing_pranet_or_deeplab_training_reference": has_existing_train_code,
        "source_interface_sufficient": interface_sufficient,
        "local_segformer_reference_files": sorted(set(local_segformer_refs)),
    }

    return (
        sorted(data_candidates, key=lambda r: r["relative_path"]),
        sorted(code_candidates, key=lambda r: r["relative_path"]),
        protocol_evidence,
    )


# ---------------------------------------------------------------------
# Training/future panel manifests
# ---------------------------------------------------------------------

def segformer_manifest():
    rows = []
    for slot, seed in enumerate(NEW_SEEDS, 1):
        rows.append({
            "architecture": THIRD_ARCHITECTURE,
            "variant": THIRD_VARIANT,
            "encoder": THIRD_ENCODER,
            "state_slot": slot,
            "seed": seed,
            "training_data": "SAME_FROZEN_SOURCE_TRAIN_AS_EXISTING_SAFETTA_MODELS",
            "validation_data": "SAME_FROZEN_SOURCE_VAL_PROTOCOL_AS_EXISTING_SAFETTA_MODELS",
            "target_data_allowed": "NO",
            "utility_labels_allowed": "NO",
            "model_size_sweep": "NO",
            "checkpoint_selection": "SOURCE_VALIDATION_ONLY",
            "status": "PLANNED_NOT_TRAINED",
        })
    return rows


def future_panel(existing_selected, new_manifest):
    rows = []
    order = 1

    for arch in EXISTING_ARCHES:
        rr = [
            r for r in existing_selected
            if r["architecture"] == arch
        ]
        for r in rr:
            rows.append({
                "panel_order": order,
                "architecture": arch,
                "state_slot": r["panel_slot"],
                "origin": "EXISTING",
                "seed": r["seed_hint"],
                "checkpoint_path": r["relative_path"],
                "checkpoint_sha256": r["sha256"],
                "status": r["reusability"],
            })
            order += 1

    for r in new_manifest:
        rows.append({
            "panel_order": order,
            "architecture": THIRD_VARIANT,
            "state_slot": r["state_slot"],
            "origin": "TO_TRAIN",
            "seed": r["seed"],
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "status": "PLANNED_NOT_TRAINED",
        })
        order += 1

    return rows


# ---------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------

def choose_decision(upstream, selected, env, source_protocol):
    counts = defaultdict(int)
    for r in selected:
        counts[r["architecture"]] += 1

    upstream_valid = (
        upstream["upstream_valid"]
        and counts["PraNet"] >= 2
        and counts["DeepLabV3-R50"] >= 2
    )

    if not upstream_valid:
        return DECISION_INVALID, {
            "upstream_valid": False,
            "source_interface_sufficient": False,
            "segformer_implementation_available": False,
        }

    source_ok = bool(source_protocol["source_interface_sufficient"])

    local_segformer = bool(source_protocol["local_segformer_reference_files"])
    implementation_ok = (
        local_segformer
        or bool(env["installed_segformer_implementation_paths"])
    )
    torch_ok = bool(env["pytorch_available"])

    if not source_ok:
        decision = DECISION_DATA
    elif not (torch_ok and implementation_ok):
        decision = DECISION_SETUP
    else:
        decision = DECISION_READY

    return decision, {
        "upstream_valid": upstream_valid,
        "selected_pranet_states": counts["PraNet"],
        "selected_deeplabv3_r50_states": counts["DeepLabV3-R50"],
        "source_interface_sufficient": source_ok,
        "torch_available": torch_ok,
        "local_segformer_reference": local_segformer,
        "installed_segformer_implementation_paths": env[
            "installed_segformer_implementation_paths"
        ],
        "segformer_implementation_available": implementation_ok,
    }


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def self_test():
    assert THIRD_ARCHITECTURE == "SegFormer"
    assert THIRD_VARIANT == "SegFormer-B0"
    assert NEW_SEEDS == (20260820, 20260821, 20260822)
    assert BALANCED_STATES_PER_ARCH == 3
    assert BALANCED_TOTAL_STATES == 9

    states = []
    for arch, n in (("PraNet", 4), ("DeepLabV3-R50", 3)):
        for i in range(n):
            states.append({
                "architecture": arch,
                "reusability": (
                    "READY_WITH_FROZEN_PREDICTIONS"
                    if i < 2
                    else "READY_CHECKPOINT_ONLY"
                ),
                "sha256": f"{arch}-{i}",
                "seed_hint": str(100+i),
                "fold_hint": "",
                "epoch_hint": "",
                "training_state_hint": "best",
                "architecture_status": "VERIFIED_EXPLICIT",
                "relative_path": f"checkpoints/{arch}/{i}.pth",
                "linked_asset_count": "1" if i < 2 else "0",
            })

    selected = select_balanced_states(states)
    assert sum(r["architecture"] == "PraNet" for r in selected) == 3
    assert sum(r["architecture"] == "DeepLabV3-R50" for r in selected) == 3

    m = segformer_manifest()
    assert len(m) == 3
    assert [r["seed"] for r in m] == list(NEW_SEEDS)

    fp = future_panel(selected, m)
    assert len(fp) == BALANCED_TOTAL_STATES

    fake_upstream = {
        "upstream_valid": True,
    }
    fake_env = {
        "pytorch_available": True,
        "installed_segformer_implementation_paths": ["transformers"],
    }
    fake_protocol = {
        "source_interface_sufficient": True,
        "local_segformer_reference_files": [],
    }
    decision, criteria = choose_decision(
        fake_upstream,
        selected,
        fake_env,
        fake_protocol,
    )
    assert decision == DECISION_READY

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "code").mkdir()
        (root / "code" / "train_pranet.py").write_text(
            """
train_images = r'F:\\project\\source_train\\images'
train_masks = r'F:\\project\\source_train\\masks'
val_images = r'F:\\project\\source_val\\images'
val_masks = r'F:\\project\\source_val\\masks'
batch_size = 8
epochs = 40
optimizer = 'Adam'
model = 'PraNet'
""",
            encoding="utf-8",
        )
        data, code_rows, proto = scan_source_interface(root)
        assert proto["source_interface_sufficient"] is True
        assert len(data) >= 1
        assert len(code_rows) >= 1

    print("FROZEN_ARCHITECTURE_TEST_PASS")
    print("BALANCED_PANEL_SELECTION_TEST_PASS")
    print("SEGFORMER_MANIFEST_TEST_PASS")
    print("SOURCE_INTERFACE_SCAN_TEST_PASS")
    print("DECISION_TEST_PASS")
    print("SELF_TEST_PASS")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    root = args.root
    output_dir = args.output_dir

    if not root.exists():
        raise FileNotFoundError(root)

    try:
        output_dir.resolve().relative_to(root.resolve())
    except Exception:
        raise RuntimeError(
            f"Output directory must remain under {root}: {output_dir}"
        )

    if str(output_dir).lower().startswith("c:\\"):
        raise RuntimeError("C-drive output is forbidden.")

    if output_dir.exists():
        raise FileExistsError(
            f"Q1-R01C output already exists: {output_dir}"
        )

    build_dir = Path(str(output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(
            f"Partial Q1-R01C build exists: {build_dir}. "
            "Remove only this __building directory if a prior technical run failed."
        )
    build_dir.mkdir(parents=True, exist_ok=False)

    print("===== Q1-R01C THIRD-ARCHITECTURE SELECTION & TRAINING PLAN =====")
    print(f"Script version: {VERSION}")
    print(f"Build: {BUILD}")
    print(f"Selected third family: {THIRD_ARCHITECTURE}")
    print(f"Selected variant: {THIRD_VARIANT}")
    print(f"Frozen new seeds: {NEW_SEEDS}")
    print("Training=NO")
    print("Inference=NO")
    print("Checkpoint loading=NO")
    print("Image opening=NO")
    print()

    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    protocol_sha = file_sha256(PROTOCOL)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Protocol SHA mismatch: "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={protocol_sha}"
        )

    states, arch_rows, gap, upstream = validate_upstream(root)
    selected = select_balanced_states(states)

    # Ensure selected checkpoint paths still exist.
    for r in selected:
        p = root / r["relative_path"]
        if not p.exists():
            raise RuntimeError(f"Selected checkpoint missing: {p}")
        if file_sha256(p) != r["sha256"]:
            raise RuntimeError(f"Selected checkpoint SHA changed: {p}")

    env = environment_readiness()
    data_candidates, code_candidates, source_protocol = scan_source_interface(root)

    # Local SegFormer implementation may exist even without installed package.
    local_segformer_files = source_protocol["local_segformer_reference_files"]

    manifest = segformer_manifest()
    future = future_panel(selected, manifest)

    decision, criteria = choose_decision(
        upstream,
        selected,
        env,
        source_protocol,
    )

    gap_analysis = {
        "decision": decision,
        "criteria": criteria,
        "third_architecture": THIRD_ARCHITECTURE,
        "third_variant": THIRD_VARIANT,
        "frozen_new_seeds": list(NEW_SEEDS),
        "future_balanced_panel_target": {
            "PraNet": 3,
            "DeepLabV3-R50": 3,
            "SegFormer-B0": 3,
            "total": 9,
        },
        "source_interface_gap": (
            None
            if criteria["source_interface_sufficient"]
            else "Source train/mask/validation interface evidence is incomplete."
        ),
        "implementation_gap": (
            None
            if criteria["segformer_implementation_available"]
            else "No local/installed SegFormer implementation path detected."
        ),
    }

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)

    upstream_path = build_dir / "upstream_panel_reproduction.json"
    write_json(upstream_path, upstream)

    selected_path = build_dir / "selected_existing_balanced_panel.csv"
    write_csv(
        selected_path,
        selected,
        [
            "architecture",
            "panel_slot",
            "relative_path",
            "sha256",
            "seed_hint",
            "fold_hint",
            "epoch_hint",
            "training_state_hint",
            "architecture_status",
            "reusability",
            "linked_asset_count",
        ],
    )

    selection_path = build_dir / "third_architecture_selection.json"
    write_json(
        selection_path,
        {
            "architecture_family": THIRD_ARCHITECTURE,
            "variant": THIRD_VARIANT,
            "encoder": THIRD_ENCODER,
            "selection_status": "FROZEN_PRETRAINING",
            "selection_basis": [
                "Transformer family increases architecture diversity versus existing CNN-heavy panel.",
                "Hierarchical multiscale Transformer encoder.",
                "Lightweight All-MLP decoder.",
                "B0 is the lightweight SegFormer scale.",
                "Selected for diversity/replicability, not target performance.",
            ],
            "alternatives_not_selected": {
                "TransUNet": "Hybrid CNN/Transformer; less clean family separation.",
                "SwinUNet": "Valid future fourth-family candidate; more complex first replication target.",
            },
            "new_seeds": list(NEW_SEEDS),
        },
    )

    env_path = build_dir / "python_environment_readiness.json"
    env_payload = {
        **env,
        "local_segformer_reference_files": local_segformer_files,
    }
    write_json(env_path, env_payload)

    data_path = build_dir / "source_data_interface_candidates.csv"
    write_csv(
        data_path,
        data_candidates,
        [
            "relative_path",
            "source_train_hits",
            "source_mask_hits",
            "validation_hits",
            "generic_mask_hits",
            "path_like_evidence",
        ],
    )

    code_path = build_dir / "training_code_reference_candidates.csv"
    write_csv(
        code_path,
        code_candidates,
        [
            "relative_path",
            "architecture_hits",
            "training_setting_hits",
            "path_like_evidence",
        ],
    )

    source_protocol_path = build_dir / "source_protocol_evidence.json"
    write_json(source_protocol_path, source_protocol)

    manifest_path = build_dir / "segformer_three_state_training_manifest.csv"
    write_csv(
        manifest_path,
        manifest,
        [
            "architecture",
            "variant",
            "encoder",
            "state_slot",
            "seed",
            "training_data",
            "validation_data",
            "target_data_allowed",
            "utility_labels_allowed",
            "model_size_sweep",
            "checkpoint_selection",
            "status",
        ],
    )

    future_path = build_dir / "future_balanced_panel_plan.csv"
    write_csv(
        future_path,
        future,
        [
            "panel_order",
            "architecture",
            "state_slot",
            "origin",
            "seed",
            "checkpoint_path",
            "checkpoint_sha256",
            "status",
        ],
    )

    gap_path = build_dir / "gap_analysis.json"
    write_json(gap_path, gap_analysis)

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    # Integrity.
    selected_counts = defaultdict(int)
    for r in selected:
        selected_counts[r["architecture"]] += 1

    integrity = {
        "q1_r00_lock_verified": upstream["q1_r00_lock_sha256"] == EXPECTED_Q1_R00_LOCK_SHA256,
        "q1_r00_decision_verified": upstream["q1_r00_decision"] == EXPECTED_Q1_R00_DECISION,
        "pranet_checkpoint_backed": "PraNet" in upstream["checkpoint_backed_architectures"],
        "deeplab_checkpoint_backed": "DeepLabV3-R50" in upstream["checkpoint_backed_architectures"],
        "selected_pranet_at_least_2": selected_counts["PraNet"] >= 2,
        "selected_deeplab_at_least_2": selected_counts["DeepLabV3-R50"] >= 2,
        "selected_checkpoint_files_exist_and_match_sha": True,
        "third_architecture_frozen_segformer": THIRD_ARCHITECTURE == "SegFormer",
        "third_variant_frozen_b0": THIRD_VARIANT == "SegFormer-B0",
        "new_seeds_exact": NEW_SEEDS == (20260820, 20260821, 20260822),
        "balanced_panel_target_3_3_3": len(future) == BALANCED_TOTAL_STATES,
        "checkpoint_not_loaded": True,
        "image_not_opened": True,
        "training_not_run": True,
        "inference_not_run": True,
        "project_code_not_imported": True,
        "target_dataset_not_opened_for_training": True,
        "deterministic_selection_used": True,
    }

    if not all(integrity.values()):
        raise RuntimeError(
            "Q1-R01C integrity failure: "
            f"{[k for k,v in integrity.items() if not v]}"
        )

    summary = f"""===== Q1-R01C THIRD-ARCHITECTURE SELECTION & TRAINING PLAN =====
Script version: {VERSION}
Build: {BUILD}

Upstream:
  Q1-R00 lock verified=YES
  Q1-R00 decision={upstream['q1_r00_decision']}
  checkpoint-backed families={upstream['checkpoint_backed_architectures']}
  PraNet unique states={upstream['pranet_unique_states']}
  DeepLabV3-R50 unique states={upstream['deeplabv3_r50_unique_states']}

Balanced existing selection:
  PraNet selected={selected_counts['PraNet']}
  DeepLabV3-R50 selected={selected_counts['DeepLabV3-R50']}

Frozen third architecture:
  family={THIRD_ARCHITECTURE}
  variant={THIRD_VARIANT}
  encoder={THIRD_ENCODER}
  seeds={NEW_SEEDS}

Future main panel:
  PraNet=3
  DeepLabV3-R50=3
  SegFormer-B0=3
  total=9

Environment:
  torch={env['modules']['torch']}
  torchvision={env['modules']['torchvision']}
  transformers={env['modules']['transformers']}
  timm={env['modules']['timm']}
  mmseg={env['modules']['mmseg']}
  mmengine={env['modules']['mmengine']}
  installed SegFormer paths={env['installed_segformer_implementation_paths']}
  local SegFormer refs={len(local_segformer_files)}

Source-interface audit:
  text files scanned={source_protocol['text_files_scanned']}
  explicit source-train evidence={source_protocol['has_explicit_source_train_evidence']}
  mask/label evidence={source_protocol['has_mask_or_label_evidence']}
  validation evidence={source_protocol['has_validation_evidence']}
  existing PraNet/DeepLab training reference={source_protocol['has_existing_pranet_or_deeplab_training_reference']}
  source interface sufficient={source_protocol['source_interface_sufficient']}

Training=NO
Inference=NO
Checkpoint loading=NO
Image opening=NO

Decision: {decision}

Decision mapping:
  {DECISION_READY}
    -> Q1-R02 SegFormer-B0 Three-Seed Source Training

  {DECISION_DATA}
    -> resolve exact source train/mask/validation interface first

  {DECISION_SETUP}
    -> prepare fixed SegFormer-B0 implementation dependency first

  {DECISION_INVALID}
    -> stop; upstream Q1-R00 panel cannot be reproduced

[OK] Outputs: {output_dir}
"""
    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(summary, encoding="utf-8")

    artifact_paths = {
        "protocol_copy": protocol_copy,
        "upstream_panel_reproduction": upstream_path,
        "selected_existing_balanced_panel": selected_path,
        "third_architecture_selection": selection_path,
        "python_environment_readiness": env_path,
        "source_data_interface_candidates": data_path,
        "training_code_reference_candidates": code_path,
        "source_protocol_evidence": source_protocol_path,
        "segformer_three_state_training_manifest": manifest_path,
        "future_balanced_panel_plan": future_path,
        "gap_analysis": gap_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "q1_r00_lock_sha256": EXPECTED_Q1_R00_LOCK_SHA256,
        "q1_r00_decision": EXPECTED_Q1_R00_DECISION,
        "third_architecture": THIRD_ARCHITECTURE,
        "third_variant": THIRD_VARIANT,
        "new_seeds": list(NEW_SEEDS),
        "balanced_panel_target": {
            "PraNet": 3,
            "DeepLabV3-R50": 3,
            "SegFormer-B0": 3,
        },
        "integrity": integrity,
        "artifacts": {
            name: {
                "filename": p.name,
                "sha256": file_sha256(p),
            }
            for name, p in artifact_paths.items()
        },
        "decision": decision,
    }

    lock_path = build_dir / "Q1_R01C_THIRD_ARCHITECTURE_PLAN_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = file_sha256(lock_path)

    for name, p in artifact_paths.items():
        if file_sha256(p) != lock["artifacts"][name]["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(output_dir)

    print()
    print((output_dir / "run_log.txt").read_text(encoding="utf-8"))
    print(
        "Q1-R01C LOCK:",
        output_dir / "Q1_R01C_THIRD_ARCHITECTURE_PLAN_LOCK.json",
    )
    print("Q1-R01C LOCK SHA256:", lock_sha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Q1-R01C: freeze SegFormer-B0 as the third architecture and audit "
            "readiness for a balanced three-family model panel."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
