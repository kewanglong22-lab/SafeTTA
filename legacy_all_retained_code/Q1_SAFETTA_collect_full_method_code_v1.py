#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Q1_SAFETTA_collect_full_method_code_v1.py

Non-destructive collector/auditor for preparing a reviewer-facing "full"
SafeTTA code repository from the retained author project.

Default author root:
    F:\MEDSEG_SAFETTA

The goal is NOT to dump every exploratory script. It collects:
  A) scripts explicitly referenced by the final reproducibility/provenance chain;
  B) exact known canonical SafeTTA method scripts;
  C) retained final scripts matching method/action/preprocessing/evaluation roles;
  D) a complete inventory of all retained Python scripts for human audit.

It writes a staging collection only. It does not modify the current GitHub repo,
does not delete anything, and does not rerun experiments.

Outputs:
  F:\MEDSEG_SAFETTA\release\full_method_code_collection_v1\
      selected_code\
      FULL_METHOD_SELECTED_CODE.csv
      FULL_METHOD_ALL_PYTHON_INVENTORY.csv
      FULL_METHOD_ROLE_CANDIDATES.csv
      FULL_METHOD_MISSING_CANONICAL.csv
      FULL_METHOD_CODE_AUDIT.txt

Run:
    D:\anaconda\envs\rare26\python.exe -u ^
      F:\MEDSEG_SAFETTA\code\Q1_SAFETTA_collect_full_method_code_v1.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-06-Q1-SAFETTA-COLLECT-FULL-METHOD-CODE-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
DEFAULT_OUT = Path(r"F:\MEDSEG_SAFETTA\release\full_method_code_collection_v1")

EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "release", "github_upload", "full_method_code_collection_v1",
}

# Exact/known canonical pieces from the frozen manuscript + audits.
# Some filenames are exact; some use patterns because the final audit records
# the stage/hash but the current public repo may not yet include the source file.
CANONICAL_EXACT_BASENAMES = {
    "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py":
        "prediction-conditioned DINOv2 representation",
    "Q1_R10L0_final_source_safety_estimator_lock_fix3.py":
        "frozen colonoscopy PCA/head/threshold estimator",
    "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2.py":
        "source ablation paired bootstrap",
    "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1.py":
        "selective adaptation utility",
    "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1.py":
        "PolypGen locked external evaluation",
    "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1.py":
        "SUN-SEG dual-action evaluation",
    "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1.py":
        "PROMISE12 locked-policy evaluation",
    "Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1.py":
        "PROMISE12 matched-coverage utility",
    "Q1_SAFETTA_harm_margin_sensitivity_v1.py":
        "harm-margin sensitivity",
    "Q1_SAFETTA_public_paper_stat_replay_v3_fix1.py":
        "public 44-anchor replay",
}

# Patterns for method components whose exact retained filename may vary.
ROLE_PATTERNS = {
    "representation": [
        r"R10K2A.*dinov2.*representation",
        r"mask.*condition.*dinov2",
        r"prediction.*condition.*representation",
    ],
    "morphology": [
        r"R10J1.*morphology",
        r"mask.*morphology.*feature",
        r"boundary.*transition",
    ],
    "source_safety_estimator": [
        r"R10L0.*source.*safety.*estimator",
        r"safety.*estimator.*lock",
    ],
    "tent_colonoscopy": [
        r"A1.*TENT",
        r"TENT.*1STEP",
        r"tent.*colono",
        r"neopolyp.*tent",
    ],
    "pl_conf90_colonoscopy": [
        r"PL.*CONF90",
        r"CONF90.*pseudo",
        r"pseudo.*label.*90",
    ],
    "polypgen_preprocessing_inference": [
        r"R10L1.*polypgen",
        r"R10L3A.*polypgen",
        r"polypgen.*prediction",
        r"polypgen.*manifest",
    ],
    "sun_preprocessing_inference": [
        r"R14B4.*sun",
        r"R14C1.*sun",
        r"R14C2.*sun",
        r"sunseg.*prediction",
        r"sunseg.*manifest",
    ],
    "mri_preprocessing": [
        r"CM[123].*mri",
        r"prostate158.*preprocess",
        r"promise12.*preprocess",
        r"resample.*352",
    ],
    "mri_tent": [
        r"CM4A.*tent",
        r"mri.*tent",
        r"prostate.*tent",
    ],
    "mri_safety_scoring": [
        r"CM4B.*safety",
        r"CM4C.*safety",
        r"CM4D.*safety",
        r"prostate.*safety",
    ],
    "promise_pre_gt": [
        r"CM5A.*promise",
        r"CM5B.*promise",
        r"promise12.*GT.*FREE",
        r"operating.*point.*transport",
    ],
    "promise_evaluation": [
        r"CM6.*promise",
        r"CM7A.*promise",
        r"CM7B.*promise",
    ],
    "metrics_dice": [
        r"dice",
        r"metric",
        r"evaluation",
    ],
    "runtime": [
        r"R15D1.*runtime",
        r"runtime.*overhead",
    ],
}

# File-content signatures for discovering scripts even when filenames are opaque.
CONTENT_SIGNATURES = {
    "dinov2_representation": [
        "facebook/dinov2-base",
        "pixel_values",
        "last_hidden_state",
    ],
    "mask_22x22_mapping": [
        "22",
        "16",
        "mask",
    ],
    "safety_lr": [
        "LogisticRegression",
        "SimpleImputer",
        "StandardScaler",
    ],
    "tent_action": [
        "softplus",
        "optimizer.step",
        "Adam",
    ],
    "pl_conf90_action": [
        "0.90",
        "pseudo",
        "confidence",
    ],
    "patient_bootstrap": [
        "default_rng",
        "bootstrap",
        "patient",
    ],
}

PROVENANCE_FILENAMES = {
    "PAPER_CHAIN_RESOLVED_V3.json",
    "canonical_numeric_anchor_provenance.json",
    "all_code_sha256.csv",
    "repro_inventory.csv",
}

PY_REF_RE = re.compile(r"(?P<name>[A-Za-z0-9_.+\-]*Q1[A-Za-z0-9_.+\-]*\.py)", re.I)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_excluded(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except Exception:
        return True
    for part in rel.parts[:-1]:
        if part.lower() in {x.lower() for x in EXCLUDE_DIR_NAMES}:
            return True
    return False


def collect_python_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*.py"):
        if p.is_file() and not is_excluded(p, root):
            files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def safe_read_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    if path.stat().st_size > max_bytes:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
        return raw.decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def syntax_status(path: Path) -> Tuple[bool, str]:
    try:
        src = safe_read_text(path, max_bytes=16 * 1024 * 1024)
        ast.parse(src)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def find_provenance_files(root: Path) -> List[Path]:
    found = []
    for name in PROVENANCE_FILENAMES:
        found.extend(
            p for p in root.rglob(name)
            if p.is_file() and not is_excluded(p, root)
        )
    return sorted(set(found), key=lambda x: str(x).lower())


def extract_py_refs_from_text(text: str) -> Set[str]:
    return {m.group("name") for m in PY_REF_RE.finditer(text)}


def extract_provenance_refs(paths: Sequence[Path]) -> Set[str]:
    refs: Set[str] = set()
    for p in paths:
        try:
            refs |= extract_py_refs_from_text(safe_read_text(p, 32 * 1024 * 1024))
        except Exception:
            continue
    return refs


def basename_index(files: Sequence[Path]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        idx[p.name.lower()].append(p)
    return idx


def role_matches(path: Path, text: Optional[str] = None) -> Set[str]:
    name = path.name.lower()
    roles = set()
    for role, pats in ROLE_PATTERNS.items():
        for pat in pats:
            if re.search(pat, name, flags=re.I):
                roles.add(role)
                break

    if text is not None:
        low = text.lower()
        for role, sigs in CONTENT_SIGNATURES.items():
            hits = sum(1 for s in sigs if s.lower() in low)
            if hits == len(sigs):
                roles.add(role)
    return roles


def copy_selected(
    root: Path,
    selected: Sequence[Tuple[Path, Set[str], Set[str]]],
    out_dir: Path,
) -> None:
    selected_root = out_dir / "selected_code"
    selected_root.mkdir(parents=True, exist_ok=True)

    iterator = list(selected)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            desc="Copy selected SafeTTA code",
            unit="file",
            dynamic_ncols=True,
        )

    for p, _, _ in iterator:
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = Path("unresolved") / p.name
        dst = selected_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    root = args.root
    out_dir = args.output

    if not root.exists():
        raise FileNotFoundError(root)
    if out_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=False)

    print("===== SAFETTA FULL METHOD CODE COLLECTION =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Output:", out_dir)
    print("Experiment rerun: NO")
    print("Current GitHub repo modified: NO")
    print()

    py_files = collect_python_files(root)
    idx = basename_index(py_files)

    provenance_files = find_provenance_files(root)
    provenance_refs = extract_provenance_refs(provenance_files)
    provenance_ref_lc = {x.lower() for x in provenance_refs}

    all_rows: List[Dict[str, object]] = []
    selected_map: Dict[Path, Dict[str, Set[str]]] = {}
    role_rows: List[Dict[str, object]] = []

    iterator = py_files
    if tqdm is not None:
        iterator = tqdm(
            py_files,
            desc="Audit retained Python",
            unit="file",
            dynamic_ncols=True,
        )

    for p in iterator:
        text = safe_read_text(p, 4 * 1024 * 1024)
        roles = role_matches(p, text=text)
        reasons: Set[str] = set()

        if p.name in CANONICAL_EXACT_BASENAMES:
            reasons.add("known_canonical_exact")
            roles.add(CANONICAL_EXACT_BASENAMES[p.name])

        if p.name.lower() in provenance_ref_lc:
            reasons.add("referenced_by_final_provenance")

        if roles:
            reasons.add("method_role_match")

        ok, syntax_err = syntax_status(p)
        rel = str(p.relative_to(root)).replace("\\", "/")
        sha = sha256_file(p)

        all_rows.append({
            "relpath": rel,
            "basename": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": sha,
            "syntax_ok": ok,
            "syntax_error": syntax_err,
            "roles": ";".join(sorted(roles)),
            "selected": bool(reasons),
            "selection_reasons": ";".join(sorted(reasons)),
        })

        if roles:
            role_rows.append({
                "relpath": rel,
                "basename": p.name,
                "roles": ";".join(sorted(roles)),
                "size_bytes": p.stat().st_size,
                "sha256": sha,
                "syntax_ok": ok,
            })

        if reasons:
            selected_map[p] = {"roles": roles, "reasons": reasons}

    # Ensure exact provenance-referenced basenames are selected when resolvable.
    for ref in provenance_refs:
        candidates = idx.get(Path(ref).name.lower(), [])
        for p in candidates:
            item = selected_map.setdefault(p, {"roles": set(), "reasons": set()})
            item["reasons"].add("referenced_by_final_provenance")

    # Missing exact canonical basenames.
    missing_rows = []
    for basename, role in sorted(CANONICAL_EXACT_BASENAMES.items()):
        candidates = idx.get(basename.lower(), [])
        if not candidates:
            missing_rows.append({
                "canonical_basename": basename,
                "expected_role": role,
                "status": "MISSING",
            })

    selected: List[Tuple[Path, Set[str], Set[str]]] = []
    selected_rows: List[Dict[str, object]] = []
    for p in sorted(selected_map, key=lambda x: str(x).lower()):
        roles = selected_map[p]["roles"]
        reasons = selected_map[p]["reasons"]
        ok, syntax_err = syntax_status(p)
        selected.append((p, roles, reasons))
        selected_rows.append({
            "relpath": str(p.relative_to(root)).replace("\\", "/"),
            "basename": p.name,
            "roles": ";".join(sorted(roles)),
            "selection_reasons": ";".join(sorted(reasons)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "syntax_ok": ok,
            "syntax_error": syntax_err,
        })

    copy_selected(root, selected, out_dir)

    write_csv(
        out_dir / "FULL_METHOD_ALL_PYTHON_INVENTORY.csv",
        all_rows,
        [
            "relpath", "basename", "size_bytes", "sha256",
            "syntax_ok", "syntax_error", "roles",
            "selected", "selection_reasons",
        ],
    )
    write_csv(
        out_dir / "FULL_METHOD_ROLE_CANDIDATES.csv",
        role_rows,
        ["relpath", "basename", "roles", "size_bytes", "sha256", "syntax_ok"],
    )
    write_csv(
        out_dir / "FULL_METHOD_SELECTED_CODE.csv",
        selected_rows,
        [
            "relpath", "basename", "roles", "selection_reasons",
            "size_bytes", "sha256", "syntax_ok", "syntax_error",
        ],
    )
    write_csv(
        out_dir / "FULL_METHOD_MISSING_CANONICAL.csv",
        missing_rows,
        ["canonical_basename", "expected_role", "status"],
    )

    role_counts = defaultdict(int)
    for row in selected_rows:
        for r in str(row["roles"]).split(";"):
            if r:
                role_counts[r] += 1

    syntax_fail_selected = [r for r in selected_rows if not bool(r["syntax_ok"])]

    summary = [
        "===== SAFETTA FULL METHOD CODE COLLECTION SUMMARY =====",
        f"Retained Python files audited: {len(py_files)}",
        f"Final provenance files found: {len(provenance_files)}",
        f"Python basenames referenced by provenance: {len(provenance_refs)}",
        f"Selected final/method candidate scripts: {len(selected_rows)}",
        f"Selected syntax failures: {len(syntax_fail_selected)}",
        f"Missing exact canonical basenames: {len(missing_rows)}",
        "",
        "ROLE COUNTS:",
    ]
    for role, count in sorted(role_counts.items()):
        summary.append(f"  {role}: {count}")

    if missing_rows:
        summary += ["", "MISSING EXACT CANONICAL:"]
        for r in missing_rows:
            summary.append(
                f"  {r['canonical_basename']} :: {r['expected_role']}"
            )

    summary += [
        "",
        "GATE="
        + (
            "READY_FOR_FULL_METHOD_REPO_REVIEW"
            if not syntax_fail_selected
            else "BLOCKED_SELECTED_SYNTAX_FAILURE"
        ),
        "NEXT=REVIEW_SELECTED_MANIFEST_AND_BUILD_PUBLIC_METHOD_PACKAGE",
    ]

    (out_dir / "FULL_METHOD_CODE_AUDIT.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\n".join(summary))
    return 0 if not syntax_fail_selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
