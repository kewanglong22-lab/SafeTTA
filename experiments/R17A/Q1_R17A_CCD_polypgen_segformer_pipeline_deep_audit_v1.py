#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_segformer_pipeline_deep_audit_v1.py

SafeTTA R17A — deeper read-only audit after the SegFormer hard-mask parity
diagnostic returned:
    POTENTIAL_PIPELINE_MISMATCH_REQUIRES_DEEPER_AUDIT

Target:
    seed=20260821
    row=1136
    sample=polypgen_static_1136

READ-ONLY / NO LEAKAGE
----------------------
Training: NO
Inference: NO
Target GT loading: NO
HARM/outcome loading: NO
Deletion/move: NO
Parameter/preprocessing search: NO

Purpose
-------
Do NOT try multiple preprocessing variants to "find one that matches" the
historical frozen mask. Instead, recover the exact historical inference
contract and environment evidence needed to explain/correct the mismatch.

This audit:
1) SHA-validates the exact historical scripts.
2) Extracts the previously omitted SegFormer helper functions:
     - segformer_tensor
     - configure_segformer_tent
     - segformer_source_mode
     - logit_to_mask
     - pack_mask
     - target/manifest construction helpers used by PolypGen lock
3) Extracts relevant constants/imports (IMAGE_SIZE, HF_CACHE, processor/model).
4) Audits current Python package versions.
5) Searches frozen release/environment files for historical package versions.
6) Audits the local Hugging Face `nvidia/mit-b0` cache/ref/snapshot metadata
   without downloading anything.
7) Audits PolypGen lock JSON/CSV metadata for environment / preprocessing /
   runner lineage fields.
8) Writes a compact decision report for the next runner fix.

No model forward pass is performed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "2026-09-09-Q1-R17A-SEGF-PIPELINE-DEEP-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_segformer_pipeline_deep_audit_v1")

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

R10_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")
R10_SHA = "4cae02c30e82781c6e7d3b74b6a312974789213a9cc59a44ca99bbfb94405708"

LOCK_ROOT_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
)
STATE_LOCK_REL = LOCK_ROOT_REL / Path(
    r"state_predictions\segformer_b0_seed20260821_state_lock.json"
)
STATE_INDEX_REL = LOCK_ROOT_REL / Path(
    r"state_predictions\segformer_b0_seed20260821_index.csv"
)

TARGET_FUNCTION_NAMES = {
    str(R03_REL).replace("\\", "/"): [
        "segformer_tensor",
        "configure_segformer_tent",
        "segformer_source_mode",
        "segformer_logits_and_z",
        "build_segformer_context",
        "load_segformer_state",
    ],
    str(R05_REL).replace("\\", "/"): [
        "logit_to_mask",
        "pack_mask",
        "run_segformer_state",
    ],
}

# R10 helper names differ by version. We extract exact functions whose body or
# name references target/manifest/image loading or SegFormer.
R10_NAME_TOKENS = (
    "target", "manifest", "image", "polypgen", "state", "prediction", "load"
)

PACKAGE_NAMES = [
    "torch",
    "torchvision",
    "transformers",
    "Pillow",
    "numpy",
    "pandas",
    "safetensors",
    "huggingface-hub",
]

ENV_FILE_PATTERNS = [
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
    "environment*.txt",
    "*freeze*.txt",
    "*lock*.txt",
    "*summary*.json",
]

ENV_ROOTS = [
    Path("release/SafeTTA_clean_test_v102"),
    Path("release/public_v1_fix4"),
    Path("release/SafeTTA_repro_release_candidate_v1"),
    Path("provenance"),
]

VERSION_TOKENS = (
    "torch", "torchvision", "transformers", "pillow",
    "numpy", "pandas", "cuda", "cudnn", "python",
    "huggingface", "safetensors",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\npath={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def source_segment(lines: List[str], node: ast.AST) -> str:
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1:end])


def extract_exact_functions(path: Path, required_names: List[str]) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))

    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in required_names:
                found[node.name] = {
                    "function": node.name,
                    "start_line": int(node.lineno),
                    "end_line": int(getattr(node, "end_lineno", node.lineno)),
                    "source": source_segment(lines, node),
                }

    missing = sorted(set(required_names) - set(found))
    if missing:
        raise RuntimeError(f"{path}: missing expected functions {missing}")

    return [found[n] for n in required_names]


def extract_r10_relevant_functions(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = source_segment(lines, node)
        low = f"{node.name}\n{body}".lower()

        if (
            any(tok in node.name.lower() for tok in R10_NAME_TOKENS)
            and any(tok in low for tok in [
                "image_path", "image_raw_sha256", "sample_id",
                "segformer", "manifest", "target",
            ])
        ):
            rows.append({
                "function": node.name,
                "start_line": int(node.lineno),
                "end_line": int(getattr(node, "end_lineno", node.lineno)),
                "source": body,
            })

    rows.sort(key=lambda x: x["start_line"])
    return rows


def extract_assignments_and_imports(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    rows = []

    wanted_names = {
        "IMAGE_SIZE", "HF_CACHE", "SEGFORMER_CHECKPOINTS",
        "SEGFORMER_SEEDS", "EXPECTED_CASES", "PACKED_BYTES",
        "DATA_ROOT", "OUTPUT_ROOT", "TENT_LR", "TENT_WEIGHT_DECAY",
    }

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rows.append({
                "kind": "IMPORT",
                "name": "",
                "line": int(node.lineno),
                "source": source_segment(lines, node),
            })
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = []
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
            for name in names:
                if name in wanted_names:
                    rows.append({
                        "kind": "ASSIGNMENT",
                        "name": name,
                        "line": int(node.lineno),
                        "source": source_segment(lines, node),
                    })
    return rows


def current_environment() -> dict:
    versions = {}
    for pkg in PACKAGE_NAMES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"

    try:
        import torch
        torch_info = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available() else None
            ),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_allow_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32)
                if torch.cuda.is_available() else None
            ),
            "cudnn_allow_tf32": (
                bool(torch.backends.cudnn.allow_tf32)
                if torch.cuda.is_available() else None
            ),
        }
    except Exception as e:
        torch_info = {"error": f"{type(e).__name__}:{e}"}

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "torch_runtime": torch_info,
    }


def scan_environment_files(root: Path) -> List[dict]:
    rows = []
    seen = set()

    for rel_root in ENV_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        candidates = []
        for pat in ENV_FILE_PATTERNS:
            candidates.extend(base.rglob(pat))

        for p in sorted(set(candidates)):
            if not p.is_file():
                continue
            try:
                rp = rel(p, root)
            except ValueError:
                continue
            if rp in seen:
                continue
            seen.add(rp)

            try:
                if p.stat().st_size > 10 * 1024 * 1024:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            matched_lines = []
            for i, line in enumerate(txt.splitlines(), 1):
                low = line.lower()
                if any(tok in low for tok in VERSION_TOKENS):
                    matched_lines.append(f"L{i}:{line.strip()[:500]}")

            if matched_lines:
                rows.append({
                    "path": rp,
                    "sha256": sha256_file(p),
                    "matched_line_count": len(matched_lines),
                    "examples": " | ".join(matched_lines[:80]),
                })
    return rows


def find_hf_cache_from_script(root: Path, r03_path: Path) -> Optional[Path]:
    text = r03_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(r03_path))

    # Simple best-effort parser for HF_CACHE = ROOT / "cache" / ...
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "HF_CACHE" for t in node.targets):
                # We do not eval arbitrary code. First try importing the SHA-validated
                # historical script only to read the constant. If that fails, return None.
                break

    try:
        import importlib.util
        name = "q1_r03_deep_audit_constant_only"
        spec = importlib.util.spec_from_file_location(name, r03_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        hf = getattr(mod, "HF_CACHE", None)
        if hf is None:
            return None
        return Path(hf)
    except Exception:
        return None


def scan_hf_cache(cache_root: Optional[Path], project_root: Path) -> dict:
    if cache_root is None:
        return {"status": "HF_CACHE_UNRESOLVED"}

    info = {
        "status": "OK",
        "hf_cache": str(cache_root),
        "exists": cache_root.exists(),
        "model_candidates": [],
    }
    if not cache_root.exists():
        return info

    # Common HF hub layout:
    #   hub/models--nvidia--mit-b0/{refs,snapshots,blobs}
    candidates = [
        cache_root / "hub" / "models--nvidia--mit-b0",
        cache_root / "models--nvidia--mit-b0",
    ]
    for base in candidates:
        if not base.exists():
            continue

        entry = {
            "base": str(base),
            "refs": {},
            "snapshots": [],
        }

        refs = base / "refs"
        if refs.exists():
            for p in refs.rglob("*"):
                if p.is_file():
                    try:
                        entry["refs"][str(p.relative_to(base)).replace("\\", "/")] = (
                            p.read_text(encoding="utf-8", errors="ignore").strip()
                        )
                    except Exception:
                        pass

        snaps = base / "snapshots"
        if snaps.exists():
            for s in sorted(p for p in snaps.iterdir() if p.is_dir()):
                files = []
                for f in sorted(s.rglob("*")):
                    if not f.is_file():
                        continue
                    relf = str(f.relative_to(s)).replace("\\", "/")
                    if relf.lower() in {
                        "preprocessor_config.json",
                        "config.json",
                    }:
                        try:
                            txt = f.read_text(encoding="utf-8", errors="ignore")
                            files.append({
                                "path": relf,
                                "sha256": sha256_file(f),
                                "content": txt[:20000],
                            })
                        except Exception:
                            pass
                entry["snapshots"].append({
                    "snapshot": s.name,
                    "selected_files": files,
                })

        info["model_candidates"].append(entry)

    return info


def flatten_json(obj, prefix="") -> List[tuple]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten_json(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def audit_lock_metadata(root: Path) -> List[dict]:
    rows = []
    lock_root = root / LOCK_ROOT_REL
    if not lock_root.exists():
        raise FileNotFoundError(lock_root)

    candidates = list(lock_root.rglob("*.json")) + list(lock_root.rglob("*.csv"))
    for p in sorted(candidates):
        low_name = p.name.lower()
        # Focus on lock/index/manifest/preflight/audit metadata.
        if not any(tok in low_name for tok in [
            "lock", "audit", "manifest", "preflight", "index"
        ]):
            continue

        try:
            if p.stat().st_size > 20 * 1024 * 1024:
                continue
        except OSError:
            continue

        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            matched = []
            for key, val in flatten_json(obj):
                low = f"{key} {val}".lower()
                if any(tok in low for tok in [
                    "segformer", "processor", "preprocess", "image_size",
                    "resize", "interpol", "checkpoint", "sha256",
                    "torch", "transformers", "cuda", "runner",
                    "historical", "lineage", "source"
                ]):
                    matched.append(f"{key}={str(val)[:500]}")
            if matched:
                rows.append({
                    "path": rel(p, root),
                    "sha256": sha256_file(p),
                    "matched_count": len(matched),
                    "examples": " | ".join(matched[:100]),
                })

        elif p.suffix.lower() == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = [str(c) for c in df.columns]
            selected = [
                c for c in cols
                if any(tok in c.lower() for tok in [
                    "model", "state", "seed", "checkpoint", "sha",
                    "image", "source", "runner", "artifact"
                ])
            ]
            if selected:
                rows.append({
                    "path": rel(p, root),
                    "sha256": sha256_file(p),
                    "matched_count": len(selected),
                    "examples": "columns=" + "|".join(selected[:100]),
                })
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A SEGF PIPELINE DEEP AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Training: NO")
    print("Inference: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Preprocessing/model variant search: NO")
    print()

    paths = {
        "R03": root / R03_REL,
        "R05": root / R05_REL,
        "R10": root / R10_REL,
    }

    print("[1/7] Validate exact historical scripts...")
    validate_sha(paths["R03"], R03_SHA, "R03")
    validate_sha(paths["R05"], R05_SHA, "R05")
    validate_sha(paths["R10"], R10_SHA, "R10")

    print("[2/7] Extract omitted SegFormer/preprocessing helpers...")
    function_rows = []
    for relp, names in TARGET_FUNCTION_NAMES.items():
        p = root / Path(relp.replace("/", "\\"))
        for row in extract_exact_functions(p, names):
            function_rows.append({
                "path": relp,
                **row,
            })

    for row in extract_r10_relevant_functions(paths["R10"]):
        function_rows.append({
            "path": str(R10_REL).replace("\\", "/"),
            **row,
        })

    assignment_rows = []
    for label, p in paths.items():
        for row in extract_assignments_and_imports(p):
            assignment_rows.append({
                "path": rel(p, root),
                **row,
            })

    print("[3/7] Capture current environment...")
    env_current = current_environment()

    print("[4/7] Search frozen historical environment evidence...")
    env_files = scan_environment_files(root)

    print("[5/7] Audit local HF mit-b0 cache/ref/snapshot...")
    hf_cache = find_hf_cache_from_script(root, paths["R03"])
    hf_info = scan_hf_cache(hf_cache, root)

    print("[6/7] Audit PolypGen lock/preprocessing lineage metadata...")
    lock_rows = audit_lock_metadata(root)

    print("[7/7] Write decision report...")
    write_csv(out_dir / "R17A_SEGF_EXACT_FUNCTIONS.csv", function_rows)
    write_csv(out_dir / "R17A_SEGF_CONSTANTS_IMPORTS.csv", assignment_rows)
    write_csv(out_dir / "R17A_SEGF_HISTORICAL_ENV_EVIDENCE.csv", env_files)
    write_csv(out_dir / "R17A_SEGF_POLYPGEN_LOCK_METADATA.csv", lock_rows)

    (out_dir / "R17A_SEGF_CURRENT_ENVIRONMENT.json").write_text(
        json.dumps(env_current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "R17A_SEGF_HF_CACHE_AUDIT.json").write_text(
        json.dumps(hf_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Heuristic decision only; no inference is run here.
    transformer_hist_mentions = []
    torch_hist_mentions = []
    for r in env_files:
        ex = r["examples"].lower()
        if "transformers" in ex:
            transformer_hist_mentions.append(r["path"])
        if "torch" in ex:
            torch_hist_mentions.append(r["path"])

    hf_snapshots = 0
    if hf_info.get("status") == "OK":
        for m in hf_info.get("model_candidates", []):
            hf_snapshots += len(m.get("snapshots", []))

    if transformer_hist_mentions and torch_hist_mentions and hf_snapshots > 0:
        decision = "READY_TO_COMPARE_CURRENT_VS_FROZEN_ENVIRONMENT_AND_BUILD_CORRECTED_RUNNER"
    elif transformer_hist_mentions or torch_hist_mentions:
        decision = "PARTIAL_ENVIRONMENT_EVIDENCE_REQUIRES_MANUAL_CONTRACT_REVIEW"
    else:
        decision = "NO_FROZEN_ENV_VERSION_EVIDENCE_FOUND_USE_EXACT_CODE_AND_CURRENT_ENV_WITH_DISCLOSED_PARITY_LIMIT"

    lines = [
        "===== R17A SEGF PIPELINE DEEP AUDIT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "INFORMATION BOUNDARY:",
        "Training: NO",
        "Inference: NO",
        "Target GT loading: NO",
        "HARM/outcome loading: NO",
        "Preprocessing/model variant search: NO",
        "",
        "CURRENT ENVIRONMENT:",
        json.dumps(env_current, indent=2, ensure_ascii=False),
        "",
        "HF CACHE:",
        json.dumps(hf_info, indent=2, ensure_ascii=False),
        "",
        "EXTRACTED EXACT FUNCTIONS:",
    ]

    for r in function_rows:
        lines.append(
            f"\n--- {r['path']} :: {r['function']} "
            f"[L{r['start_line']}-L{r['end_line']}] ---\n{r['source']}"
        )

    lines += ["", "CONSTANTS / IMPORTS:"]
    for r in assignment_rows:
        lines.append(
            f"{r['path']} L{r['line']} {r['kind']} {r['name']} :: {r['source']}"
        )

    lines += ["", "FROZEN ENVIRONMENT EVIDENCE:"]
    for r in env_files[:80]:
        lines.append(
            f"{r['path']} :: {r['examples'][:4000]}"
        )

    lines += ["", "POLYPGEN LOCK / LINEAGE METADATA:"]
    for r in lock_rows[:80]:
        lines.append(
            f"{r['path']} :: {r['examples'][:4000]}"
        )

    lines += [
        "",
        f"DECISION={decision}",
        "",
        "NEXT RULE:",
        "- Do not rerun the full 9-state CCD runner yet.",
        "- First compare current package/HF processor/config evidence against the",
        "  frozen historical environment/lock metadata identified here.",
        "- If an exact historical environment mismatch is identified, recreate",
        "  that environment/processor contract and re-test ONLY seed20260821 row1136.",
        "- Do not tune preprocessing variants against the frozen mask.",
    ]

    report = out_dir / "R17A_SEGF_PIPELINE_DEEP_AUDIT_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"DECISION={decision}")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
