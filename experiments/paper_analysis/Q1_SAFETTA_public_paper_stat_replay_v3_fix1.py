#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Q1_SAFETTA_numeric_replay_v1.py

SafeTTA deterministic paper-result replay harness.

Goal
----
Run the retained *final analysis scripts* into NEW output directories, compare
their regenerated numeric summaries against the frozen originals, and close the
paper's deterministic numeric replay gate.

Important scope
---------------
This is a paper-analysis replay, not a claim that every historical upstream
segmentation-training run can be recreated bit-for-bit.

Fresh replay is required for deterministic paper statistics:
- SOURCE safety threshold lock
- SOURCE core ablation + paired bootstrap
- PolypGen external evaluation
- SUN-SEG dual-action evaluation
- selective-utility/random-control analysis
- PROMISE12 locked GT-reveal evaluation
- PROMISE12 matched-random utility analysis
- HARM-margin sensitivity (paper-facing PolypGen + PROMISE12 only; fresh direct recomputation)

Prostate158 macro AUROCs are recomputed directly from the retained 10,659-row
CM4B SOURCE OOF score table.

Runtime is intentionally NOT required to match bit-for-bit because wall-clock
latency is hardware/noise dependent. The frozen R15D1 benchmark SHA/protocol is
audited separately.

Safety
------
- Frozen output directories are NEVER modified.
- Every executed stage must expose --output-dir or --out-dir.
- Replay output root must be fresh unless --resume is explicitly used.
- Project scripts are executed as subprocesses; they are not imported.
- Any stage failure blocks PASS_NUMERIC_REPLAY.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-06-Q1-SAFETTA-PUBLIC-PAPER-STAT-REPLAY-v3-fix1"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
NUMERIC_ATOL = 1e-12
MAX_COMPARE_BYTES = 80 * 1024 * 1024

CANONICAL_PROVENANCE = (
    "Q1_SAFETTA_numeric_provenance_canonical_v3",
    "canonical_numeric_anchor_provenance.json",
)
CHAIN_GATE = (
    "Q1_SAFETTA_exact_repro_numeric_provenance_v1",
    "PAPER_CHAIN_RESOLVED_V3.json",
)

RUNTIME_ANCHORS = {
    "RUNTIME_B1_TOTAL_MS",
    "RUNTIME_B1_THROUGHPUT",
    "RUNTIME_POST_DINO_MS",
}


@dataclass(frozen=True)
class ReplayStage:
    stage_id: str
    script_name: str
    frozen_output_dir: str
    anchor_ids: Tuple[str, ...]
    summary_name_tokens: Tuple[str, ...]
    expected_script_sha256: Optional[str] = None


STAGES: Tuple[ReplayStage, ...] = (
    ReplayStage(
        "R10L0",
        "Q1_R10L0_final_source_safety_estimator_lock_fix3.py",
        "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1",
        ("COLON_THRESHOLD",),
        ("threshold", "summary", "lock"),
        "ed828784d835b2fc77d93e3afb6865b382da7774dae6826add17ca8534cb2bb6",
    ),
    ReplayStage(
        "R15A1",
        "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2.py",
        "Q1_R15A1_core_ablation_case_clustered_paired_bootstrap_fix2_v1",
        (
            "SRC_FINAL_AUROC",
            "SRC_FINAL_AUPRC",
            "SRC_M2_AUROC",
            "SRC_IMAGECLS_AUROC",
            "SRC_FG_AUROC",
            "SRC_BG_AUROC",
            "SRC_FINAL_MINUS_M2_AUROC",
            "SRC_FINAL_MINUS_IMAGECLS_FPR",
        ),
        ("point", "summary", "delta", "bootstrap", "ci", "metric"),
    ),
    ReplayStage(
        "R15B1",
        "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1.py",
        "Q1_R15B1_selective_adaptation_risk_coverage_utility_fix1_v1",
        (
            "POLYPGEN_COVERAGE",
            "POLYPGEN_PREVENTED_HARM",
            "POLYPGEN_DEPLOYED_DICE",
            "POLYPGEN_RANDOM_DELTA_DICE",
            "SUN_TENT1_DEPLOYED_DICE",
            "SUN_PL_DEPLOYED_DICE",
            "SUN_TENT1_RANDOM_DELTA_DICE",
        ),
        ("utility", "summary", "bootstrap", "delta", "ci", "random"),
    ),
)

PROSTATE_ANCHORS = (
    "PROSTATE_M2_AUROC",
    "PROSTATE_CONDDINO_AUROC",
    "PROSTATE_FINAL_AUROC",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_python_syntax(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8", errors="replace"))


def inventory_sha_map(root: Path) -> Dict[str, str]:
    """
    Optional second provenance source. If chain resolver's all_code_sha256.csv
    exists, use it to verify the currently retained script bytes.
    """
    candidates = [
        root / "outputs" / "Q1_SAFETTA_repro_chain_resolver_v2" / "all_code_sha256.csv",
        root / "outputs" / "Q1_SAFETTA_repro_inventory_v1" / "repro_inventory.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        path_col = next((c for c in df.columns if c.lower() in {"relpath", "path"}), None)
        sha_col = next((c for c in df.columns if "sha256" in c.lower()), None)
        if path_col is None or sha_col is None:
            continue

        out = {}
        for _, row in df.iterrows():
            rel = str(row[path_col])
            sha = str(row[sha_col])
            if sha and sha != "nan":
                out[Path(rel).name] = sha.lower()
        if out:
            return out
    return {}



def detect_output_argument_static(script: Path) -> Optional[str]:
    """
    Detect --output-dir / --out-dir from source AST only.

    IMPORTANT: do NOT run ``python script.py --help`` here. A historical script
    that does not use argparse may ignore --help and execute its full workload,
    potentially touching its frozen output directory.
    """
    source = script.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    found: List[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value in {"--output-dir", "--out-dir"}:
                    found.append(arg.value)

    uniq = sorted(set(found))
    if not uniq:
        return None
    if len(uniq) > 1:
        raise RuntimeError(
            f"{script.name}: multiple directory-output CLI options detected: {uniq}"
        )
    return uniq[0]


OUTPUT_TARGET_TOKENS = (
    "out", "output", "save", "result", "run_dir", "work_dir", "artifact"
)


def _target_names(target: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_target_names(elt))
    elif isinstance(target, ast.Attribute):
        names.append(target.attr)
    return names


def _assignment_targets_and_value(node: ast.AST) -> Tuple[List[str], Optional[ast.AST]]:
    if isinstance(node, ast.Assign):
        names: List[str] = []
        for t in node.targets:
            names.extend(_target_names(t))
        return names, node.value
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target), node.value
    return [], None


def _is_outputish_target(names: Sequence[str]) -> bool:
    blob = " ".join(str(x).lower() for x in names)
    return any(tok in blob for tok in OUTPUT_TARGET_TOKENS)


def _line_col_to_abs(source: str, lineno: int, byte_col: int) -> int:
    """
    Convert AST's 1-based line + UTF-8 byte column to Python string index.
    """
    lines = source.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        raise IndexError((lineno, len(lines)))
    prefix = "".join(lines[: lineno - 1])
    line = lines[lineno - 1]
    raw = line.encode("utf-8")
    char_col = len(raw[:byte_col].decode("utf-8"))
    return len(prefix) + char_col


def build_static_output_patch_plan(
    script: Path,
    frozen_output_dir_name: str,
    replacement_output_dir_name: str,
) -> Dict[str, Any]:
    """
    Find string literals that contain the frozen output directory name and are
    assigned to output-like variables (OUT_DIR, OUTPUT_DIR, RESULT_DIR, etc.).

    The returned plan is read-only. No source is modified here.
    """
    source = script.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)

    candidates: List[Dict[str, Any]] = []
    other_literals: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target_names, value = _assignment_targets_and_value(node)
        if value is None:
            continue
        outputish = _is_outputish_target(target_names)

        for sub in ast.walk(value):
            if not (
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and frozen_output_dir_name in sub.value
            ):
                continue

            rec = {
                "target_names": target_names,
                "literal_value": sub.value,
                "lineno": sub.lineno,
                "col_offset": sub.col_offset,
                "end_lineno": sub.end_lineno,
                "end_col_offset": sub.end_col_offset,
            }
            if outputish:
                candidates.append(rec)
            else:
                other_literals.append(rec)

    safe = len(candidates) >= 1 and len(other_literals) == 0
    reason = (
        "SAFE_OUTPUT_LITERAL_PATCH"
        if safe
        else (
            "NO_OUTPUT_LITERAL_CANDIDATE"
            if not candidates
            else "FROZEN_OUTPUT_LITERAL_USED_OUTSIDE_OUTPUT_ASSIGNMENT"
        )
    )

    return {
        "safe": safe,
        "reason": reason,
        "frozen_output_dir_name": frozen_output_dir_name,
        "replacement_output_dir_name": replacement_output_dir_name,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "non_output_literal_count": len(other_literals),
        "non_output_literals": other_literals,
    }


def apply_static_output_patch(
    script: Path,
    plan: Dict[str, Any],
    patched_copy: Path,
) -> Dict[str, Any]:
    if not plan.get("safe"):
        raise RuntimeError(f"Unsafe output patch plan: {plan}")

    source = script.read_text(encoding="utf-8", errors="replace")
    old = str(plan["frozen_output_dir_name"])
    new = str(plan["replacement_output_dir_name"])

    patches: List[Tuple[int, int, str, str]] = []
    for rec in plan["candidates"]:
        start = _line_col_to_abs(source, int(rec["lineno"]), int(rec["col_offset"]))
        end = _line_col_to_abs(source, int(rec["end_lineno"]), int(rec["end_col_offset"]))
        segment = source[start:end]
        if old not in segment:
            raise RuntimeError(
                f"{script.name}: AST-selected literal no longer contains {old!r}: {segment!r}"
            )
        replacement_segment = segment.replace(old, new)
        patches.append((start, end, segment, replacement_segment))

    # Apply from end to start so offsets remain valid.
    patched = source
    for start, end, before, after in sorted(patches, key=lambda x: x[0], reverse=True):
        if patched[start:end] != before:
            raise RuntimeError(f"{script.name}: patch span drift detected")
        patched = patched[:start] + after + patched[end:]

    # Syntax and scientific-logic guard: the only textual edits are the selected
    # output-directory string literals above.
    ast.parse(patched)
    if patched == source:
        raise RuntimeError(f"{script.name}: patch produced no source change")

    patched_copy.parent.mkdir(parents=True, exist_ok=True)
    patched_copy.write_text(patched, encoding="utf-8")

    audit = {
        "original_script": str(script),
        "original_sha256": sha256_file(script),
        "patched_copy": str(patched_copy),
        "patched_sha256": sha256_file(patched_copy),
        "replacement_count": len(patches),
        "old_output_token": old,
        "new_output_token": new,
        "patches": [
            {
                "start": s,
                "end": e,
                "before": b,
                "after": a,
            }
            for s, e, b, a in patches
        ],
        "scientific_logic_changed": False,
        "only_output_directory_literals_changed": True,
    }
    return audit



LEGACY_PROJECT_ROOT = r"F:\MEDSEG_SAFETTA"


def _constant_span(source: str, node: ast.Constant) -> Tuple[int, int]:
    return (
        _line_col_to_abs(source, int(node.lineno), int(node.col_offset)),
        _line_col_to_abs(source, int(node.end_lineno), int(node.end_col_offset)),
    )


def apply_portable_execution_patch(
    script: Path,
    root: Path,
    stage: ReplayStage,
    strategy: Dict[str, Any],
    replay_dir: Path,
    patched_copy: Path,
) -> Dict[str, Any]:
    """
    Create a temporary execution-only copy of a retained historical script.

    Allowed textual changes:
      1) exact legacy project-root path literals:
           F:\\MEDSEG_SAFETTA -> <current --root>
      2) for scripts without an output CLI, the single statically audited
         output-directory literal -> <fresh replay stage directory>

    The retained source file itself is SHA-verified before this function is
    called and is never modified.
    """
    source = script.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    output_plan = strategy.get("patch_plan")
    output_spans = set()
    if output_plan is not None:
        if not output_plan.get("safe"):
            raise RuntimeError(f"Unsafe output patch plan: {output_plan}")
        for rec in output_plan.get("candidates", []):
            output_spans.add(
                (
                    int(rec["lineno"]),
                    int(rec["col_offset"]),
                    int(rec["end_lineno"]),
                    int(rec["end_col_offset"]),
                )
            )

    patches: List[Tuple[int, int, str, str, Dict[str, Any]]] = []
    root_text = str(root.resolve())

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and hasattr(node, "end_lineno")
        ):
            continue

        span_key = (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        )
        original_value = node.value
        new_value = original_value
        reasons: List[str] = []

        if LEGACY_PROJECT_ROOT.lower() in original_value.lower():
            # Preserve exact suffix/casing while replacing the known absolute
            # project-root prefix only.
            pattern = re.compile(re.escape(LEGACY_PROJECT_ROOT), re.IGNORECASE)
            new_value = pattern.sub(lambda _: root_text, new_value)
            reasons.append("PROJECT_ROOT_PORTABILITY")

        if span_key in output_spans:
            # The static plan has already proven this literal is assigned only
            # to an output-like variable and contains the frozen output dirname.
            new_value = str(replay_dir)
            reasons.append("FRESH_REPLAY_OUTPUT_REDIRECTION")

        if new_value == original_value:
            continue

        start, end = _constant_span(source, node)
        before = source[start:end]
        after = repr(new_value)
        patches.append(
            (
                start,
                end,
                before,
                after,
                {
                    "original_value": original_value,
                    "new_value": new_value,
                    "reasons": reasons,
                    "line": int(node.lineno),
                },
            )
        )

    patched = source
    for start, end, before, after, _meta in sorted(
        patches, key=lambda x: x[0], reverse=True
    ):
        if patched[start:end] != before:
            raise RuntimeError(f"{script.name}: portability patch span drift")
        patched = patched[:start] + after + patched[end:]

    ast.parse(patched)
    patched_copy.parent.mkdir(parents=True, exist_ok=True)
    patched_copy.write_text(patched, encoding="utf-8")

    root_change_needed = str(root.resolve()).lower() != LEGACY_PROJECT_ROOT.lower()
    if root_change_needed:
        # If relocated, at least one legacy-root literal must have been patched
        # for scripts that contain the legacy root.
        legacy_present = LEGACY_PROJECT_ROOT.lower() in source.lower()
        root_patch_count = sum(
            "PROJECT_ROOT_PORTABILITY" in m["reasons"]
            for *_x, m in patches
        )
        if legacy_present and root_patch_count == 0:
            raise RuntimeError(
                f"{script.name}: legacy root found but no portability patch applied"
            )

    audit = {
        "original_script": str(script),
        "original_sha256": sha256_file(script),
        "patched_copy": str(patched_copy),
        "patched_sha256": sha256_file(patched_copy),
        "legacy_project_root": LEGACY_PROJECT_ROOT,
        "runtime_project_root": root_text,
        "replacement_count": len(patches),
        "patches": [m for *_x, m in patches],
        "scientific_logic_changed": False,
        "allowed_changes_only": [
            "PROJECT_ROOT_PORTABILITY",
            "FRESH_REPLAY_OUTPUT_REDIRECTION",
        ],
    }
    return audit


PATCH_LAUNCHER = r"""
import sys
from pathlib import Path

patched_path = Path(sys.argv[1])
original_path = str(Path(sys.argv[2]))
extra_args = list(sys.argv[3:])
source = patched_path.read_text(encoding="utf-8")

# Preserve the retained script's logical __file__/argv[0], while forwarding
# normal CLI arguments such as --output-dir.
sys.argv = [original_path] + extra_args
ns = {
    "__name__": "__main__",
    "__file__": original_path,
    "__package__": None,
    "__cached__": None,
}
exec(compile(source, original_path, "exec"), ns, ns)
"""


def build_execution_strategy(
    root: Path,
    replay_root: Path,
    stage: ReplayStage,
) -> Dict[str, Any]:
    script = require(root / "code" / stage.script_name)
    opt = detect_output_argument_static(script)
    if opt is not None:
        return {
            "safe": True,
            "strategy": "CLI_OUTPUT_DIRECTORY",
            "output_arg": opt,
            "patch_plan": None,
        }

    replacement_name = f"{replay_root.name}/{stage.stage_id}"
    plan = build_static_output_patch_plan(
        script,
        stage.frozen_output_dir,
        replacement_name,
    )
    return {
        "safe": bool(plan["safe"]),
        "strategy": "STATIC_OUTPUT_LITERAL_PATCH" if plan["safe"] else "BLOCKED",
        "output_arg": None,
        "patch_plan": plan,
    }


def verify_v1_preflight_side_effect_guard(
    root: Path,
    canonical: Dict[str, Any],
) -> Dict[str, Any]:
    """
    numeric_replay_v1 used ``script --help`` for output-argument detection.
    For scripts without argparse that can, in principle, execute the script.
    Verify key frozen artifacts used by those two blocked stages still have the
    exact SHA recorded *before* the v1 preflight.
    """
    rows = {r["anchor_id"]: r for r in canonical["final_rows"]}
    guard_ids = [
        "COLON_THRESHOLD",
        "SENS_POLYPGEN_M01_AUROC",
        "SENS_POLYPGEN_M05_AUROC",
        "SENS_PROMISE_M01_AUROC",
        "SENS_PROMISE_M05_AUROC",
    ]

    checks = []
    for aid in guard_ids:
        row = rows[aid]
        rel = row.get("relpath")
        expected_sha = row.get("file_sha256")
        if not rel or not expected_sha:
            raise RuntimeError(
                f"{aid}: canonical provenance lacks preflight-side-effect SHA evidence"
            )
        p = require(root / rel)
        got = sha256_file(p)
        ok = got.lower() == str(expected_sha).lower()
        checks.append({
            "anchor_id": aid,
            "path": str(p),
            "expected_sha256": expected_sha,
            "current_sha256": got,
            "ok": ok,
        })

    return {
        "checks": checks,
        "gate": "PASS" if all(x["ok"] for x in checks) else "BLOCKED",
        "interpretation": (
            "PASS verifies key R10L0 and sensitivity frozen artifacts were not "
            "changed by the earlier unsafe --help preflight probe."
        ),
    }




def _console_safe_write(text: str) -> None:
    """
    Write child output without allowing the parent Windows console encoding
    (often GBK/cp936) to crash the replay harness.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        sys.stdout.write(safe)
        sys.stdout.flush()
    except Exception:
        # Logging is the canonical record; console rendering must never decide
        # scientific replay success/failure.
        pass


def stream_subprocess(cmd: List[str], log_path: Path, cwd: Path) -> int:
    env = os.environ.copy()

    # Scientific seeds are unchanged. These settings affect text I/O only.
    env.setdefault("PYTHONHASHSEED", "0")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        log.write("TEXT_IO_ENV:\n")
        log.write("  PYTHONUTF8=1\n")
        log.write("  PYTHONIOENCODING=utf-8\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _console_safe_write(line)
            log.write(line)
        return proc.wait()



def is_summary_candidate(path: Path, tokens: Sequence[str]) -> bool:
    if path.suffix.lower() not in {".csv", ".json"}:
        return False
    n = path.name.lower()
    if any(x in n for x in ("prediction", "predictions", "outcomes", "panel", "replicates")):
        return False
    return any(t.lower() in n for t in tokens)


def common_summary_pairs(
    frozen_dir: Path,
    replay_dir: Path,
    tokens: Sequence[str],
) -> List[Tuple[Path, Path]]:
    replay_files = [p for p in replay_dir.rglob("*") if p.is_file()]
    pairs = []

    for rp in replay_files:
        rel = rp.relative_to(replay_dir)
        fp = frozen_dir / rel
        if not fp.is_file():
            continue
        if rp.stat().st_size > MAX_COMPARE_BYTES or fp.stat().st_size > MAX_COMPARE_BYTES:
            continue
        if is_summary_candidate(rp, tokens):
            pairs.append((fp, rp))

    if pairs:
        return sorted(pairs, key=lambda x: str(x[1]).lower())

    # Fallback: small CSV/JSON common files, still excluding huge row-level assets.
    for rp in replay_files:
        rel = rp.relative_to(replay_dir)
        fp = frozen_dir / rel
        if not fp.is_file():
            continue
        if rp.suffix.lower() not in {".csv", ".json"}:
            continue
        if rp.stat().st_size > 10 * 1024 * 1024 or fp.stat().st_size > 10 * 1024 * 1024:
            continue
        n = rp.name.lower()
        if any(x in n for x in ("prediction", "predictions", "outcomes", "panel", "replicates")):
            continue
        pairs.append((fp, rp))
    return sorted(pairs, key=lambda x: str(x[1]).lower())


def numeric_series_equal(a: pd.Series, b: pd.Series) -> Tuple[bool, float]:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    if aa.shape != bb.shape:
        return False, math.inf
    nan_same = np.array_equal(np.isnan(aa), np.isnan(bb))
    if not nan_same:
        return False, math.inf
    mask = ~(np.isnan(aa) | np.isnan(bb))
    if not np.any(mask):
        return True, 0.0
    maxerr = float(np.max(np.abs(aa[mask] - bb[mask])))
    return maxerr <= NUMERIC_ATOL, maxerr


def compare_csv(frozen: Path, replay: Path) -> Dict[str, Any]:
    a = pd.read_csv(frozen)
    b = pd.read_csv(replay)

    if list(a.columns) != list(b.columns):
        return {
            "status": "FAIL_COLUMNS",
            "frozen_columns": list(a.columns),
            "replay_columns": list(b.columns),
        }
    if len(a) != len(b):
        return {
            "status": "FAIL_ROWS",
            "frozen_rows": len(a),
            "replay_rows": len(b),
        }

    numeric_cols = []
    maxerr = 0.0
    failed = []

    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            numeric_cols.append(c)
            ok, err = numeric_series_equal(a[c], b[c])
            maxerr = max(maxerr, err)
            if not ok:
                failed.append({"column": c, "max_abs_error": err})

    if not numeric_cols:
        return {"status": "NO_NUMERIC_COLUMNS", "rows": len(a)}

    return {
        "status": "PASS" if not failed else "FAIL_NUMERIC",
        "rows": len(a),
        "numeric_columns_compared": numeric_cols,
        "max_abs_error": maxerr,
        "failed_columns": failed,
    }


IGNORE_JSON_KEY_TOKENS = (
    "path", "dir", "sha", "time", "timestamp", "created", "output",
    "runtime", "duration", "elapsed", "wall_clock",
)


def flatten_numeric_json(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(tok in kl for tok in IGNORE_JSON_KEY_TOKENS):
                continue
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric_json(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_numeric_json(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        x = float(obj)
        if math.isfinite(x):
            out[prefix] = x
    return out


def compare_json(frozen: Path, replay: Path) -> Dict[str, Any]:
    a = flatten_numeric_json(load_json(frozen))
    b = flatten_numeric_json(load_json(replay))
    keys = sorted(set(a).intersection(b))
    if not keys:
        return {"status": "NO_COMMON_NUMERIC_KEYS"}

    failed = []
    maxerr = 0.0
    for k in keys:
        err = abs(a[k] - b[k])
        maxerr = max(maxerr, err)
        if err > NUMERIC_ATOL:
            failed.append({"key": k, "frozen": a[k], "replay": b[k], "abs_error": err})

    return {
        "status": "PASS" if not failed else "FAIL_NUMERIC",
        "numeric_keys_compared": len(keys),
        "max_abs_error": maxerr,
        "failed_keys": failed[:100],
    }


def compare_pair(frozen: Path, replay: Path) -> Dict[str, Any]:
    if replay.suffix.lower() == ".csv":
        result = compare_csv(frozen, replay)
    elif replay.suffix.lower() == ".json":
        result = compare_json(frozen, replay)
    else:
        result = {"status": "SKIP"}
    result.update({
        "frozen_path": str(frozen),
        "replay_path": str(replay),
        "frozen_sha256": sha256_file(frozen),
        "replay_sha256": sha256_file(replay),
    })
    return result



SENSITIVITY_SCRIPT_NAME = "Q1_SAFETTA_harm_margin_sensitivity_v1.py"
SENSITIVITY_SCRIPT_SHA256 = "37226877f7879554379c5a362ab1042313c738d18624cd424faa45b836ffcbeb"
SENSITIVITY_FROZEN_DIR = "Q1_SAFETTA_harm_margin_sensitivity_v1"
SENSITIVITY_FROZEN_FILE = "harm_margin_sensitivity_ranked.csv"
SENSITIVITY_MARGINS = (0.01, 0.02, 0.05)

SENSITIVITY_ANCHOR_MAP = {
    ("PROMISE12", 0.01): "SENS_PROMISE_M01_AUROC",
    ("PROMISE12", 0.05): "SENS_PROMISE_M05_AUROC",
    ("PolypGen", 0.01): "SENS_POLYPGEN_M01_AUROC",
    ("PolypGen", 0.05): "SENS_POLYPGEN_M05_AUROC",
}


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        raise RuntimeError("Degenerate HARM labels in sensitivity replay.")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) != 2:
        raise RuntimeError("Degenerate HARM labels in sensitivity replay.")
    return float(average_precision_score(y, s))


def recompute_sensitivity(root: Path, replay_root: Path) -> Dict[str, Any]:
    """
    Freshly recompute HARM-margin sensitivity from the exact frozen row-level
    evaluation panels. This avoids modifying/executing the historical
    Q1_SAFETTA_harm_margin_sensitivity_v1.py, which has no safe output redirection.

    Scientific definition:
      HARM_m = 1[delta_dice <= -m]
      pooled AUROC/AUPRC over all rows
      macro-family AUROC/AUPRC = arithmetic mean over model families
    """
    script = require(root / "code" / SENSITIVITY_SCRIPT_NAME)
    check_python_syntax(script)
    got_script_sha = sha256_file(script).lower()
    if got_script_sha != SENSITIVITY_SCRIPT_SHA256.lower():
        raise RuntimeError(
            "Sensitivity historical script SHA changed. "
            f"expected={SENSITIVITY_SCRIPT_SHA256} got={got_script_sha}"
        )

    frozen_csv = require(
        root / "outputs" / SENSITIVITY_FROZEN_DIR / SENSITIVITY_FROZEN_FILE
    )
    frozen_df = pd.read_csv(frozen_csv)

    datasets = [
        {
            "dataset_hint": "PROMISE12",
            "csv": require(
                root / "outputs"
                / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
                / "CM6_PROMISE12_OUTCOMES.csv"
            ),
            "score_column": "risk_score",
            "delta_column": "delta_dice",
            "family_column": "family",
            "cluster_column": "",
            "priority": 5,
        },
        {
            "dataset_hint": "PolypGen",
            "csv": require(
                root / "outputs"
                / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
                / "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
            ),
            "score_column": "frozen_safety_probability",
            "delta_column": "delta_dice",
            "family_column": "model_family",
            "cluster_column": "sample_id",
            "priority": 0,
        },
    ]

    rows: List[Dict[str, Any]] = []
    iterator = datasets
    if tqdm is not None:
        iterator = tqdm(
            datasets,
            desc="Fresh HARM-margin sensitivity",
            unit="dataset",
            dynamic_ncols=True,
        )

    source_shas = {}
    for spec in iterator:
        p = spec["csv"]
        df = pd.read_csv(p)
        source_shas[spec["dataset_hint"]] = sha256_file(p)

        for c in (
            spec["score_column"],
            spec["delta_column"],
            spec["family_column"],
        ):
            if c not in df.columns:
                raise KeyError(f"{p}: missing sensitivity field {c!r}")

        score = pd.to_numeric(df[spec["score_column"]], errors="raise").to_numpy(float)
        delta = pd.to_numeric(df[spec["delta_column"]], errors="raise").to_numpy(float)

        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(delta)):
            raise RuntimeError(f"{p}: non-finite score/delta values")

        family = df[spec["family_column"]].astype(str)

        for margin in SENSITIVITY_MARGINS:
            y = (delta <= -float(margin)).astype(int)

            pooled_auc = _safe_auroc(y, score)
            pooled_ap = _safe_auprc(y, score)

            fam_auc = []
            fam_ap = []
            for fam in sorted(family.unique()):
                mask = family.eq(fam).to_numpy()
                yf = y[mask]
                sf = score[mask]
                fam_auc.append(_safe_auroc(yf, sf))
                fam_ap.append(_safe_auprc(yf, sf))

            rows.append({
                "dataset_hint": spec["dataset_hint"],
                "csv": str(p),
                "score_column": spec["score_column"],
                "delta_column": spec["delta_column"],
                "family_column": spec["family_column"],
                "cluster_column": spec["cluster_column"],
                "rows": int(len(df)),
                "harm_margin": float(margin),
                "harm_prevalence": float(np.mean(y)),
                "pooled_auroc": pooled_auc,
                "pooled_auprc": pooled_ap,
                "macro_family_auroc": float(np.mean(fam_auc)),
                "macro_family_auprc": float(np.mean(fam_ap)),
                "priority": int(spec["priority"]),
            })

    fresh = pd.DataFrame(rows)
    expected_cols = [
        "dataset_hint","csv","score_column","delta_column","family_column",
        "cluster_column","rows","harm_margin","harm_prevalence","pooled_auroc",
        "pooled_auprc","macro_family_auroc","macro_family_auprc","priority",
    ]
    fresh = fresh[expected_cols]

    out_dir = replay_root / "SENSITIVITY"
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_csv = out_dir / SENSITIVITY_FROZEN_FILE
    fresh.to_csv(fresh_csv, index=False)

    # Compare only the manuscript-facing sensitivity rows.
    #
    # The historical sensitivity CSV can contain additional exploratory datasets.
    # The paper's frozen sensitivity section uses exactly:
    #   PROMISE12 × margins {0.01, 0.02, 0.05}
    #   PolypGen  × margins {0.01, 0.02, 0.05}
    # i.e. six canonical rows. Extra historical rows are preserved/audited but
    # must not block replay of the paper-facing analysis.
    key = ["dataset_hint", "harm_margin"]
    if not set(key).issubset(frozen_df.columns):
        raise RuntimeError(f"Frozen sensitivity CSV lacks keys {key}")

    paper_datasets = {"PROMISE12", "PolypGen"}
    frozen_margin = pd.to_numeric(
        frozen_df["harm_margin"], errors="raise"
    ).astype(float)

    dataset_mask = frozen_df["dataset_hint"].astype(str).isin(paper_datasets)
    margin_mask = np.zeros(len(frozen_df), dtype=bool)
    for m in SENSITIVITY_MARGINS:
        margin_mask |= np.isclose(
            frozen_margin.to_numpy(float),
            float(m),
            atol=0.0,
            rtol=0.0,
        )

    frozen_paper = frozen_df.loc[dataset_mask & margin_mask].copy()
    frozen_extra = frozen_df.loc[~(dataset_mask & margin_mask)].copy()

    if len(frozen_paper) != 6:
        raise RuntimeError(
            "Paper-facing frozen sensitivity subset must contain exactly 6 rows "
            f"(2 datasets × 3 margins), got {len(frozen_paper)}. "
            f"datasets={sorted(frozen_paper['dataset_hint'].astype(str).unique().tolist())}"
        )
    if len(fresh) != 6:
        raise RuntimeError(
            f"Fresh paper-facing sensitivity replay must contain 6 rows, got {len(fresh)}"
        )

    for dataset in sorted(paper_datasets):
        for margin in SENSITIVITY_MARGINS:
            fmask = (
                frozen_paper["dataset_hint"].astype(str).eq(dataset)
                & np.isclose(
                    pd.to_numeric(
                        frozen_paper["harm_margin"], errors="raise"
                    ).to_numpy(float),
                    float(margin),
                    atol=0.0,
                    rtol=0.0,
                )
            )
            nmask = (
                fresh["dataset_hint"].astype(str).eq(dataset)
                & np.isclose(
                    pd.to_numeric(
                        fresh["harm_margin"], errors="raise"
                    ).to_numpy(float),
                    float(margin),
                    atol=0.0,
                    rtol=0.0,
                )
            )
            if int(fmask.sum()) != 1 or int(nmask.sum()) != 1:
                raise RuntimeError(
                    f"Sensitivity key not unique for {dataset}, margin={margin}: "
                    f"frozen={int(fmask.sum())}, fresh={int(nmask.sum())}"
                )

    numeric_fields = [
        "rows","harm_prevalence","pooled_auroc","pooled_auprc",
        "macro_family_auroc","macro_family_auprc","priority",
    ]
    comparisons = []
    for _, fr in fresh.iterrows():
        mask = (
            frozen_paper["dataset_hint"].astype(str).eq(str(fr["dataset_hint"]))
            & np.isclose(
                pd.to_numeric(
                    frozen_paper["harm_margin"], errors="raise"
                ).to_numpy(float),
                float(fr["harm_margin"]),
                atol=0.0,
                rtol=0.0,
            )
        )
        old = frozen_paper.loc[mask]
        if len(old) != 1:
            raise RuntimeError(
                f"Frozen sensitivity key not unique: "
                f"{fr['dataset_hint']} margin={fr['harm_margin']}"
            )
        old = old.iloc[0]
        per_field = {}
        for field in numeric_fields:
            ov = float(old[field])
            nv = float(fr[field])
            err = abs(ov - nv)
            per_field[field] = {
                "frozen": ov,
                "fresh": nv,
                "abs_error": err,
                "pass": err <= NUMERIC_ATOL,
            }
        comparisons.append({
            "dataset_hint": fr["dataset_hint"],
            "harm_margin": float(fr["harm_margin"]),
            "fields": per_field,
        })

    all_numeric_pass = all(
        x["pass"]
        for comp in comparisons
        for x in comp["fields"].values()
    )

    anchor_results = {}
    for (dataset, margin), aid in SENSITIVITY_ANCHOR_MAP.items():
        row = fresh.loc[
            fresh["dataset_hint"].eq(dataset)
            & np.isclose(fresh["harm_margin"].to_numpy(float), margin, atol=0.0, rtol=0.0)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Fresh sensitivity anchor row not unique: {aid}")
        value = float(row.iloc[0]["pooled_auroc"])
        anchor_results[aid] = {
            "value": value,
            "status": "PASS" if all_numeric_pass else "FAIL",
        }

    report = {
        "historical_script": str(script),
        "historical_script_sha256": got_script_sha,
        "historical_script_executed": False,
        "reason_historical_script_not_executed": (
            "No safe CLI or hard-coded output-directory redirection was available; "
            "the deterministic statistic is freshly recomputed from canonical row-level inputs."
        ),
        "frozen_csv": str(frozen_csv),
        "frozen_csv_sha256": sha256_file(frozen_csv),
        "frozen_total_rows": int(len(frozen_df)),
        "frozen_paper_facing_rows": int(len(frozen_paper)),
        "frozen_nonpaper_rows_excluded_from_numeric_replay": int(len(frozen_extra)),
        "frozen_nonpaper_row_keys": [
            {
                "dataset_hint": str(r.get("dataset_hint", "")),
                "harm_margin": float(r.get("harm_margin")),
            }
            for _, r in frozen_extra.iterrows()
        ],
        "paper_facing_scope": {
            "datasets": sorted(paper_datasets),
            "harm_margins": list(SENSITIVITY_MARGINS),
            "expected_rows": 6,
        },
        "fresh_csv": str(fresh_csv),
        "fresh_csv_sha256": sha256_file(fresh_csv),
        "input_sha256": source_shas,
        "comparisons": comparisons,
        "anchor_results": anchor_results,
        "status": "PASS" if all_numeric_pass else "FAIL",
    }
    (out_dir / "SENSITIVITY_FRESH_RECOMPUTE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report



POLYPGEN_DIRECT_ANCHORS = (
    "POLYPGEN_AUROC",
    "POLYPGEN_AUPRC",
)

SUN_DIRECT_ANCHORS = (
    "SUN_TENT1_AUROC",
    "SUN_PL_AUROC",
    "SUN_ACTION_JACCARD",
)

PROMISE_DIRECT_ANCHORS = (
    "PROMISE_MACRO_AUROC",
    "PROMISE_POOLED_AUROC",
    "PROMISE_UNET_AUROC",
    "PROMISE_DEEPLAB_AUROC",
    "PROMISE_SEGFORMER_AUROC",
    "PROMISE_PRIMARY_CI_LOW",
    "PROMISE_PRIMARY_CI_HIGH",
    "PROMISE_SOURCE_TAU",
)


def _canonical_anchor_row(canonical: Dict[str, Any], anchor_id: str) -> Dict[str, Any]:
    rows = {r["anchor_id"]: r for r in canonical["final_rows"]}
    if anchor_id not in rows:
        raise KeyError(f"Canonical anchor not found: {anchor_id}")
    return rows[anchor_id]


def _anchor_compare(
    canonical: Dict[str, Any],
    anchor_id: str,
    fresh_value: float,
) -> Dict[str, Any]:
    row = _canonical_anchor_row(canonical, anchor_id)
    # Compare to canonical observed value (the retained exact result) using the
    # paper anchor tolerance recorded in the canonical provenance.
    target = float(row["observed_value"])
    tol = float(row["tolerance"])
    err = abs(float(fresh_value) - target)
    return {
        "anchor_id": anchor_id,
        "fresh_value": float(fresh_value),
        "canonical_observed_value": target,
        "paper_tolerance": tol,
        "abs_error": err,
        "status": "PASS" if err <= tol else "FAIL",
    }


def _resolve_exact_or_tokens(
    df: pd.DataFrame,
    exact_candidates: Sequence[str],
    required_tokens: Sequence[str],
    label: str,
) -> str:
    for c in exact_candidates:
        if c in df.columns:
            return c

    hits = []
    for c in df.columns:
        lc = c.lower()
        if all(tok.lower() in lc for tok in required_tokens):
            hits.append(c)

    if len(hits) != 1:
        raise RuntimeError(
            f"Cannot resolve {label}: exact={list(exact_candidates)} "
            f"tokens={list(required_tokens)} hits={hits} columns={list(df.columns)}"
        )
    return hits[0]


def _macro_family_auc(
    df: pd.DataFrame,
    family_col: str,
    y: np.ndarray,
    score: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    fam_vals: Dict[str, float] = {}
    families = df[family_col].astype(str)
    for fam in sorted(families.unique()):
        mask = families.eq(fam).to_numpy()
        yf = y[mask]
        sf = score[mask]
        if len(np.unique(yf)) != 2:
            raise RuntimeError(f"Degenerate HARM labels in family={fam}")
        fam_vals[fam] = float(roc_auc_score(yf, sf))
    return float(np.mean(list(fam_vals.values()))), fam_vals


def _macro_family_auprc(
    df: pd.DataFrame,
    family_col: str,
    y: np.ndarray,
    score: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import average_precision_score
    fam_vals: Dict[str, float] = {}
    families = df[family_col].astype(str)
    for fam in sorted(families.unique()):
        mask = families.eq(fam).to_numpy()
        yf = y[mask]
        sf = score[mask]
        if len(np.unique(yf)) != 2:
            raise RuntimeError(f"Degenerate HARM labels in family={fam}")
        fam_vals[fam] = float(average_precision_score(yf, sf))
    return float(np.mean(list(fam_vals.values()))), fam_vals


def recompute_polypgen_from_frozen_panel(
    root: Path,
    canonical: Dict[str, Any],
    replay_root: Path,
) -> Dict[str, Any]:
    """
    Public Level-A replay for PolypGen.

    The author-side full historical replay already regenerated this panel from
    locked SOURCE/A1 predictions + first-reveal GT. The public package avoids
    redistributing those large prediction NPZs / raw GT by freshly recomputing
    the paper ranking statistics from the frozen post-GT row-level panel.
    """
    panel_path = require(
        root
        / "outputs"
        / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
        / "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
    )
    df = pd.read_csv(panel_path)

    family_col = _resolve_exact_or_tokens(
        df, ("model_family", "family"), ("family",), "PolypGen family"
    )
    score_col = _resolve_exact_or_tokens(
        df,
        ("frozen_safety_probability", "risk_score", "safety_score"),
        ("safety", "prob"),
        "PolypGen score",
    )
    delta_col = _resolve_exact_or_tokens(
        df, ("delta_dice",), ("delta", "dice"), "PolypGen delta Dice"
    )

    delta = pd.to_numeric(df[delta_col], errors="raise").to_numpy(float)
    score = pd.to_numeric(df[score_col], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(score)):
        raise RuntimeError("PolypGen panel contains non-finite delta/score.")

    y = (delta <= -0.02).astype(np.int64)
    macro_auc, fam_auc = _macro_family_auc(df, family_col, y, score)
    macro_ap, fam_ap = _macro_family_auprc(df, family_col, y, score)

    results = {
        "POLYPGEN_AUROC": _anchor_compare(canonical, "POLYPGEN_AUROC", macro_auc),
        "POLYPGEN_AUPRC": _anchor_compare(canonical, "POLYPGEN_AUPRC", macro_ap),
    }
    status = "PASS" if all(x["status"] == "PASS" for x in results.values()) else "FAIL"

    report = {
        "status": status,
        "mode": "FRESH_STATISTIC_RECOMPUTE_FROM_FROZEN_POST_GT_PANEL",
        "panel": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "rows": int(len(df)),
        "family_column": family_col,
        "score_column": score_col,
        "delta_column": delta_col,
        "harm_rule": "delta_dice <= -0.02",
        "macro_auroc": macro_auc,
        "macro_auprc": macro_ap,
        "family_aurocs": fam_auc,
        "family_auprcs": fam_ap,
        "results": results,
        "boundary": (
            "Raw PolypGen GT and SOURCE/A1 prediction NPZs are not needed for "
            "public Level-A paper-statistic replay. Their exact historical stage "
            "replay was validated separately by the author-side PASS_NUMERIC_REPLAY."
        ),
    }
    out_dir = replay_root / "DIRECT_POLYPGEN"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "POLYPGEN_DIRECT_RECOMPUTE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def recompute_sun_from_frozen_panel(
    root: Path,
    canonical: Dict[str, Any],
    replay_root: Path,
) -> Dict[str, Any]:
    """
    Freshly recompute SUN TENT1/PL HARM ranking and action HARM-set Jaccard from
    the frozen dual-action post-GT evaluation panel.
    """
    panel_path = require(
        root
        / "outputs"
        / "Q1_R14C3B_sunseg_gt_reveal_dual_action_frozen_score_evaluation_fix1_v1"
        / "R14C3B_LOCKED_SCORE_DUAL_ACTION_EVALUATION_PANEL.csv"
    )
    df = pd.read_csv(panel_path)

    family_col = _resolve_exact_or_tokens(
        df, ("model_family", "family"), ("family",), "SUN family"
    )
    score_col = _resolve_exact_or_tokens(
        df,
        ("frozen_safety_probability", "risk_score", "safety_score"),
        ("safety",),
        "SUN score",
    )

    # Prefer delta columns; if unavailable, fall back to explicit harm columns.
    try:
        tent_delta_col = _resolve_exact_or_tokens(
            df,
            ("tent1_delta_dice", "delta_dice_tent1", "tent_delta_dice"),
            ("tent", "delta"),
            "SUN TENT1 delta",
        )
        tent_delta = pd.to_numeric(df[tent_delta_col], errors="raise").to_numpy(float)
        tent_harm = tent_delta <= -0.02
    except RuntimeError:
        tent_harm_col = _resolve_exact_or_tokens(
            df,
            ("tent1_harm", "tent1_harmful", "tent_harm"),
            ("tent", "harm"),
            "SUN TENT1 harm",
        )
        tent_delta_col = None
        tent_harm = df[tent_harm_col].astype(int).to_numpy() == 1

    try:
        pl_delta_col = _resolve_exact_or_tokens(
            df,
            ("pl_delta_dice", "pl_conf90_delta_dice", "delta_dice_pl"),
            ("pl", "delta"),
            "SUN PL delta",
        )
        pl_delta = pd.to_numeric(df[pl_delta_col], errors="raise").to_numpy(float)
        pl_harm = pl_delta <= -0.02
    except RuntimeError:
        pl_harm_col = _resolve_exact_or_tokens(
            df,
            ("pl_harm", "pl_conf90_harm", "pl_harmful"),
            ("pl", "harm"),
            "SUN PL harm",
        )
        pl_delta_col = None
        pl_harm = df[pl_harm_col].astype(int).to_numpy() == 1

    score = pd.to_numeric(df[score_col], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(score)):
        raise RuntimeError("SUN panel contains non-finite score.")

    tent_auc, tent_fam = _macro_family_auc(
        df, family_col, tent_harm.astype(np.int64), score
    )
    pl_auc, pl_fam = _macro_family_auc(
        df, family_col, pl_harm.astype(np.int64), score
    )

    inter = int(np.logical_and(tent_harm, pl_harm).sum())
    union = int(np.logical_or(tent_harm, pl_harm).sum())
    jaccard = 1.0 if union == 0 else inter / union

    results = {
        "SUN_TENT1_AUROC": _anchor_compare(
            canonical, "SUN_TENT1_AUROC", tent_auc
        ),
        "SUN_PL_AUROC": _anchor_compare(
            canonical, "SUN_PL_AUROC", pl_auc
        ),
        "SUN_ACTION_JACCARD": _anchor_compare(
            canonical, "SUN_ACTION_JACCARD", jaccard
        ),
    }
    status = "PASS" if all(x["status"] == "PASS" for x in results.values()) else "FAIL"

    report = {
        "status": status,
        "mode": "FRESH_STATISTIC_RECOMPUTE_FROM_FROZEN_DUAL_ACTION_POST_GT_PANEL",
        "panel": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "rows": int(len(df)),
        "family_column": family_col,
        "score_column": score_col,
        "tent_delta_column": tent_delta_col,
        "pl_delta_column": pl_delta_col,
        "tent_macro_auroc": tent_auc,
        "pl_macro_auroc": pl_auc,
        "tent_family_aurocs": tent_fam,
        "pl_family_aurocs": pl_fam,
        "harm_intersection": inter,
        "harm_union": union,
        "harm_jaccard": float(jaccard),
        "results": results,
        "boundary": (
            "Raw SUN GT and SOURCE/TENT1/PL prediction arrays are excluded from "
            "the public Level-A replay; their historical post-GT stage was "
            "already rerun exactly in the author-side PASS_NUMERIC_REPLAY."
        ),
    }
    out_dir = replay_root / "DIRECT_SUN"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUN_DIRECT_RECOMPUTE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _resolve_promise_case_col(df: pd.DataFrame) -> str:
    for c in ("case_key", "patient_key", "patient_id", "case_id", "patient", "case"):
        if c in df.columns:
            return c

    hits = [
        c for c in df.columns
        if ("case" in c.lower() or "patient" in c.lower())
        and "index" not in c.lower()
    ]
    # Prefer a column with exactly 50 unique values.
    exact50 = [c for c in hits if df[c].astype(str).nunique() == 50]
    if len(exact50) == 1:
        return exact50[0]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(
        f"Cannot resolve PROMISE patient/case column: hits={hits}, "
        f"unique50={exact50}, columns={list(df.columns)}"
    )


def _promise_macro_bootstrap(
    df: pd.DataFrame,
    family_col: str,
    case_col: str,
    score_col: str,
    delta_col: str,
    reps: int = 2000,
    seed: int = 20260906,
) -> Dict[str, Any]:
    """
    Patient-cluster bootstrap of macro-family HARM AUROC.

    Uses np.random.default_rng(seed), 2000 resamples, and percentile 95% CI.
    Duplicate patient draws are represented with multiplicity by concatenating
    the corresponding rows for each draw.
    """
    cases = sorted(df[case_col].astype(str).unique().tolist())
    if len(cases) != 50:
        raise RuntimeError(f"PROMISE bootstrap expected 50 patients, got {len(cases)}")

    grouped = {
        c: df.loc[df[case_col].astype(str).eq(c)].copy()
        for c in cases
    }
    rng = np.random.default_rng(seed)
    vals: List[float] = []

    iterator = range(reps)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            desc="PROMISE patient bootstrap",
            unit="rep",
            dynamic_ncols=True,
            leave=False,
        )

    for _ in iterator:
        sampled = rng.choice(cases, size=len(cases), replace=True)
        boot = pd.concat([grouped[str(c)] for c in sampled], ignore_index=True)
        y = (
            pd.to_numeric(boot[delta_col], errors="raise").to_numpy(float)
            <= -0.02
        ).astype(np.int64)
        s = pd.to_numeric(boot[score_col], errors="raise").to_numpy(float)
        macro, _ = _macro_family_auc(boot, family_col, y, s)
        vals.append(float(macro))

    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) != reps or not np.all(np.isfinite(arr)):
        raise RuntimeError("PROMISE patient bootstrap produced invalid replicates.")

    return {
        "reps_requested": reps,
        "reps_valid": int(len(arr)),
        "seed": seed,
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


def recompute_promise_from_frozen_outcomes(
    root: Path,
    canonical: Dict[str, Any],
    replay_root: Path,
) -> Dict[str, Any]:
    """
    Public Level-A PROMISE12 replay from the frozen first-GT-reveal outcome table.

    This reproduces ranking statistics and the patient-cluster primary CI
    without needing PROMISE12 raw GT volumes or prediction-mask caches.
    """
    outcomes_path = require(
        root
        / "outputs"
        / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
        / "CM6_PROMISE12_OUTCOMES.csv"
    )
    df = pd.read_csv(outcomes_path)

    family_col = _resolve_exact_or_tokens(
        df, ("family", "model_family"), ("family",), "PROMISE family"
    )
    score_col = _resolve_exact_or_tokens(
        df, ("risk_score", "frozen_safety_probability"), ("risk", "score"),
        "PROMISE risk score"
    )
    delta_col = _resolve_exact_or_tokens(
        df, ("delta_dice",), ("delta", "dice"), "PROMISE delta Dice"
    )
    case_col = _resolve_promise_case_col(df)

    y = (
        pd.to_numeric(df[delta_col], errors="raise").to_numpy(float) <= -0.02
    ).astype(np.int64)
    score = pd.to_numeric(df[score_col], errors="raise").to_numpy(float)

    macro_auc, fam_auc = _macro_family_auc(df, family_col, y, score)
    pooled_auc = float(roc_auc_score(y, score))

    # Robustly identify the manuscript's three family anchors by normalized name.
    def family_value(kind: str) -> float:
        normalized = {
            re.sub(r"[^a-z0-9]", "", k.lower()): v
            for k, v in fam_auc.items()
        }
        if kind == "unet":
            hits = [v for k, v in normalized.items() if "unet" in k]
        elif kind == "deeplab":
            hits = [v for k, v in normalized.items() if "deeplab" in k]
        elif kind == "segformer":
            hits = [v for k, v in normalized.items() if "segformer" in k]
        else:
            raise KeyError(kind)
        if len(hits) != 1:
            raise RuntimeError(
                f"PROMISE family resolution failed for {kind}: {fam_auc}"
            )
        return float(hits[0])

    boot = _promise_macro_bootstrap(
        df=df,
        family_col=family_col,
        case_col=case_col,
        score_col=score_col,
        delta_col=delta_col,
        reps=2000,
        seed=20260906,
    )

    policy_path = require(
        root
        / "outputs"
        / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
        / "CM6_LOCKED_POLICY_EVALUATION.csv"
    )
    policy = pd.read_csv(policy_path)
    if "threshold" not in policy.columns:
        raise RuntimeError("CM6_LOCKED_POLICY_EVALUATION.csv lacks threshold.")
    if "policy" in policy.columns:
        fixed = policy.loc[
            policy["policy"].astype(str).eq("FIXED_SOURCE_THRESHOLD")
        ]
        if len(fixed) == 0:
            raise RuntimeError("FIXED_SOURCE_THRESHOLD rows not found.")
    else:
        fixed = policy

    taus = pd.to_numeric(fixed["threshold"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(taus)):
        raise RuntimeError("PROMISE SOURCE threshold contains non-finite values.")
    source_tau = float(np.median(taus))
    if np.max(np.abs(taus - source_tau)) > 1e-12:
        raise RuntimeError(
            f"PROMISE fixed SOURCE threshold inconsistent across rows: {taus}"
        )

    values = {
        "PROMISE_MACRO_AUROC": macro_auc,
        "PROMISE_POOLED_AUROC": pooled_auc,
        "PROMISE_UNET_AUROC": family_value("unet"),
        "PROMISE_DEEPLAB_AUROC": family_value("deeplab"),
        "PROMISE_SEGFORMER_AUROC": family_value("segformer"),
        "PROMISE_PRIMARY_CI_LOW": boot["ci95_low"],
        "PROMISE_PRIMARY_CI_HIGH": boot["ci95_high"],
        "PROMISE_SOURCE_TAU": source_tau,
    }
    results = {
        aid: _anchor_compare(canonical, aid, val)
        for aid, val in values.items()
    }
    status = "PASS" if all(x["status"] == "PASS" for x in results.values()) else "FAIL"

    report = {
        "status": status,
        "mode": "FRESH_STATISTIC_RECOMPUTE_FROM_FROZEN_CM6_OUTCOMES",
        "outcomes": str(outcomes_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "rows": int(len(df)),
        "patients": int(df[case_col].astype(str).nunique()),
        "family_column": family_col,
        "case_column": case_col,
        "score_column": score_col,
        "delta_column": delta_col,
        "harm_rule": "delta_dice <= -0.02",
        "macro_auroc": macro_auc,
        "pooled_auroc": pooled_auc,
        "family_aurocs": fam_auc,
        "patient_bootstrap": boot,
        "source_tau": source_tau,
        "results": results,
        "boundary": (
            "PROMISE12 raw GT volumes and locked prediction-mask caches are not "
            "redistributed in the minimal public replay. Their exact CM6 historical "
            "stage replay was validated separately by the author-side PASS_NUMERIC_REPLAY."
        ),
    }
    out_dir = replay_root / "DIRECT_PROMISE"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PROMISE_DIRECT_RECOMPUTE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report



PROMISE_UTILITY_DIRECT_ANCHORS = (
    "PROMISE_SAFE_PATIENT_DICE",
    "PROMISE_SOURCE_PATIENT_DICE",
    "PROMISE_ADAPTALL_PATIENT_DICE",
    "PROMISE_RANDOM_SLICE_DICE_GAIN",
    "PROMISE_RANDOM_PREVENTED_HARM_GAIN",
)


def _pooled_binary_dice_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> float:
    tp_sum = int(np.asarray(tp, dtype=np.int64).sum())
    fp_sum = int(np.asarray(fp, dtype=np.int64).sum())
    fn_sum = int(np.asarray(fn, dtype=np.int64).sum())
    den = 2 * tp_sum + fp_sum + fn_sum
    return 1.0 if den == 0 else (2.0 * tp_sum) / den


def _resolve_delta_metric_column(
    df: pd.DataFrame,
    metric_tokens: Sequence[str],
    label: str,
) -> str:
    """
    Resolve one paired-delta column in the frozen CM7B bootstrap-replicate table.
    Preference is given to columns containing both metric tokens and 'delta'.
    """
    hits = []
    for c in df.columns:
        lc = c.lower()
        if "delta" not in lc:
            continue
        if all(tok.lower() in lc for tok in metric_tokens):
            hits.append(c)
    if len(hits) != 1:
        raise RuntimeError(
            f"Cannot uniquely resolve {label} delta column: hits={hits}, "
            f"columns={list(df.columns)}"
        )
    return hits[0]


def recompute_promise_utility_from_released_artifacts(
    root: Path,
    canonical: Dict[str, Any],
    replay_root: Path,
) -> Dict[str, Any]:
    """
    Public Level-A PROMISE12 utility replay.

    Point utility is freshly reconstructed from CM6 row-level outcomes using
    pooled voxel counts per (family, patient). The matched-random paper means are
    freshly recomputed from the frozen 2000-replicate CM7B paired-bootstrap table.

    This avoids redistributing CM7B's large prediction-mask caches while still
    recomputing the paper-facing statistics from released intermediate artifacts.
    """
    outcomes_path = require(
        root
        / "outputs"
        / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
        / "CM6_PROMISE12_OUTCOMES.csv"
    )
    policy_path = require(
        root
        / "outputs"
        / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
        / "CM6_LOCKED_POLICY_EVALUATION.csv"
    )
    reps_path = require(
        root
        / "outputs"
        / "Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1_v1"
        / "CM7B_PAIRED_PATIENT_BOOTSTRAP_REPLICATES.csv"
    )

    df = pd.read_csv(outcomes_path)
    policy = pd.read_csv(policy_path)
    reps = pd.read_csv(reps_path)

    family_col = _resolve_exact_or_tokens(
        df, ("family", "model_family"), ("family",), "PROMISE utility family"
    )
    case_col = _resolve_promise_case_col(df)
    score_col = _resolve_exact_or_tokens(
        df,
        ("risk_score", "frozen_safety_probability"),
        ("risk", "score"),
        "PROMISE utility risk score",
    )

    required_count_cols = (
        "source_tp", "source_fp", "source_fn",
        "tent_tp", "tent_fp", "tent_fn",
    )
    missing = [c for c in required_count_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"CM6 outcomes missing pooled-count columns needed for patient 3D Dice: {missing}"
        )

    # Family-specific locked SOURCE threshold used by CM6/CM7B.
    if not {"policy", "scope", "threshold"}.issubset(policy.columns):
        raise RuntimeError(
            "CM6_LOCKED_POLICY_EVALUATION.csv lacks policy/scope/threshold."
        )
    fixed = policy.loc[
        policy["policy"].astype(str).eq("FIXED_SOURCE_THRESHOLD")
    ].copy()
    if len(fixed) == 0:
        raise RuntimeError("No FIXED_SOURCE_THRESHOLD rows in CM6 policy table.")

    tau_map = {}
    for _, r in fixed.iterrows():
        scope = str(r["scope"])
        tau = float(r["threshold"])
        if not np.isfinite(tau):
            raise RuntimeError(f"Non-finite threshold for {scope}")
        tau_map[scope] = tau

    # Resolve policy scope to family names by normalized alphanumeric identity.
    def norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    norm_tau = {norm_name(k): v for k, v in tau_map.items()}
    family_tau = {}
    for fam in sorted(df[family_col].astype(str).unique()):
        nf = norm_name(fam)
        exact = [v for k, v in norm_tau.items() if k == nf]
        if len(exact) == 1:
            family_tau[fam] = exact[0]
            continue
        fuzzy = [v for k, v in norm_tau.items() if k in nf or nf in k]
        if len(fuzzy) != 1:
            raise RuntimeError(
                f"Cannot map PROMISE family {fam!r} to locked policy scopes {list(tau_map)}"
            )
        family_tau[fam] = fuzzy[0]

    patient_rows = []
    for (fam, case_key), g in df.groupby(
        [family_col, case_col],
        sort=False,
    ):
        fam = str(fam)
        s = pd.to_numeric(g[score_col], errors="raise").to_numpy(float)
        tau = float(family_tau[fam])
        adapt = s < tau

        source_tp = g["source_tp"].to_numpy(dtype=np.int64)
        source_fp = g["source_fp"].to_numpy(dtype=np.int64)
        source_fn = g["source_fn"].to_numpy(dtype=np.int64)

        tent_tp = g["tent_tp"].to_numpy(dtype=np.int64)
        tent_fp = g["tent_fp"].to_numpy(dtype=np.int64)
        tent_fn = g["tent_fn"].to_numpy(dtype=np.int64)

        deployed_tp = np.where(adapt, tent_tp, source_tp)
        deployed_fp = np.where(adapt, tent_fp, source_fp)
        deployed_fn = np.where(adapt, tent_fn, source_fn)

        patient_rows.append({
            "family": fam,
            "case_key": str(case_key),
            "source_dice_3d": _pooled_binary_dice_from_counts(
                source_tp, source_fp, source_fn
            ),
            "tent_all_dice_3d": _pooled_binary_dice_from_counts(
                tent_tp, tent_fp, tent_fn
            ),
            "safettta_dice_3d": _pooled_binary_dice_from_counts(
                deployed_tp, deployed_fp, deployed_fn
            ),
            "adaptation_coverage": float(np.mean(adapt)),
            "threshold": tau,
        })

    patient_df = pd.DataFrame(patient_rows)
    expected_patient_family_rows = (
        int(df[family_col].astype(str).nunique())
        * int(df[case_col].astype(str).nunique())
    )
    if len(patient_df) != expected_patient_family_rows:
        raise RuntimeError(
            f"Unexpected patient-family utility rows: {len(patient_df)} "
            f"expected={expected_patient_family_rows}"
        )

    source_mean = float(patient_df["source_dice_3d"].mean())
    tent_all_mean = float(patient_df["tent_all_dice_3d"].mean())
    safettta_mean = float(patient_df["safettta_dice_3d"].mean())

    slice_delta_col = _resolve_delta_metric_column(
        reps,
        ("slice", "deployed", "dice"),
        "CM7B mean slice deployed Dice",
    )
    harm_delta_col = _resolve_delta_metric_column(
        reps,
        ("prevented", "harm"),
        "CM7B prevented HARM fraction",
    )

    slice_delta = pd.to_numeric(
        reps[slice_delta_col], errors="raise"
    ).to_numpy(float)
    harm_delta = pd.to_numeric(
        reps[harm_delta_col], errors="raise"
    ).to_numpy(float)

    if len(reps) != 2000:
        raise RuntimeError(f"CM7B replicate table expected 2000 rows, got {len(reps)}")
    if not np.all(np.isfinite(slice_delta)) or not np.all(np.isfinite(harm_delta)):
        raise RuntimeError("CM7B replicate deltas contain non-finite values.")

    slice_gain = float(np.mean(slice_delta))
    harm_gain = float(np.mean(harm_delta))

    values = {
        "PROMISE_SAFE_PATIENT_DICE": safettta_mean,
        "PROMISE_SOURCE_PATIENT_DICE": source_mean,
        "PROMISE_ADAPTALL_PATIENT_DICE": tent_all_mean,
        "PROMISE_RANDOM_SLICE_DICE_GAIN": slice_gain,
        "PROMISE_RANDOM_PREVENTED_HARM_GAIN": harm_gain,
    }
    results = {
        aid: _anchor_compare(canonical, aid, value)
        for aid, value in values.items()
    }
    status = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"

    out_dir = replay_root / "DIRECT_PROMISE_UTILITY"
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(
        out_dir / "PROMISE_PATIENT_3D_UTILITY_RECOMPUTE.csv",
        index=False,
    )

    report = {
        "status": status,
        "mode": (
            "FRESH_POINT_UTILITY_FROM_CM6_ROW_COUNTS_PLUS_"
            "FRESH_BOOTSTRAP_SUMMARY_FROM_FROZEN_CM7B_REPLICATES"
        ),
        "outcomes": str(outcomes_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "bootstrap_replicates": str(reps_path),
        "bootstrap_replicates_sha256": sha256_file(reps_path),
        "bootstrap_replicates_rows": int(len(reps)),
        "family_thresholds": family_tau,
        "patient_family_rows": int(len(patient_df)),
        "source_patient_3d_dice": source_mean,
        "tent_all_patient_3d_dice": tent_all_mean,
        "safettta_patient_3d_dice": safettta_mean,
        "random_slice_dice_gain_mean": slice_gain,
        "random_prevented_harm_gain_mean": harm_gain,
        "random_slice_dice_gain_ci95": [
            float(np.percentile(slice_delta, 2.5)),
            float(np.percentile(slice_delta, 97.5)),
        ],
        "random_prevented_harm_gain_ci95": [
            float(np.percentile(harm_delta, 2.5)),
            float(np.percentile(harm_delta, 97.5)),
        ],
        "results": results,
        "boundary": (
            "The public Level-A package does not rerun CM7B's heavy mask-level "
            "matched-random construction. Instead it recomputes point patient 3D "
            "utility from CM6 row-level voxel counts and recomputes bootstrap "
            "summary statistics from the released 2000-replicate paired table. "
            "The exact historical CM7B stage itself already passed author-side "
            "PASS_NUMERIC_REPLAY."
        ),
    }
    (out_dir / "PROMISE_UTILITY_DIRECT_RECOMPUTE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def recompute_prostate(root: Path, canonical: Dict[str, Any]) -> Dict[str, Any]:
    p = require(
        root / "outputs"
        / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1"
        / "CM4B_SOURCE_SAFETY_OOF_SCORES.csv"
    )
    df = pd.read_csv(p)
    required = {"family", "harmful", "score_M2", "score_CondDINO", "score_Final"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Prostate replay missing columns: {sorted(missing)}")
    if len(df) != 10659:
        raise RuntimeError(f"Prostate replay rows={len(df)} != 10659")

    canonical_rows = {r["anchor_id"]: r for r in canonical["final_rows"]}
    specs = {
        "PROSTATE_M2_AUROC": "score_M2",
        "PROSTATE_CONDDINO_AUROC": "score_CondDINO",
        "PROSTATE_FINAL_AUROC": "score_Final",
    }

    results = {}
    for aid, score_col in specs.items():
        family_vals = {}
        for fam, g in df.groupby("family", sort=True):
            y = g["harmful"].astype(int).to_numpy()
            s = g[score_col].astype(float).to_numpy()
            family_vals[str(fam)] = float(roc_auc_score(y, s))
        value = float(np.mean(list(family_vals.values())))

        expected = float(canonical_rows[aid]["observed_value"])
        err = abs(value - expected)
        results[aid] = {
            "score_column": score_col,
            "value": value,
            "canonical_value": expected,
            "abs_error": err,
            "family_aurocs": family_vals,
            "status": "PASS" if err <= NUMERIC_ATOL else "FAIL",
        }
    return {
        "source_path": str(p),
        "source_sha256": sha256_file(p),
        "results": results,
        "status": "PASS" if all(v["status"] == "PASS" for v in results.values()) else "FAIL",
    }


def validate_gates(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    canon_path = require(
        root / "outputs" / CANONICAL_PROVENANCE[0] / CANONICAL_PROVENANCE[1]
    )
    canonical = load_json(canon_path)
    if canonical.get("gate") != "PASS_CANONICAL_NUMERIC_PROVENANCE":
        raise RuntimeError(
            f"Canonical numeric provenance not PASS: {canonical.get('gate')}"
        )

    chain_path = require(
        root / "outputs" / CHAIN_GATE[0] / CHAIN_GATE[1]
    )
    chain = load_json(chain_path)
    if chain.get("gate") != "PASS_CHAIN_ASSET_RESOLUTION":
        raise RuntimeError(f"Chain gate not PASS: {chain.get('gate')}")
    if int(chain.get("selected_stage_pass_after_canonical_resolution", -1)) != 43:
        raise RuntimeError("Paper-chain stage count is not 43/43.")

    return canonical, chain




def _assert_replay_dir_is_safe(replay_root: Path, replay_dir: Path) -> None:
    """
    Safety guard for deleting/recreating a failed *replay* stage only.
    Frozen project outputs are never eligible.
    """
    rr = replay_root.resolve()
    rd = replay_dir.resolve()
    if rd == rr:
        raise RuntimeError("Refusing to remove replay root itself.")
    if rr not in rd.parents:
        raise RuntimeError(
            f"Replay stage directory escapes replay root: root={rr} stage={rd}"
        )
    if not replay_dir.name:
        raise RuntimeError("Invalid replay stage directory.")



def _inventory_failed_replay_path(path: Path) -> Dict[str, Any]:
    """
    Create an explicit audit record BEFORE deleting a failed replay artifact.
    This is important for post-GT stages such as R14C3B: the historical script
    intentionally refuses to silently delete partial post-reveal builds.
    """
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "kind": None,
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        }

    files: List[Dict[str, Any]] = []
    total_bytes = 0

    if path.is_file():
        size = int(path.stat().st_size)
        total_bytes = size
        files.append({
            "relpath": path.name,
            "size_bytes": size,
            "sha256": sha256_file(path),
        })
        kind = "file"
    else:
        kind = "directory"
        for p in sorted(path.rglob("*"), key=lambda x: str(x).lower()):
            if not p.is_file():
                continue
            size = int(p.stat().st_size)
            total_bytes += size
            files.append({
                "relpath": str(p.relative_to(path)),
                "size_bytes": size,
                "sha256": sha256_file(p),
            })

    return {
        "path": str(path),
        "exists": True,
        "kind": kind,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _remove_replay_artifact(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def cleanup_failed_replay_stage(
    replay_root: Path,
    stage: ReplayStage,
    replay_dir: Path,
    meta_dir: Path,
    previous_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Explicitly audit and remove ONLY failed replay artifacts under replay_root.

    Known historical pattern:
      R14C3B --output-dir <root>/R14C3B
      temporary post-GT build: <root>/R14C3B__building

    We remove the final replay dir (if any) and the exact "__building" sibling
    only after writing a SHA-audited cleanup record. No frozen output path can
    pass the replay-root containment guard.
    """
    candidates = [
        replay_dir,
        replay_root / f"{stage.stage_id}__building",
    ]

    audited = []
    for p in candidates:
        _assert_replay_dir_is_safe(replay_root, p)
        audited.append(_inventory_failed_replay_path(p))

    audit = {
        "version": VERSION,
        "stage_id": stage.stage_id,
        "reason": "FAILED_REPLAY_STAGE_EXPLICIT_CLEANUP_BEFORE_RESUME",
        "previous_process_record": previous_record,
        "replay_root": str(replay_root),
        "frozen_outputs_touched": False,
        "candidate_artifacts": audited,
        "cleanup_policy": (
            "Delete only the failed replay stage directory and the exact "
            "<stage_id>__building sibling inside replay_root after auditing them."
        ),
    }

    audit_path = meta_dir / f"{stage.stage_id}__FAILED_REPLAY_CLEANUP_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for p in candidates:
        _remove_replay_artifact(p)

    remaining = [str(p) for p in candidates if p.exists()]
    if remaining:
        raise RuntimeError(
            f"{stage.stage_id}: failed replay cleanup incomplete: {remaining}"
        )

    audit["cleanup_completed"] = True
    audit["cleanup_audit_path"] = str(audit_path)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit


def run_stage(
    root: Path,
    replay_root: Path,
    stage: ReplayStage,
    sha_map: Dict[str, str],
    resume: bool,
) -> Dict[str, Any]:
    script = require(root / "code" / stage.script_name)
    frozen = require(root / "outputs" / stage.frozen_output_dir)
    check_python_syntax(script)

    current_sha = sha256_file(script).lower()
    expected_sha = stage.expected_script_sha256
    inventory_sha = sha_map.get(stage.script_name)

    if expected_sha and current_sha != expected_sha.lower():
        raise RuntimeError(
            f"{stage.stage_id}: script SHA changed.\n"
            f"expected={expected_sha}\ncurrent={current_sha}"
        )
    if inventory_sha and current_sha != inventory_sha.lower():
        raise RuntimeError(
            f"{stage.stage_id}: current script SHA differs from inventory.\n"
            f"inventory={inventory_sha}\ncurrent={current_sha}"
        )

    strategy = build_execution_strategy(root, replay_root, stage)
    if not strategy["safe"]:
        return {
            "stage_id": stage.stage_id,
            "status": "BLOCKED_NO_SAFE_OUTPUT_REDIRECTION",
            "script": str(script),
            "script_sha256": current_sha,
            "frozen_output": str(frozen),
            "execution_strategy": strategy,
            "anchors": list(stage.anchor_ids),
        }

    replay_dir = replay_root / stage.stage_id

    # IMPORTANT:
    # Several retained final scripts (e.g. R15A1) deliberately require their
    # --output-dir to NOT exist and raise FileExistsError otherwise. Therefore
    # the harness must never pre-create a stage output directory before launching
    # the historical script.
    #
    # Process metadata is stored outside the stage output tree so that freshness
    # checks do not themselves make the output directory exist.
    if replay_dir.exists() and not resume:
        return {
            "stage_id": stage.stage_id,
            "status": "BLOCKED_REPLAY_DIR_ALREADY_EXISTS",
            "replay_output": str(replay_dir),
            "anchors": list(stage.anchor_ids),
        }

    meta_dir = replay_root / "_process_records"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stage_log = replay_root / f"{stage.stage_id}.log"
    record_path = meta_dir / f"{stage.stage_id}__REPLAY_PROCESS.json"
    patch_audit = None

    cleanup_audit = None
    transient_build_dir = replay_root / f"{stage.stage_id}__building"

    if resume and record_path.exists():
        old_record = load_json(record_path)
        prior_rc = int(old_record.get("returncode", 1))

        if prior_rc == 0:
            if not replay_dir.exists():
                return {
                    "stage_id": stage.stage_id,
                    "status": "BLOCKED_SUCCESS_RECORD_BUT_OUTPUT_MISSING",
                    "replay_output": str(replay_dir),
                    "process_record": str(record_path),
                    "anchors": list(stage.anchor_ids),
                }
            # Successful historical process: do not rerun; re-compare its fresh
            # outputs below.
            rc = 0
            executed = False
        else:
            # Explicitly audit and remove failed replay artifacts. This handles
            # R14C3B's sibling post-GT "<stage>__building" directory rather than
            # silently deleting it.
            cleanup_audit = cleanup_failed_replay_stage(
                replay_root=replay_root,
                stage=stage,
                replay_dir=replay_dir,
                meta_dir=meta_dir,
                previous_record=old_record,
            )
            if record_path.exists():
                record_path.unlink()
            executed = True
            rc = None
    else:
        if replay_dir.exists() or transient_build_dir.exists():
            return {
                "stage_id": stage.stage_id,
                "status": "BLOCKED_REPLAY_ARTIFACT_WITHOUT_PROCESS_RECORD",
                "replay_output": str(replay_dir),
                "transient_build_output": str(transient_build_dir),
                "process_record": str(record_path),
                "anchors": list(stage.anchor_ids),
            }
        executed = True
        rc = None

    if executed:
        patched_copy = (
            replay_root
            / "_patched_sources"
            / f"{stage.stage_id}__{script.name}"
        )
        patch_audit = apply_portable_execution_patch(
            script=script,
            root=root,
            stage=stage,
            strategy=strategy,
            replay_dir=replay_dir,
            patched_copy=patched_copy,
        )
        patch_audit_path = (
            replay_root
            / "_patched_sources"
            / f"{stage.stage_id}__PATCH_AUDIT.json"
        )
        patch_audit_path.write_text(
            json.dumps(patch_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-u",
            "-c",
            PATCH_LAUNCHER,
            str(patched_copy),
            str(script),
        ]
        if strategy["strategy"] == "CLI_OUTPUT_DIRECTORY":
            cmd += [
                str(strategy["output_arg"]),
                str(replay_dir),
            ]
        elif strategy["strategy"] == "STATIC_OUTPUT_LITERAL_PATCH":
            pass
        else:
            raise AssertionError(strategy)

        t0 = time.time()
        rc = stream_subprocess(cmd, stage_log, root / "code")
        elapsed = time.time() - t0
        record = {
            "stage_id": stage.stage_id,
            "original_script": str(script),
            "original_script_sha256": current_sha,
            "execution_strategy": strategy["strategy"],
            "output_arg": strategy.get("output_arg"),
            "patch_audit": patch_audit,
            "failed_replay_cleanup_audit": cleanup_audit,
            "command": cmd,
            "returncode": rc,
            "elapsed_seconds": elapsed,
        }
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        executed = True

    if rc != 0:
        return {
            "stage_id": stage.stage_id,
            "status": "FAIL_PROCESS",
            "returncode": rc,
            "script": str(script),
            "script_sha256": current_sha,
            "execution_strategy": strategy,
            "replay_output": str(replay_dir),
            "log": str(stage_log),
            "anchors": list(stage.anchor_ids),
        }

    if not replay_dir.exists() or not replay_dir.is_dir():
        return {
            "stage_id": stage.stage_id,
            "status": "FAIL_PROCESS_DID_NOT_CREATE_OUTPUT_DIR",
            "script": str(script),
            "script_sha256": current_sha,
            "execution_strategy": strategy,
            "replay_output": str(replay_dir),
            "log": str(stage_log),
            "anchors": list(stage.anchor_ids),
        }

    pairs = common_summary_pairs(frozen, replay_dir, stage.summary_name_tokens)
    if not pairs:
        return {
            "stage_id": stage.stage_id,
            "status": "FAIL_NO_COMMON_SUMMARY_ARTIFACTS",
            "script": str(script),
            "execution_strategy": strategy,
            "replay_output": str(replay_dir),
            "frozen_output": str(frozen),
            "anchors": list(stage.anchor_ids),
        }

    comparisons = []
    for fp, rp in pairs:
        try:
            comparisons.append(compare_pair(fp, rp))
        except Exception as e:
            comparisons.append({
                "status": "COMPARE_ERROR",
                "frozen_path": str(fp),
                "replay_path": str(rp),
                "error": repr(e),
            })

    comparable = [
        c for c in comparisons
        if c["status"] not in {
            "NO_NUMERIC_COLUMNS",
            "NO_COMMON_NUMERIC_KEYS",
            "SKIP",
        }
    ]
    bad = [c for c in comparable if c["status"] != "PASS"]

    if not comparable:
        status = "FAIL_NO_NUMERIC_SUMMARY_COMPARISON"
    elif bad:
        status = "FAIL_NUMERIC_MISMATCH"
    else:
        status = "PASS"

    return {
        "stage_id": stage.stage_id,
        "status": status,
        "script": str(script),
        "script_sha256": current_sha,
        "inventory_sha256": inventory_sha,
        "execution_strategy": strategy["strategy"],
        "output_arg": strategy.get("output_arg"),
        "patch_audit": patch_audit,
        "failed_replay_cleanup_audit": cleanup_audit,
        "executed_this_run": executed,
        "frozen_output": str(frozen),
        "replay_output": str(replay_dir),
        "anchors": list(stage.anchor_ids),
        "summary_pairs_compared": len(comparisons),
        "numeric_summary_pairs": len(comparable),
        "comparisons": comparisons,
        "log": str(stage_log),
    }



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Static-only preflight: gates, SHA/syntax, CLI output arg or safe output-literal patch.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed fresh replay stage directories.",
    )
    ap.add_argument(
        "--replay-root",
        type=Path,
        default=None,
        help=(
            "Optional replay output root. Use with --resume to continue an "
            "incomplete prior replay directory, e.g. the existing fix4 run."
        ),
    )
    ap.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional subset of stage IDs. Default: all deterministic replay stages.",
    )
    args = ap.parse_args()

    root = args.root
    default_replay_root = (
        root / "outputs" / "Q1_SAFETTA_public_paper_stat_replay_v3"
    )
    replay_root = args.replay_root if args.replay_root is not None else default_replay_root

    # Replay output must stay under the project's outputs directory. This protects
    # frozen assets from accidental --replay-root misuse.
    outputs_root_resolved = (root / "outputs").resolve()
    replay_root_resolved = replay_root.resolve()
    if outputs_root_resolved not in replay_root_resolved.parents:
        raise RuntimeError(
            f"--replay-root must be inside {root / 'outputs'}; got {replay_root}"
        )

    # The top-level replay root may be created by preflight, but stage output
    # directories must remain absent until each retained script creates them.
    replay_root.mkdir(parents=True, exist_ok=True)

    canonical, chain = validate_gates(root)
    canonical_rows = {r["anchor_id"]: r for r in canonical["final_rows"]}
    if len(canonical_rows) != 44:
        raise RuntimeError(f"Canonical anchor count={len(canonical_rows)} != 44")

    sha_map = inventory_sha_map(root)

    selected = list(STAGES)
    if args.stages:
        requested = set(args.stages)
        known = {s.stage_id for s in STAGES}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}; known={sorted(known)}")
        selected = [s for s in STAGES if s.stage_id in requested]

    print("===== SAFETTA PUBLIC PAPER-STATISTIC REPLAY V3 =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Frozen outputs modified: NO")
    print("Chain gate:", chain["gate"], "43/43")
    print("Canonical numeric provenance:", canonical["gate"], "44/44")
    print("Replay root:", replay_root)
    print("Selected executable stages:", [s.stage_id for s in selected])
    print("Direct deterministic recomputes: PROSTATE, SENSITIVITY")
    print()

    # Static preflight only. No project script is executed here.
    side_effect_guard = verify_v1_preflight_side_effect_guard(root, canonical)
    (replay_root / "V1_PREFLIGHT_SIDE_EFFECT_GUARD.json").write_text(
        json.dumps(side_effect_guard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if side_effect_guard["gate"] != "PASS":
        print("V1_PRELIGHT_SIDE_EFFECT_GUARD=BLOCKED")
        print("See:", replay_root / "V1_PREFLIGHT_SIDE_EFFECT_GUARD.json")
        return 2

    preflight_rows = []
    preflight_blocked = []
    for s in selected:
        script = root / "code" / s.script_name
        row = {
            "stage_id": s.stage_id,
            "script": str(script),
            "exists": script.exists(),
            "syntax_ok": False,
            "sha_ok": False,
            "execution_strategy": None,
            "output_arg": None,
            "static_patch_safe": None,
        }
        if script.exists():
            try:
                check_python_syntax(script)
                row["syntax_ok"] = True
                got_sha = sha256_file(script).lower()
                row["script_sha256"] = got_sha
                expected = s.expected_script_sha256
                inv = sha_map.get(s.script_name)
                sha_ok = True
                if expected:
                    sha_ok &= got_sha == expected.lower()
                if inv:
                    sha_ok &= got_sha == inv.lower()
                row["sha_ok"] = bool(sha_ok)

                strategy = build_execution_strategy(root, replay_root, s)
                row["execution_strategy"] = strategy["strategy"]
                row["output_arg"] = strategy.get("output_arg")
                row["strategy_safe"] = strategy["safe"]
                if strategy.get("patch_plan") is not None:
                    row["static_patch_safe"] = strategy["patch_plan"]["safe"]
                    row["static_patch_reason"] = strategy["patch_plan"]["reason"]
                    row["static_patch_candidate_count"] = strategy["patch_plan"]["candidate_count"]
                    row["static_patch_non_output_literal_count"] = (
                        strategy["patch_plan"]["non_output_literal_count"]
                    )
            except Exception as e:
                row["error"] = repr(e)

        if not (
            row["exists"]
            and row["syntax_ok"]
            and row["sha_ok"]
            and row.get("strategy_safe") is True
        ):
            preflight_blocked.append(s.stage_id)
        preflight_rows.append(row)

    (replay_root / "REPLAY_PREFLIGHT.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "preflight_mode": "STATIC_ONLY_NO_PROJECT_SCRIPT_EXECUTION",
                "v1_preflight_side_effect_guard": side_effect_guard,
                "rows": preflight_rows,
                "blocked": preflight_blocked,
                "gate": "PASS" if not preflight_blocked else "BLOCKED",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if preflight_blocked:
        print("PREFLIGHT=BLOCKED")
        print("Blocked stages:", preflight_blocked)
        print("See:", replay_root / "REPLAY_PREFLIGHT.json")
        return 2

    print("PREFLIGHT=PASS")
    print("Preflight mode=STATIC_ONLY_NO_PROJECT_SCRIPT_EXECUTION")
    print("Stage output directories pre-created: NO")
    print("V1 side-effect guard=PASS")
    if args.preflight_only:
        print("Decision=READY_FOR_PUBLIC_PAPER_STATISTIC_REPLAY")
        return 0

    # Prostate direct deterministic recomputation.
    prostate = recompute_prostate(root, canonical)
    (replay_root / "PROSTATE_FRESH_RECOMPUTE.json").write_text(
        json.dumps(prostate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sensitivity = recompute_sensitivity(root, replay_root)

    polypgen_direct = recompute_polypgen_from_frozen_panel(
        root, canonical, replay_root
    )
    sun_direct = recompute_sun_from_frozen_panel(
        root, canonical, replay_root
    )
    promise_direct = recompute_promise_from_frozen_outcomes(
        root, canonical, replay_root
    )
    promise_utility_direct = recompute_promise_utility_from_released_artifacts(
        root, canonical, replay_root
    )

    iterator = selected
    if tqdm is not None:
        iterator = tqdm(selected, desc="Replay paper analyses", unit="stage", dynamic_ncols=True)

    stage_results = []
    for s in iterator:
        print("\n" + "=" * 88)
        print("REPLAY STAGE:", s.stage_id)
        print("=" * 88)
        result = run_stage(root, replay_root, s, sha_map, args.resume)
        stage_results.append(result)
        print("STAGE RESULT:", s.stage_id, result["status"])

        if result["status"] != "PASS":
            # Stop immediately; a later stage must not hide an earlier failure.
            break

    all_selected_pass = (
        len(stage_results) == len(selected)
        and all(r["status"] == "PASS" for r in stage_results)
    )
    prostate_pass = prostate["status"] == "PASS"

    # Anchor coverage accounting.
    anchor_status: Dict[str, str] = {}

    for r in stage_results:
        state = "PASS_REPLAY_STAGE" if r["status"] == "PASS" else "FAIL_REPLAY_STAGE"
        for aid in r.get("anchors", []):
            anchor_status[aid] = state

    for aid in PROSTATE_ANCHORS:
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE"
            if prostate["results"][aid]["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE"
        )

    for aid, r in sensitivity["anchor_results"].items():
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE_SENSITIVITY"
            if sensitivity["status"] == "PASS" and r["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE_SENSITIVITY"
        )

    for aid, r in polypgen_direct["results"].items():
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE_POLYPGEN_PANEL"
            if polypgen_direct["status"] == "PASS" and r["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE_POLYPGEN_PANEL"
        )

    for aid, r in sun_direct["results"].items():
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE_SUN_PANEL"
            if sun_direct["status"] == "PASS" and r["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE_SUN_PANEL"
        )

    for aid, r in promise_direct["results"].items():
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE_PROMISE_OUTCOMES"
            if promise_direct["status"] == "PASS" and r["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE_PROMISE_OUTCOMES"
        )

    for aid, r in promise_utility_direct["results"].items():
        anchor_status[aid] = (
            "PASS_FRESH_RECOMPUTE_PROMISE_UTILITY"
            if promise_utility_direct["status"] == "PASS" and r["status"] == "PASS"
            else "FAIL_FRESH_RECOMPUTE_PROMISE_UTILITY"
        )

    # Runtime is not an exact replay metric; verify canonical provenance only.
    for aid in RUNTIME_ANCHORS:
        row = canonical_rows[aid]
        source_path = Path(row["source_path"])
        if not source_path.is_file():
            anchor_status[aid] = "FAIL_FROZEN_RUNTIME_PROVENANCE"
        else:
            got = sha256_file(source_path)
            expected = row.get("source_sha256")
            anchor_status[aid] = (
                "PASS_FROZEN_RUNTIME_PROVENANCE"
                if expected and got.lower() == str(expected).lower()
                else "FAIL_FROZEN_RUNTIME_PROVENANCE"
            )

    # If a subset was requested, do not claim full replay.
    full_stage_selection = {s.stage_id for s in selected} == {s.stage_id for s in STAGES}

    all_44_accounted = set(anchor_status) == set(canonical_rows)
    all_anchor_states_pass = (
        all_44_accounted
        and all(v.startswith("PASS_") for v in anchor_status.values())
    )

    sensitivity_pass = sensitivity["status"] == "PASS"
    polypgen_pass = polypgen_direct["status"] == "PASS"
    sun_pass = sun_direct["status"] == "PASS"
    promise_pass = promise_direct["status"] == "PASS"
    promise_utility_pass = promise_utility_direct["status"] == "PASS"

    if (
        full_stage_selection
        and all_selected_pass
        and prostate_pass
        and sensitivity_pass
        and polypgen_pass
        and sun_pass
        and promise_pass
        and promise_utility_pass
        and all_anchor_states_pass
    ):
        gate = "PASS_PUBLIC_PAPER_STATISTIC_REPLAY"
    else:
        gate = "BLOCKED_PUBLIC_PAPER_STATISTIC_REPLAY"

    report = {
        "version": VERSION,
        "root": str(root),
        "chain_gate": chain["gate"],
        "canonical_provenance_gate": canonical["gate"],
        "selected_stage_ids": [s.stage_id for s in selected],
        "full_stage_selection": full_stage_selection,
        "stage_results": stage_results,
        "prostate_fresh_recompute": prostate,
        "sensitivity_fresh_recompute": sensitivity,
        "polypgen_panel_fresh_recompute": polypgen_direct,
        "sun_panel_fresh_recompute": sun_direct,
        "promise_outcomes_fresh_recompute": promise_direct,
        "promise_utility_fresh_recompute": promise_utility_direct,
        "public_level_a_mode": (
            "R10L0, R15A1 and R15B1 execute retained historical analysis scripts. "
            "PolypGen R10L3C, SUN R14C3B, PROMISE CM6 and PROMISE CM7B "
            "paper-facing statistics are freshly recomputed from released frozen "
            "row-level panels/outcomes or bootstrap-replicate tables to avoid "
            "redistributing raw GT and large prediction/mask arrays."
        ),
        "author_side_full_historical_replay_provenance": (
            "The complete historical analysis-stage replay was separately validated "
            "on the author workspace with PASS_NUMERIC_REPLAY and retained in the "
            "release provenance."
        ),
        "runtime_policy": (
            "Frozen R15D1 benchmark provenance/hash checked; fresh wall-clock "
            "latency is intentionally excluded from bit-exact numeric replay."
        ),
        "subprocess_text_io_policy": (
            "Historical analysis scripts are executed with PYTHONUTF8=1 and "
            "PYTHONIOENCODING=utf-8; parent subprocess decoding is explicitly "
            "UTF-8. This changes console/text encoding only, not scientific logic."
        ),
        "portability_policy": (
            "Retained historical scripts are SHA-verified unchanged, then an "
            "execution-only temporary copy replaces only the exact legacy project "
            "root path and, where required, the audited output-directory literal. "
            "Scientific/statistical code is not changed."
        ),
        "anchor_status": anchor_status,
        "anchor_accounted_count": len(anchor_status),
        "canonical_anchor_count": len(canonical_rows),
        "all_44_accounted": all_44_accounted,
        "gate": gate,
        "reproducibility_boundary": (
            "PASS_PUBLIC_PAPER_STATISTIC_REPLAY proves the 44 paper anchors are "
            "reproduced/verified from released frozen intermediate artifacts. Heavy "
            "post-GT reconstruction stages are replaced publicly by fresh statistic "
            "recomputation from their frozen row-level panels; the separate author-side "
            "full historical analysis replay remains archived as provenance."
        ),
    }

    (replay_root / "NUMERIC_REPLAY_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = replay_root / "numeric_replay_stage_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "stage_id", "status", "script", "script_sha256", "execution_strategy",
            "frozen_output", "replay_output",
            "summary_pairs_compared", "numeric_summary_pairs", "log",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in stage_results:
            w.writerow(r)

    lines = [
        "===== SAFETTA NUMERIC REPLAY SUMMARY =====",
        f"Version: {VERSION}",
        f"Chain gate: {chain['gate']} (43/43)",
        f"Canonical provenance: {canonical['gate']} (44/44)",
        f"Prostate fresh recompute: {prostate['status']}",
        f"Sensitivity fresh recompute: {sensitivity['status']}",
        f"PolypGen panel fresh recompute: {polypgen_direct['status']}",
        f"SUN dual-action panel fresh recompute: {sun_direct['status']}",
        f"PROMISE outcomes fresh recompute: {promise_direct['status']}",
        f"PROMISE utility fresh recompute: {promise_utility_direct['status']}",
        "",
        "Replay stages:",
    ]
    for r in stage_results:
        lines.append(
            f"  {r['stage_id']}: {r['status']} "
            f"numeric_summary_pairs={r.get('numeric_summary_pairs', 0)}"
        )
    lines += [
        "",
        f"Anchors accounted: {len(anchor_status)}/44",
        f"All anchor states PASS: {all_anchor_states_pass}",
        "",
        "Runtime boundary:",
        "  Frozen R15D1 benchmark SHA/protocol is verified.",
        "  Fresh timing is not required to be bit-identical.",
        "",
        f"GATE={gate}",
    ]

    if gate != "PASS_PUBLIC_PAPER_STATISTIC_REPLAY":
        lines += [
            "",
            "Do not publish the minimal public replay package yet.",
            "Inspect NUMERIC_REPLAY_REPORT.json and the first failed stage log.",
        ]
    else:
        lines += [
            "",
            "Decision=READY_FOR_MINIMAL_PUBLIC_RELEASE_VALIDATION",
        ]

    (replay_root / "NUMERIC_REPLAY_SUMMARY.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    return 0 if gate == "PASS_PUBLIC_PAPER_STATISTIC_REPLAY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
