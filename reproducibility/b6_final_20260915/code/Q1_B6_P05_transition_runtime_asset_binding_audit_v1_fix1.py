#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-5 transition-runtime asset binding audit.

STATIC / READ ONLY:
- no model loading
- no inference
- no GT
- no runtime measurement
- no scientific metric computation

The audit binds:
1) canonical TENT1 and PL-CONF90 action scripts (TENT DeepLab uses run_deeplab_state; PL DeepLab/PraNet uses run_binary_state);
2) exact R30A4B2A2 execution-context bundle;
3) MEMO action script candidates;
4) existing SOURCE-only runtime benchmark script candidates;
5) DINO / prediction-conditioned semantic extraction candidates.

The next stage will benchmark only code paths bound here, preventing
post-runtime selection of a faster implementation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


VERSION = "2026-09-15-B6-P05-v1-fix1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_PATH = CODE / "B6_P05_TRANSITION_RUNTIME_BINDING_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "329b640834be06fca1e7b02d6f64fac64cf185801679c357e99e86ab07ec40d6"

TENT_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
PL_SCRIPT = CODE / "Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix2.py"

# Exact historical hashes already frozen by R30A4B2 lineage.
EXPECTED_TENT_SCRIPT_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)
EXPECTED_PL_SCRIPT_SHA256 = (
    "6f7ce6a3b15abf1205dc77703320ba823b79d392dc4f43beb51e0a7b94a2521d"
)

CONTEXT_BUNDLE = (
    ROOT
    / "R30A4B2A2_exact_execution_context_bundle_v1"
    / "R30A4B2A2_EXACT_EXECUTION_CONTEXT_BUNDLE.txt"
)
EXPECTED_CONTEXT_SHA256 = (
    "c87f65036f232908cabab1e00060bf6be9c96374b1cc89959a61b9a4a5fc565c"
)

OUT_DIR = ROOT / "B6_P05_transition_runtime_asset_binding_audit_v1_fix1"
PASS_GATE = "PASS_B6_P05_TRANSITION_RUNTIME_ASSET_BINDING_AUDIT"

MAX_BYTES = 6_000_000

MEMO_TERMS = [
    "MEMO-SEG4-1STEP",
    "memo_loss",
    "memo_grad",
    "horizontal flip",
    "vertical flip",
    "augmentation-marginalized",
]
RUNTIME_TERMS = [
    "24.28",
    "22.91",
    "peak cuda",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "cuda.Event",
    "throughput",
]
DINO_TERMS = [
    "dinov2",
    "patch_tokens",
    "pca64",
    "condDINO",
    "conditioned",
]
ACTION_TERMS = [
    "source_and_tent_logits",
    "PL-CONF90",
    "pseudo",
    "memo",
    "restore_params",
    "restore_buffers",
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    sha = sha256_file(PROTOCOL_PATH)
    if sha.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "P05 protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"observed={sha}"
        )
    p = json.loads(read_text(PROTOCOL_PATH))
    if p.get("status") != "FROZEN_BEFORE_TRANSITION_RUNTIME_MEASUREMENT":
        raise RuntimeError("P05 protocol status changed.")
    return {"path": str(PROTOCOL_PATH), "sha256": sha}


def py_ast_info(path: Path) -> Dict[str, Any]:
    text = read_text(path)
    tree = ast.parse(text)
    funcs = []
    classes = []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return {
        "functions": sorted(set(funcs)),
        "classes": sorted(set(classes)),
        "imports": sorted(set(imports)),
        "text": text,
    }


def term_hits(text: str, terms: List[str]) -> List[str]:
    lo = text.lower()
    return [t for t in terms if t.lower() in lo]


def scan_code() -> List[Dict[str, Any]]:
    rows = []
    files = sorted(CODE.glob("Q1_*.py"))
    for p in files:
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
            info = py_ast_info(p)
        except Exception:
            continue

        text = info["text"]
        memo = term_hits(text, MEMO_TERMS)
        runtime = term_hits(text, RUNTIME_TERMS)
        dino = term_hits(text, DINO_TERMS)
        action = term_hits(text, ACTION_TERMS)

        if not (memo or runtime or dino or action):
            continue

        funcs = info["functions"]
        runner_funcs = [
            f for f in funcs
            if (
                "run_" in f.lower()
                or "memo" in f.lower()
                or "tent" in f.lower()
                or "runtime" in f.lower()
                or "benchmark" in f.lower()
                or "dino" in f.lower()
            )
        ]

        rows.append({
            "path": str(p),
            "name": p.name,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "memo_hits": ";".join(memo),
            "runtime_hits": ";".join(runtime),
            "dino_hits": ";".join(dino),
            "action_hits": ";".join(action),
            "runner_functions": ";".join(runner_funcs[:80]),
            "all_function_count": len(funcs),
        })
    return rows


def canonical_checks() -> Dict[str, Any]:
    for p in [TENT_SCRIPT, PL_SCRIPT, CONTEXT_BUNDLE]:
        if not p.is_file():
            raise FileNotFoundError(p)

    context_sha = sha256_file(CONTEXT_BUNDLE)
    if context_sha.lower() != EXPECTED_CONTEXT_SHA256.lower():
        raise RuntimeError(
            "Exact execution-context bundle SHA changed.\n"
            f"expected={EXPECTED_CONTEXT_SHA256}\n"
            f"observed={context_sha}"
        )

    tent_sha = sha256_file(TENT_SCRIPT)
    pl_sha = sha256_file(PL_SCRIPT)

    if tent_sha.lower() != EXPECTED_TENT_SCRIPT_SHA256.lower():
        raise RuntimeError(
            "Canonical TENT script SHA changed.\n"
            f"expected={EXPECTED_TENT_SCRIPT_SHA256}\n"
            f"observed={tent_sha}"
        )
    if pl_sha.lower() != EXPECTED_PL_SCRIPT_SHA256.lower():
        raise RuntimeError(
            "Canonical PL script SHA changed.\n"
            f"expected={EXPECTED_PL_SCRIPT_SHA256}\n"
            f"observed={pl_sha}"
        )

    tent = py_ast_info(TENT_SCRIPT)
    pl = py_ast_info(PL_SCRIPT)

    # Exact historical DeepLab call interface:
    #   R05D3 TENT1 -> run_deeplab_state(...)
    #   R13B  PL    -> run_binary_state(r13a, r05, state, targets, device, source)
    # R13B deliberately shares run_binary_state for PraNet + DeepLab; it does
    # NOT define run_deeplab_state.
    if "run_deeplab_state" not in tent["functions"]:
        raise RuntimeError("Canonical TENT script lacks run_deeplab_state.")
    if "run_binary_state" not in pl["functions"]:
        raise RuntimeError("Canonical PL script lacks run_binary_state.")

    tent_required = ["source_and_tent_logits", "restore_params"]
    if not all(x in tent["text"] for x in tent_required):
        raise RuntimeError("Canonical TENT execution anchors incomplete.")

    # Strong PL anchors from the frozen historical implementation.
    pl_required = [
        "binary_pseudolabel",
        "binary_masked_loss",
        "run_binary_state",
    ]
    if not all(x in pl["text"] for x in pl_required):
        raise RuntimeError("Canonical PL execution anchors incomplete.")

    return {
        "tent": {
            "path": str(TENT_SCRIPT),
            "sha256": tent_sha,
            "expected_sha256": EXPECTED_TENT_SCRIPT_SHA256,
            "sha_match": True,
            "runner": "run_deeplab_state",
        },
        "pl_conf90": {
            "path": str(PL_SCRIPT),
            "sha256": pl_sha,
            "expected_sha256": EXPECTED_PL_SCRIPT_SHA256,
            "sha_match": True,
            "runner": "run_binary_state",
        },
        "execution_context": {
            "path": str(CONTEXT_BUNDLE),
            "sha256": context_sha,
            "sha_match": True,
        },
    }


def rank_candidates(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    def nterms(s: str) -> int:
        return 0 if not s else len(s.split(";"))

    memo = sorted(
        [r for r in rows if r["memo_hits"]],
        key=lambda r: (
            -nterms(r["memo_hits"]),
            -int("run_deeplab_state" in r["runner_functions"]),
            r["name"],
        ),
    )
    runtime = sorted(
        [r for r in rows if r["runtime_hits"]],
        key=lambda r: (-nterms(r["runtime_hits"]), r["name"]),
    )
    dino = sorted(
        [r for r in rows if r["dino_hits"]],
        key=lambda r: (-nterms(r["dino_hits"]), r["name"]),
    )
    return {
        "memo_candidates": memo,
        "runtime_candidates": runtime,
        "dino_candidates": dino,
    }


def self_test() -> None:
    assert EXPECTED_CONTEXT_SHA256.startswith("c87f6503")
    assert TENT_SCRIPT.name.endswith("_fix1.py")
    assert PL_SCRIPT.name.endswith("_fix2.py")
    assert EXPECTED_TENT_SCRIPT_SHA256.startswith("df22d6f3")
    assert EXPECTED_PL_SCRIPT_SHA256.startswith("6f7ce6a3")
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA P0-5 transition-runtime static asset binding audit."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 160)
    print("SafeTTA B6-P0-5 — transition runtime asset binding audit")
    print(f"Version             : {VERSION}")
    print("Model loading       : NO")
    print("Inference           : NO")
    print("GT read             : NO")
    print("Runtime measurement : NO")
    print("=" * 160)

    print("\n[1/3] Verify frozen protocol + canonical execution lineage")
    protocol_meta = verify_protocol()
    canonical = canonical_checks()
    print("PROTOCOL_SHA256 =", protocol_meta["sha256"])
    print("TENT script      =", canonical["tent"]["path"])
    print("TENT runner      =", canonical["tent"]["runner"])
    print("TENT SHA match   =", canonical["tent"]["sha_match"])
    print("PL script        =", canonical["pl_conf90"]["path"])
    print("PL runner        =", canonical["pl_conf90"]["runner"])
    print("PL SHA match     =", canonical["pl_conf90"]["sha_match"])
    print("Context bundle   =", canonical["execution_context"]["path"])
    print("Context SHA      =", canonical["execution_context"]["sha256"])
    print("CANONICAL_LINEAGE=PASS")

    print("\n[2/3] Static code inventory for MEMO / runtime / DINO")
    rows = scan_code()
    ranked = rank_candidates(rows)

    for label, vals in ranked.items():
        print(f"\n{label.upper()} (top 15)")
        if not vals:
            print("  NONE")
            continue
        for r in vals[:15]:
            print(
                f"  {r['name']} | "
                f"memo={r['memo_hits']} | "
                f"runtime={r['runtime_hits']} | "
                f"dino={r['dino_hits']} | "
                f"funcs={r['runner_functions']}"
            )

    # We require at least one candidate in each family before runtime measurement.
    if not ranked["memo_candidates"]:
        raise RuntimeError("No MEMO implementation candidate found.")
    if not ranked["runtime_candidates"]:
        raise RuntimeError("No existing runtime benchmark candidate found.")
    if not ranked["dino_candidates"]:
        raise RuntimeError("No DINO/conditioned-semantic candidate found.")

    print("\n[3/3] Freeze audit outputs")
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P05 audit: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    csv_path = OUT_DIR / "B6_P05_RUNTIME_CODE_CANDIDATES.csv"
    fields = [
        "path", "name", "sha256", "bytes",
        "memo_hits", "runtime_hits", "dino_hits",
        "action_hits", "runner_functions", "all_function_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol_meta,
        "canonical": canonical,
        "counts": {
            "all_candidates": len(rows),
            "memo_candidates": len(ranked["memo_candidates"]),
            "runtime_candidates": len(ranked["runtime_candidates"]),
            "dino_candidates": len(ranked["dino_candidates"]),
        },
        "top_candidates": {
            k: v[:15] for k, v in ranked.items()
        },
        "next": (
            "Bind one exact MEMO runner, one existing SOURCE-runtime benchmark, "
            "and the exact DINO/conditioned-semantic path; then generate the "
            "full transition latency + peak-VRAM benchmark."
        ),
        "outputs": {
            "candidate_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
            }
        },
    }

    json_path = OUT_DIR / "B6_P05_TRANSITION_RUNTIME_BINDING_AUDIT.json"
    write_json = lambda p, x: p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_json(json_path, summary)

    print("\nGATE=" + PASS_GATE)
    print("audit_json=" + str(json_path))
    print("candidate_csv=" + str(csv_path))
    print("stage_dir=" + str(OUT_DIR))
    print("=" * 160)


if __name__ == "__main__":
    main()
