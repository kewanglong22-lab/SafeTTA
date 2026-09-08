#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_ADIC_matched6_native_dropout_feasibility_audit_v1.py

SafeTTA R17A — TEGDA-ADIC / MC-dropout feasibility audit for the exact
PolypGen matched-6 panel already used by the faithful SicTTA-CCD comparison.

SCOPE
-----
Families:
  - DeepLabV3-R50 seeds 20260817/18/19
  - PraNet        seeds 20260817/18/19

SegFormer is deliberately excluded from this matched-subset audit because its
new SOURCE regeneration did not reproduce the historical SOURCE state.

OFFICIAL TEGDA CONTRACT TO PRESERVE
-----------------------------------
From the official MICCAI 2025 repository `code/sota/adic2d.py`:
  - stochastic evaluation uses n_iter=10;
  - only existing `nn.Dropout` modules are switched to train mode;
  - the model itself is otherwise used as a no-adaptation source model;
  - the `dropout` function argument is NOT used to overwrite module p;
  - dropout hard predictions are compared against the deterministic current
    prediction;
  - the same stochastic passes also yield MC predictive uncertainty.

Therefore R17A must NOT inject arbitrary new dropout layers/rates merely to
make ADIC available. If a frozen architecture has no compatible native
`nn.Dropout`, that family is unsupported for faithful ADIC.

THIS SCRIPT
-----------
Read-only. It does NOT compute ADIC and does NOT open target GT/HARM.

It:
1) SHA-checks authoritative SafeTTA historical runners.
2) extracts the exact PraNet/DeepLab model loading, preprocessing, forward,
   source-mode and prediction helpers from the historical code;
3) scans relevant architecture/import files for native Dropout definitions;
4) inventories the six frozen SOURCE checkpoints;
5) reports whether each family has a faithful path to:
       deterministic SOURCE forward
       + native nn.Dropout stochastic forward
       + no-BN-update inference
6) emits the exact next-step contract.

Training: NO
Inference: NO
Target GT loading: NO
HARM/outcome loading: NO
Model modification: NO
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Set

VERSION = "2026-09-09-Q1-R17A-ADIC-MATCHED6-NATIVE-DROPOUT-FEASIBILITY-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_ADIC_matched6_native_dropout_feasibility_audit_v1")

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R10_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")
R10_SHA = "4cae02c30e82781c6e7d3b74b6a312974789213a9cc59a44ca99bbfb94405708"

CHECKPOINTS = [
    ("DeepLabV3-R50", 20260817, Path(r"outputs\S05_B_deeplabv3_r50_source_only_seed20260817_v1\checkpoints\best_source_val_dice.pt")),
    ("DeepLabV3-R50", 20260818, Path(r"outputs\S05_B_deeplabv3_r50_source_only_seed20260818_v1\checkpoints\best_source_val_dice.pt")),
    ("DeepLabV3-R50", 20260819, Path(r"outputs\S05_B_deeplabv3_r50_source_only_seed20260819_v1\checkpoints\best_source_val_dice.pt")),
    ("PraNet", 20260817, Path(r"outputs\S01_B_pranet_source_only_seed20260817_v1\checkpoints\best_source_val_dice.pt")),
    ("PraNet", 20260818, Path(r"outputs\S01_B_pranet_source_only_seed20260818_v1\checkpoints\best_source_val_dice.pt")),
    ("PraNet", 20260819, Path(r"outputs\S01_B_pranet_source_only_seed20260819_v1\checkpoints\best_source_val_dice.pt")),
]

TOKENS = {
    "DeepLabV3-R50": ("deeplab", "aspp", "resnet"),
    "PraNet": ("pranet", "reverse_attention", "aggregation"),
}

FUNCTION_HINTS = (
    "pranet", "deeplab", "source", "logit", "tensor", "load", "build",
    "forward", "mode", "predict", "state"
)

DROPOUT_PATTERNS = (
    "nn.Dropout(",
    "torch.nn.Dropout(",
    "nn.Dropout2d(",
    "torch.nn.Dropout2d(",
    "F.dropout(",
    "torch.nn.functional.dropout(",
)

SEARCH_ROOTS = (
    Path("code"),
    Path("release/SafeTTA_clean_test_v102"),
)

MAX_FILE_BYTES = 3 * 1024 * 1024


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


def extract_relevant_functions(path: Path, family_tokens: tuple[str, ...]) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start = int(node.lineno)
        end = int(getattr(node, "end_lineno", node.lineno))
        src = "\n".join(lines[start - 1:end])
        low = (node.name + "\n" + src).lower()

        if (
            any(tok in low for tok in family_tokens)
            and any(h in node.name.lower() or h in low for h in FUNCTION_HINTS)
        ):
            rows.append({
                "function": node.name,
                "start_line": start,
                "end_line": end,
                "source": src,
            })

    rows.sort(key=lambda r: r["start_line"])
    return rows


def scan_dropout_code(root: Path) -> List[dict]:
    rows = []
    seen: Set[str] = set()

    for rr in SEARCH_ROOTS:
        base = root / rr
        if not base.exists():
            continue

        for p in base.rglob("*.py"):
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            rp = rel(p, root)
            if rp in seen:
                continue
            seen.add(rp)

            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            low_path = rp.lower()
            family_hits = [
                family
                for family, toks in TOKENS.items()
                if any(tok in low_path or tok in txt.lower() for tok in toks)
            ]
            if not family_hits:
                continue

            matches = []
            for i, line in enumerate(txt.splitlines(), 1):
                if any(pat.lower() in line.lower() for pat in DROPOUT_PATTERNS):
                    matches.append(f"L{i}:{line.strip()[:500]}")

            if matches:
                rows.append({
                    "path": rp,
                    "families": "|".join(sorted(family_hits)),
                    "dropout_matches": " | ".join(matches[:120]),
                    "sha256": sha256_file(p),
                })

    return rows


def scan_historical_runner_dropout_handling(path: Path) -> List[dict]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for i, line in enumerate(txt.splitlines(), 1):
        low = line.lower()
        if any(tok in low for tok in [
            "dropout", "model.eval()", ".eval()", ".train()",
            "batchnorm", "track_running_stats",
        ]):
            rows.append({
                "line": i,
                "source": line.strip()[:1000],
            })
    return rows


def checkpoint_inventory(root: Path) -> List[dict]:
    rows = []
    for family, seed, rp in CHECKPOINTS:
        p = root / rp
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append({
            "model_family": family,
            "training_seed": seed,
            "path": rel(p, root),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
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


def family_support(dropout_rows: List[dict], family: str) -> dict:
    hits = [
        r for r in dropout_rows
        if family in str(r.get("families", "")).split("|")
    ]
    # Code-level evidence only. The next score runner must still instantiate
    # the exact model and require at least one native nn.Dropout module.
    return {
        "family": family,
        "code_files_with_dropout_evidence": len(hits),
        "candidate_paths": [r["path"] for r in hits[:30]],
        "decision": (
            "CANDIDATE_NATIVE_DROPOUT_ROUTE"
            if hits else
            "NO_CODE_LEVEL_NATIVE_DROPOUT_EVIDENCE"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A ADIC MATCHED-6 NATIVE-DROPOUT FEASIBILITY AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Families: DeepLabV3-R50 + PraNet")
    print("States: 6")
    print("Training: NO")
    print("Inference: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Model modification: NO")
    print()

    r05 = root / R05_REL
    r03 = root / R03_REL
    r10 = root / R10_REL

    print("[1/5] Validate historical runner lineage...")
    validate_sha(r05, R05_SHA, "R05D3")
    validate_sha(r03, R03_SHA, "R03")
    validate_sha(r10, R10_SHA, "R10L3A")

    print("[2/5] Extract exact DeepLab/PraNet model+preprocessing helpers...")
    helper_rows = []
    for family, toks in TOKENS.items():
        for script in [r05, r03, r10]:
            for row in extract_relevant_functions(script, toks):
                helper_rows.append({
                    "model_family": family,
                    "script": rel(script, root),
                    **row,
                })

    print("[3/5] Scan retained code for native Dropout definitions...")
    dropout_rows = scan_dropout_code(root)

    print("[4/5] Inventory exact six SOURCE checkpoints...")
    ckpt_rows = checkpoint_inventory(root)

    print("[5/5] Audit historical source/eval/dropout mode handling...")
    mode_rows = []
    for script in [r05, r03, r10]:
        for row in scan_historical_runner_dropout_handling(script):
            mode_rows.append({
                "script": rel(script, root),
                **row,
            })

    support = [
        family_support(dropout_rows, "DeepLabV3-R50"),
        family_support(dropout_rows, "PraNet"),
    ]

    if all(s["decision"] == "CANDIDATE_NATIVE_DROPOUT_ROUTE" for s in support):
        gate = "READY_FOR_ADIC_MATCHED6_EXACT_MODEL_INSTANTIATION_AUDIT"
    elif any(s["decision"] == "CANDIDATE_NATIVE_DROPOUT_ROUTE" for s in support):
        gate = "PARTIAL_ADIC_SUPPORT_REVIEW_FAMILY_SPECIFIC"
    else:
        gate = "STOP_ADIC_MATCHED6_NO_NATIVE_DROPOUT_EVIDENCE"

    write_csv(out_dir / "R17A_ADIC_EXACT_HELPER_CONTRACT.csv", helper_rows)
    write_csv(out_dir / "R17A_ADIC_NATIVE_DROPOUT_CODE_AUDIT.csv", dropout_rows)
    write_csv(out_dir / "R17A_ADIC_SOURCE_CHECKPOINTS.csv", ckpt_rows)
    write_csv(out_dir / "R17A_ADIC_MODE_HANDLING_AUDIT.csv", mode_rows)

    summary = {
        "version": VERSION,
        "scope": {
            "families": ["DeepLabV3-R50", "PraNet"],
            "states": 6,
            "segformer_included": False,
        },
        "official_tegda_contract": {
            "stochastic_passes": 10,
            "native_module_type": "nn.Dropout",
            "inject_new_dropout": False,
            "overwrite_dropout_p": False,
            "bn_or_norm_update": False,
            "same_passes_can_produce_mc_uncertainty": True,
        },
        "family_support": support,
        "gate": gate,
        "next": (
            "INSTANTIATE_EXACT_FROZEN_MODELS_AND_REQUIRE_NATIVE_NN_DROPOUT_"
            "PLUS_DETERMINISTIC_SOURCE_PARITY_BEFORE_SCORE_LOCK"
        ),
    }
    (out_dir / "R17A_ADIC_FEASIBILITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A ADIC MATCHED-6 NATIVE-DROPOUT FEASIBILITY REPORT =====",
        f"Version: {VERSION}",
        "Scope: PolypGen DeepLabV3-R50 + PraNet, 6 states",
        "",
        "INFORMATION BOUNDARY:",
        "Training: NO",
        "Inference: NO",
        "Target GT loading: NO",
        "HARM/outcome loading: NO",
        "Model modification: NO",
        "",
        "OFFICIAL TEGDA IMPLEMENTATION CONTRACT:",
        "- n_iter=10 stochastic passes.",
        "- Existing nn.Dropout modules are switched to train mode.",
        "- No new dropout layer is injected.",
        "- The function's `dropout` argument is not used to rewrite module p.",
        "- Deterministic prediction remains the reference prediction.",
        "- The same 10 passes can provide MC predictive uncertainty.",
        "",
        "FAMILY SUPPORT:",
    ]

    for s in support:
        lines.append(
            f"{s['family']}: {s['decision']} | "
            f"code_files_with_dropout_evidence={s['code_files_with_dropout_evidence']}"
        )
        for p in s["candidate_paths"][:10]:
            lines.append(f"  {p}")

    lines += [
        "",
        f"GATE={gate}",
        "NEXT=INSTANTIATE_EXACT_FROZEN_MODELS_AND_REQUIRE_NATIVE_NN_DROPOUT_PLUS_DETERMINISTIC_SOURCE_PARITY_BEFORE_SCORE_LOCK",
    ]

    report = out_dir / "R17A_ADIC_MATCHED6_FEASIBILITY_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"GATE={gate}")
    print("NEXT=INSTANTIATE_EXACT_FROZEN_MODELS_AND_REQUIRE_NATIVE_NN_DROPOUT_PLUS_DETERMINISTIC_SOURCE_PARITY_BEFORE_SCORE_LOCK")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
