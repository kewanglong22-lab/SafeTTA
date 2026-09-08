#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_segformer_fix1_lineage_audit_v1.py

SafeTTA R17A — targeted provenance audit for the frozen PolypGen SegFormer
SOURCE masks after current-environment re-inference failed exact parity.

READ-ONLY
---------
Training: NO
Inference: NO
Target GT loading: NO
HARM/outcome loading: NO
Deletion/move: NO

Scientific purpose
------------------
The current package versions match the final frozen release environment, so an
obvious package-version mismatch is not supported by the previous audit.

However, Q1_R10L3A Fix2 explicitly bootstrapped completed state artifacts from
a Fix1 partial build. This audit determines whether the current frozen
SegFormer masks are exact copies of Fix1 artifacts and whether Fix1 used the
same runner/script lineage.

It also checks whether Fix1 retained any soft-logit/probability artifact that
would let R17A compute CCD directly without re-inference.

Targets
-------
SegFormer-B0 seeds:
  20260820, 20260821, 20260822

Current canonical route:
  outputs/Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1

Historical partial route (if still present):
  outputs/Q1_R10L3A_polypgen_source_a1_prediction_lock_fix1_v1__building

No scientific file is modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "2026-09-09-Q1-R17A-POLYPGEN-SEGF-FIX1-LINEAGE-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_segformer_fix1_lineage_audit_v1")

FIX2_OUT_REL = Path(r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1")
DEFAULT_FIX1_OUT_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix1_v1__building"
)

FIX1_SCRIPT_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix1.py")
FIX2_SCRIPT_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

SEEDS = (20260820, 20260821, 20260822)

CURRENT_EXPECTED = {
    20260820: {
        "npz_sha256": "d36ad27453e0532516f955c5c4d55e7282f38a04426aaf58bfb10bf5c9d089e0",
        "index_sha256": "d47255986c7951347eb6a4894c52f39e82b567153cee75c5c993d2885e508b5f",
    },
    20260821: {
        "npz_sha256": "387fbadd066ab9c668bb59895a22fd2dd1580bad6c4225f9a42cd48a4e325dee",
        "index_sha256": "771e0dc563a54f412e4a160eb3d010b4d0d53ffc9acd56f6446f2de2f7dd5e5c",
    },
    20260822: {
        "npz_sha256": "a26d7c9eb9cb11055779410106d5f1e0476b3990609a0698ad7005e24b989611",
        "index_sha256": "9cce1721c1d3e254a2110e29253aad0642d723a4300e4ed7d6cd9619966e08d9",
    },
}

SOFT_TOKENS = ("logit", "logits", "prob", "probs", "softmax", "probability")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def flatten(obj, prefix="") -> Dict[str, object]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def find_fix1_dir(root: Path, fix2_dir: Path) -> Path:
    upstream = fix2_dir / "upstream_audit.json"
    if upstream.exists():
        try:
            obj = load_json(upstream)
            raw = obj.get("fix1_partial_build_dir")
            if raw:
                p = Path(str(raw))
                if p.exists():
                    return p
        except Exception:
            pass
    return root / DEFAULT_FIX1_OUT_REL


def artifact_paths(state_dir: Path, seed: int) -> Dict[str, Path]:
    stem = f"segformer_b0_seed{seed}"
    return {
        "npz": state_dir / f"{stem}_predictions.npz",
        "index": state_dir / f"{stem}_index.csv",
        "lock": state_dir / f"{stem}_state_lock.json",
    }


def compare_json(a: dict, b: dict) -> List[dict]:
    fa = flatten(a)
    fb = flatten(b)
    keys = sorted(set(fa) | set(fb))
    rows = []
    for k in keys:
        va = fa.get(k, "<MISSING>")
        vb = fb.get(k, "<MISSING>")
        if va != vb:
            rows.append({
                "key": k,
                "fix1": str(va)[:2000],
                "fix2": str(vb)[:2000],
            })
    return rows


def extract_function_source(path: Path, names: List[str]) -> List[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            rows.append({
                "function": node.name,
                "start_line": start,
                "end_line": end,
                "source": "\n".join(lines[start-1:end]),
            })
    rows.sort(key=lambda r: r["start_line"])
    return rows


def scan_script_lineage(path: Path) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "sha256": "",
            "r05_mentions": [],
            "r03_mentions": [],
            "bootstrap_mentions": [],
        }
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    def hits(tokens):
        out = []
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(t.lower() in low for t in tokens):
                out.append(f"L{i}:{line.strip()[:500]}")
        return out

    return {
        "exists": True,
        "sha256": sha256_file(path),
        "r05_mentions": hits([
            "R05D3", "EXPECTED_R05D3_SCRIPT_SHA256",
            "run_segformer_state",
        ]),
        "r03_mentions": hits([
            "Q1_R03", "r03", "segformer",
        ]),
        "bootstrap_mentions": hits([
            "bootstrap", "fix1", "resume verified",
            "bootstrapped",
        ]),
    }


def scan_soft_assets(base: Path, root: Path) -> List[dict]:
    rows = []
    if not base.exists():
        return rows
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if not any(tok in low for tok in SOFT_TOKENS):
            continue
        if p.suffix.lower() not in {".npy", ".npz", ".pt", ".pth", ".csv"}:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        rows.append({
            "path": rel(p, root) if root in p.parents else str(p),
            "size_bytes": int(size),
            "sha256": sha256_file(p) if size <= 2 * 1024**3 else "SKIPPED_GT_2GIB",
        })
    rows.sort(key=lambda r: -r["size_bytes"])
    return rows


def search_bootstrap_logs(base: Path, root: Path) -> List[dict]:
    rows = []
    if not base.exists():
        return rows
    patterns = (
        "BOOTSTRAP VERIFIED",
        "Verified Fix1 states imported",
        "RESUME VERIFIED",
    )
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".log", ".json", ".csv"}:
            continue
        try:
            if p.stat().st_size > 20 * 1024 * 1024:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        matched = []
        for i, line in enumerate(txt.splitlines(), 1):
            if any(tok.lower() in line.lower() for tok in patterns):
                matched.append(f"L{i}:{line.strip()[:500]}")
        if matched:
            rows.append({
                "path": rel(p, root) if root in p.parents else str(p),
                "matches": " | ".join(matched[:100]),
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

    fix2_dir = root / FIX2_OUT_REL
    if not fix2_dir.exists():
        raise FileNotFoundError(fix2_dir)

    fix1_dir = find_fix1_dir(root, fix2_dir)

    print("===== R17A POLYPGEN SEGF FIX1 LINEAGE AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Fix2 canonical:", fix2_dir)
    print("Fix1 historical:", fix1_dir)
    print("Training: NO")
    print("Inference: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Deletion/move: NO")
    print()

    # Verify authoritative helper scripts first.
    r05 = root / R05_REL
    r03 = root / R03_REL
    if sha256_file(r05) != R05_SHA:
        raise RuntimeError("R05 authoritative script SHA mismatch.")
    if sha256_file(r03) != R03_SHA:
        raise RuntimeError("R03 authoritative script SHA mismatch.")

    print("[1/6] Compare Fix1 vs Fix2 SegFormer artifacts...")
    artifact_rows = []
    json_diff_rows = []

    fix2_state = fix2_dir / "state_predictions"
    fix1_state = fix1_dir / "state_predictions"

    all_npz_equal = True
    all_index_equal = True
    all_fix1_present = True

    for seed in SEEDS:
        p2 = artifact_paths(fix2_state, seed)
        p1 = artifact_paths(fix1_state, seed)

        row = {"seed": seed}

        for kind in ["npz", "index", "lock"]:
            row[f"fix2_{kind}_path"] = rel(p2[kind], root)
            row[f"fix2_{kind}_exists"] = p2[kind].exists()
            row[f"fix2_{kind}_sha256"] = (
                sha256_file(p2[kind]) if p2[kind].exists() else ""
            )

            if p1[kind].exists():
                try:
                    row[f"fix1_{kind}_path"] = rel(p1[kind], root)
                except ValueError:
                    row[f"fix1_{kind}_path"] = str(p1[kind])
                row[f"fix1_{kind}_exists"] = True
                row[f"fix1_{kind}_sha256"] = sha256_file(p1[kind])
                row[f"{kind}_sha_equal"] = (
                    row[f"fix1_{kind}_sha256"] == row[f"fix2_{kind}_sha256"]
                )
            else:
                row[f"fix1_{kind}_path"] = str(p1[kind])
                row[f"fix1_{kind}_exists"] = False
                row[f"fix1_{kind}_sha256"] = ""
                row[f"{kind}_sha_equal"] = False
                all_fix1_present = False

        # Guard current canonical SHAs.
        exp = CURRENT_EXPECTED[seed]
        if row["fix2_npz_sha256"] != exp["npz_sha256"]:
            raise RuntimeError(f"Current Fix2 NPZ SHA drift seed={seed}")
        if row["fix2_index_sha256"] != exp["index_sha256"]:
            raise RuntimeError(f"Current Fix2 index SHA drift seed={seed}")

        all_npz_equal &= bool(row["npz_sha_equal"])
        all_index_equal &= bool(row["index_sha_equal"])

        if p1["lock"].exists() and p2["lock"].exists():
            a = load_json(p1["lock"])
            b = load_json(p2["lock"])
            for d in compare_json(a, b):
                json_diff_rows.append({
                    "seed": seed,
                    **d,
                })

        artifact_rows.append(row)

    print("[2/6] Inspect Fix1/Fix2 scripts and exact runner lineage...")
    fix1_script = root / FIX1_SCRIPT_REL
    fix2_script = root / FIX2_SCRIPT_REL
    script_rows = [
        {
            "script": rel(fix1_script, root),
            **scan_script_lineage(fix1_script),
        },
        {
            "script": rel(fix2_script, root),
            **scan_script_lineage(fix2_script),
        },
    ]

    function_rows = []
    for label, p in [("fix1", fix1_script), ("fix2", fix2_script)]:
        for r in extract_function_source(
            p,
            [
                "run_state",
                "bootstrap_verified_fix1_states",
                "load_upstream_locks",
            ],
        ):
            function_rows.append({
                "script_variant": label,
                "script_path": rel(p, root),
                **r,
            })

    print("[3/6] Inspect sidecar runner/script SHA lineage...")
    sidecar_rows = []
    for seed in SEEDS:
        for variant, state_dir in [("fix1", fix1_state), ("fix2", fix2_state)]:
            lockp = artifact_paths(state_dir, seed)["lock"]
            if not lockp.exists():
                continue
            obj = load_json(lockp)
            flat = flatten(obj)
            sidecar_rows.append({
                "seed": seed,
                "variant": variant,
                "path": (
                    rel(lockp, root) if root in lockp.parents else str(lockp)
                ),
                "decision": obj.get("decision", ""),
                "model_family": obj.get("model_family", ""),
                "model_state_id": obj.get("model_state_id", ""),
                "training_seed": obj.get("training_seed", ""),
                "checkpoint_sha256": obj.get("checkpoint_sha256", ""),
                "r05d3_script_sha256": obj.get("r05d3_script_sha256", ""),
                "npz_sha256": obj.get("npz_sha256", ""),
                "index_sha256": obj.get("index_sha256", ""),
                "all_sha_or_runner_fields": " | ".join(
                    f"{k}={v}"
                    for k, v in flat.items()
                    if any(tok in k.lower() for tok in [
                        "sha", "script", "runner", "lineage",
                        "historical", "canonical", "import",
                    ])
                )[:10000],
            })

    print("[4/6] Search Fix1 for retained soft logits/probabilities...")
    soft_rows = scan_soft_assets(fix1_dir, root)

    print("[5/6] Search for bootstrap/resume provenance logs...")
    log_rows = search_bootstrap_logs(fix2_dir, root)
    log_rows += search_bootstrap_logs(fix1_dir, root)

    print("[6/6] Build provenance decision...")
    fix1_r05_shas = {
        str(r.get("r05d3_script_sha256", "")).lower()
        for r in sidecar_rows
        if r["variant"] == "fix1" and r.get("r05d3_script_sha256")
    }
    fix2_r05_shas = {
        str(r.get("r05d3_script_sha256", "")).lower()
        for r in sidecar_rows
        if r["variant"] == "fix2" and r.get("r05d3_script_sha256")
    }

    soft_segformer = [
        r for r in soft_rows
        if "segformer" in r["path"].lower()
    ]

    if all_fix1_present and all_npz_equal and all_index_equal:
        if soft_segformer:
            decision = "FIX2_SEGF_ARTIFACTS_ARE_EXACT_FIX1_COPIES_AND_FIX1_SOFT_ASSETS_EXIST"
        else:
            decision = "FIX2_SEGF_ARTIFACTS_ARE_EXACT_FIX1_COPIES_NO_FIX1_SOFT_ASSETS"
    elif all_fix1_present:
        decision = "FIX1_EXISTS_BUT_FIX2_SEGF_ARTIFACTS_DIFFER"
    else:
        decision = "FIX1_HISTORICAL_ARTIFACTS_INCOMPLETE_OR_MISSING"

    summary = {
        "version": VERSION,
        "fix1_dir": str(fix1_dir),
        "fix1_exists": fix1_dir.exists(),
        "all_fix1_state_artifacts_present": all_fix1_present,
        "all_fix1_fix2_npz_sha_equal": all_npz_equal,
        "all_fix1_fix2_index_sha_equal": all_index_equal,
        "fix1_r05d3_script_shas": sorted(fix1_r05_shas),
        "fix2_r05d3_script_shas": sorted(fix2_r05_shas),
        "authoritative_r05_sha": R05_SHA,
        "soft_asset_count_in_fix1": len(soft_rows),
        "segformer_soft_asset_count_in_fix1": len(soft_segformer),
        "decision": decision,
    }

    write_csv(out_dir / "R17A_FIX1_FIX2_SEGF_ARTIFACT_COMPARISON.csv", artifact_rows)
    write_csv(out_dir / "R17A_FIX1_FIX2_STATE_LOCK_DIFF.csv", json_diff_rows)
    write_csv(out_dir / "R17A_FIX1_FIX2_SCRIPT_LINEAGE.csv", script_rows)
    write_csv(out_dir / "R17A_FIX1_FIX2_FUNCTIONS.csv", function_rows)
    write_csv(out_dir / "R17A_FIX1_FIX2_SIDECAR_LINEAGE.csv", sidecar_rows)
    write_csv(out_dir / "R17A_FIX1_SOFT_ASSET_SEARCH.csv", soft_rows)
    write_csv(out_dir / "R17A_FIX1_BOOTSTRAP_LOG_SEARCH.csv", log_rows)

    (out_dir / "R17A_FIX1_LINEAGE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A POLYPGEN SEGF FIX1 LINEAGE AUDIT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        f"Fix1 dir: {fix1_dir}",
        f"Fix1 exists: {fix1_dir.exists()}",
        "",
        "INFORMATION BOUNDARY:",
        "Training: NO",
        "Inference: NO",
        "Target GT loading: NO",
        "HARM/outcome loading: NO",
        "",
        "ARTIFACT COMPARISON:",
    ]

    for r in artifact_rows:
        lines.append(
            f"seed={r['seed']} "
            f"npz_equal={r['npz_sha_equal']} "
            f"index_equal={r['index_sha_equal']} "
            f"lock_equal={r['lock_sha_equal']} "
            f"fix1_npz={r['fix1_npz_sha256']} "
            f"fix2_npz={r['fix2_npz_sha256']}"
        )

    lines += [
        "",
        f"Fix1 sidecar R05D3 SHAs: {sorted(fix1_r05_shas)}",
        f"Fix2 sidecar R05D3 SHAs: {sorted(fix2_r05_shas)}",
        f"Authoritative R05D3 SHA: {R05_SHA}",
        f"Fix1 soft-like assets found: {len(soft_rows)}",
        f"Fix1 SegFormer soft-like assets found: {len(soft_segformer)}",
    ]

    for r in soft_segformer[:30]:
        lines.append(
            f"  SOFT_CANDIDATE: {r['path']} size={r['size_bytes']} sha={r['sha256']}"
        )

    lines += [
        "",
        f"DECISION={decision}",
        "",
        "INTERPRETATION RULE:",
        "- If Fix2 NPZ+index are exact Fix1 copies, the canonical hard masks were",
        "  not freshly regenerated by Fix2; they inherited Fix1 runtime provenance.",
        "- If Fix1 retained SegFormer soft logits/probabilities, use those directly",
        "  after exact sample/state alignment and do not re-infer.",
        "- If no Fix1 soft asset exists and the same scripts/checkpoints are recorded,",
        "  the historical hard-mask lock alone is insufficient to reconstruct exact",
        "  historical soft probabilities.",
        "- Do not tune preprocessing or thresholds to reproduce Fix1 hard masks.",
        "",
        "NEXT=USE_FIX1_SOFT_ASSET_IF_PRESENT_ELSE_DECIDE_CURRENT_ENV_SOFT_REGENERATION_WITH_DISCLOSED_HARD_PARITY_LIMIT",
    ]

    report = out_dir / "R17A_POLYPGEN_SEGF_FIX1_LINEAGE_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"DECISION={decision}")
    print("NEXT=USE_FIX1_SOFT_ASSET_IF_PRESENT_ELSE_DECIDE_CURRENT_ENV_SOFT_REGENERATION_WITH_DISCLOSED_HARD_PARITY_LIMIT")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
