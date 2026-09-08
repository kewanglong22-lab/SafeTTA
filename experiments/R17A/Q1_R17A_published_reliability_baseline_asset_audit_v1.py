#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_published_reliability_baseline_asset_audit_v1.py

READ-ONLY feasibility audit for SafeTTA R17A.
No training, inference, deletion, or moving.
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-PUBLISHED-RELIABILITY-BASELINE-ASSET-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_published_reliability_baseline_asset_audit_v1")

SEARCH_ROOTS = [
    Path("outputs"),
    Path("code"),
    Path("release/SafeTTA_clean_test_v102"),
    Path("release/public_v1_fix4"),
    Path("data"),
    Path("cross_modality_prostate_mri"),
]

GROUPS = {
    "polypgen_assets": ["polypgen", "s06_c_polypgen", "r10l3a", "r10l3b"],
    "sun_assets": ["sunseg", "r14c1", "r14c2", "r14c3"],
    "promise_assets": ["promise12", "cm5a", "cm6", "cm7a"],
    "prostate158_assets": ["prostate158", "cm3", "cm4a", "cm4b", "cm4d"],
    "harm_outcome_tables": ["harm", "delta_dice", "outcome", "policy_evaluation", "gt_reveal"],
    "checkpoint_assets": ["best_source_val_dice.pt", "final.pt", "last.pt", "best.pt", ".pth"],
}

INTERESTING_SUFFIXES = {".csv", ".json", ".txt", ".npz", ".npy", ".pt", ".pth", ".py", ".md"}


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{n} B"


def norm(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def scan(root: Path):
    rows = []
    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".pytest_cache"}]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in INTERESTING_SUFFIXES:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                rel = norm(p, root)
                low = rel.lower()
                matched = [g for g, pats in GROUPS.items() if any(x in low for x in pats)]
                if matched:
                    rows.append({
                        "path": rel,
                        "size_bytes": int(size),
                        "size_human": fmt_bytes(int(size)),
                        "suffix": p.suffix.lower(),
                        "groups": "|".join(sorted(matched)),
                    })
    return rows


def dropout_code_audit(root: Path):
    hits = []
    roots = [
        root / "code",
        root / "release/SafeTTA_clean_test_v102/method_core",
        root / "release/SafeTTA_clean_test_v102/experiments",
    ]
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matched = []
            for i, line in enumerate(txt.splitlines(), 1):
                if "dropout" in line.lower():
                    matched.append(f"L{i}:{line.strip()[:220]}")
            if matched:
                hits.append({
                    "path": norm(p, root),
                    "dropout_hit_count": len(matched),
                    "examples": " | ".join(matched[:12]),
                })
    return hits


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A PUBLISHED RELIABILITY BASELINE ASSET AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Read-only: YES")
    print("Training/inference: NO")
    print("Deletion/move: NO")

    print("[1/3] Search existing frozen assets...")
    rows = scan(root)
    rows.sort(key=lambda x: (x["groups"], -x["size_bytes"]))

    print("[2/3] Audit Dropout-capable code paths...")
    dropout = dropout_code_audit(root)

    grouped = defaultdict(list)
    for r in rows:
        for g in r["groups"].split("|"):
            grouped[g].append(r)

    summary = {
        "version": VERSION,
        "root": str(root),
        "matched_asset_count": len(rows),
        "dropout_code_file_count": len(dropout),
        "groups": {
            g: {
                "count": len(v),
                "bytes": sum(int(x["size_bytes"]) for x in v),
                "human": fmt_bytes(sum(int(x["size_bytes"]) for x in v)),
            }
            for g, v in sorted(grouped.items())
        },
        "gate": "READY_FOR_R17A_IMPLEMENTATION_FEASIBILITY_REVIEW",
    }

    print("[3/3] Write reports...")
    write_csv(out_dir / "R17A_MATCHED_ASSETS.csv", rows)
    write_csv(out_dir / "R17A_DROPOUT_CODE_AUDIT.csv", dropout)
    (out_dir / "R17A_ASSET_AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "===== R17A PUBLISHED RELIABILITY BASELINE ASSET AUDIT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "GROUP SUMMARY:",
    ]
    for g, v in sorted(summary["groups"].items()):
        lines.append(f"{g}: {v['count']} files, {v['human']}")
    lines += [
        "",
        f"Dropout-related Python files: {len(dropout)}",
        "",
        "DECISION RULE:",
        "- CCD is feasible if exact frozen SOURCE probability/logit alignment exists.",
        "- ADIC is feasible only if a faithful architecture-specific dropout path can be established without changing SOURCE weights or using target GT.",
        "- If ADIC cannot be made faithful, do NOT invent a TEGDA-like approximation.",
        "- Preserve PolypGen/SUN/PROMISE12 locked prediction assets and final SOURCE checkpoints until R17A is closed.",
        "",
        "GATE=READY_FOR_R17A_IMPLEMENTATION_FEASIBILITY_REVIEW",
        "NEXT=REVIEW_ASSET_AND_DROPOUT_AUDIT_THEN_IMPLEMENT_CCD_FIRST",
    ]
    (out_dir / "R17A_ASSET_AUDIT_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GATE=READY_FOR_R17A_IMPLEMENTATION_FEASIBILITY_REVIEW")
    print("NEXT=REVIEW_ASSET_AND_DROPOUT_AUDIT_THEN_IMPLEMENT_CCD_FIRST")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
