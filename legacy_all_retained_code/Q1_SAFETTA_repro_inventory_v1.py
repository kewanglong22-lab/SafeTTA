#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_SAFETTA_repro_inventory_v1.py

Read-only reproducibility inventory for the frozen SafeTTA paper pipeline.
Builds an auditable manifest of retained files under F:\\MEDSEG_SAFETTA.
It never imports/executes experiment scripts and never trains/tunes models.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, platform, subprocess, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-06-Q1-SAFETTA-REPRO-INVENTORY-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
ROUTE_TOKENS = [
    "Q1_R10K2A", "Q1X_CM0_", "Q1X_CM2B_", "Q1X_CM3_", "Q1X_CM4A_",
    "Q1X_CM4B_", "Q1X_CM4C_", "Q1X_CM4D_", "Q1X_CM5A_", "Q1X_CM5B_",
    "Q1X_CM6_", "Q1X_CM7A_", "Q1X_CM7B_", "Q1_SAFETTA_harm_margin_sensitivity",
    "runtime",
]
SIDECAR_TOKENS = ["LOCK", "COMPLETE", "PREFLIGHT", "MANIFEST", "META", "AUDIT", "SUMMARY"]
MAX_SMALL_BYTES = 50 * 1024 * 1024

@dataclass
class FileRecord:
    path: str
    relpath: str
    size_bytes: int
    sha256: Optional[str]
    suffix: str
    category: str
    syntax_ok: Optional[bool]
    route_tokens: str


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_files(root: Path):
    for base in [root / "code", root / "outputs"]:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    yield p


def classify(root: Path, p: Path):
    sp = str(p)
    hits = [t for t in ROUTE_TOKENS if t.lower() in sp.lower()]
    if str(p).lower().startswith(str(root / "code").lower()) and p.suffix.lower() == ".py":
        return ("candidate_final_script" if hits else "other_code"), hits
    if p.suffix.lower() in {".json", ".csv", ".txt", ".yaml", ".yml", ".md"}:
        if hits or any(t in p.name.upper() for t in SIDECAR_TOKENS):
            return "protocol_or_result_sidecar", hits
        return "other_text", hits
    if hits:
        return "route_binary_or_large_artifact", hits
    return "other", hits


def syntax_check(p: Path):
    if p.suffix.lower() != ".py":
        return None
    try:
        ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        return True
    except Exception:
        return False


def pip_freeze():
    try:
        cp = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        return cp.stdout if cp.returncode == 0 else f"# pip freeze failed\n{cp.stderr}"
    except Exception as e:
        return f"# pip freeze unavailable: {e}\n"


def environment_summary():
    out = {"python_executable": sys.executable, "python_version": sys.version.replace("\n", " "), "platform": platform.platform()}
    for label, modname in [("torch","torch"),("torchvision","torchvision"),("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("sklearn","sklearn"),("transformers","transformers"),("joblib","joblib")]:
        try:
            m = __import__(modname)
            out[label] = getattr(m, "__version__", "UNKNOWN")
        except Exception as e:
            out[label] = f"NOT_IMPORTABLE: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--hash-large", action="store_true", help="Also hash large route artifacts/checkpoints.")
    args = ap.parse_args()

    root = args.root
    outdir = args.output_dir or (root / "outputs" / "Q1_SAFETTA_repro_inventory_v1")
    outdir.mkdir(parents=True, exist_ok=True)

    print("===== SAFETTA REPRODUCIBILITY INVENTORY =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Output:", outdir)
    print("Read-only project scan: YES")
    print("Project experiment scripts executed/imported: NO")
    print("Large-file hashing:", "YES" if args.hash_large else "NO")

    if not root.exists():
        raise FileNotFoundError(root)
    if not (root / "code").exists():
        raise FileNotFoundError(root / "code")

    files = list(iter_files(root))
    iterator = tqdm(files, desc="Inventory SafeTTA files", unit="file", dynamic_ncols=True) if tqdm else files
    records = []
    for p in iterator:
        category, hits = classify(root, p)
        if category not in {"candidate_final_script", "protocol_or_result_sidecar", "route_binary_or_large_artifact"}:
            continue
        size = p.stat().st_size
        do_hash = category in {"candidate_final_script", "protocol_or_result_sidecar"} and size <= MAX_SMALL_BYTES
        if category == "route_binary_or_large_artifact" and args.hash_large:
            do_hash = True
        digest = sha256_file(p) if do_hash else None
        records.append(FileRecord(str(p), str(p.relative_to(root)), size, digest, p.suffix.lower(), category, syntax_check(p), ";".join(hits)))

    records.sort(key=lambda r: (r.category, r.relpath.lower()))
    csv_path = outdir / "repro_inventory.csv"
    fieldnames = ["path","relpath","size_bytes","sha256","suffix","category","syntax_ok","route_tokens"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in records: w.writerow(asdict(r))

    env = environment_summary()
    (outdir / "environment_summary.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "environment_freeze.txt").write_text(pip_freeze(), encoding="utf-8")

    scripts = [r for r in records if r.category == "candidate_final_script"]
    sidecars = [r for r in records if r.category == "protocol_or_result_sidecar"]
    binaries = [r for r in records if r.category == "route_binary_or_large_artifact"]
    syntax_failures = [r.relpath for r in scripts if r.syntax_ok is False]
    unhashed_large = [r.relpath for r in binaries if not r.sha256]

    report = {
        "version": VERSION,
        "root": str(root),
        "read_only": True,
        "project_scripts_executed_or_imported": False,
        "hash_large": bool(args.hash_large),
        "environment": env,
        "counts": {
            "candidate_final_scripts": len(scripts),
            "protocol_or_result_sidecars": len(sidecars),
            "route_binary_or_large_artifacts": len(binaries),
            "syntax_failures": len(syntax_failures),
            "unhashed_route_binary_or_large_artifacts": len(unhashed_large),
        },
        "syntax_failures": syntax_failures,
        "unhashed_route_binary_or_large_artifacts": unhashed_large,
        "records": [asdict(r) for r in records],
        "reproducibility_gate": {
            "paper_repo_can_be_called_exact_reproducibility_ready": False,
            "reason": "Inventory only. Exact reproducibility requires wiring the frozen scripts/artifacts into one-command paper-table reproduction and asserting reported numeric outputs."
        }
    }
    (outdir / "repro_inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "===== SAFETTA REPRODUCIBILITY INVENTORY SUMMARY =====",
        f"Version: {VERSION}", f"Root: {root}",
        f"Candidate final scripts: {len(scripts)}",
        f"Protocol/result sidecars: {len(sidecars)}",
        f"Route binary/large artifacts: {len(binaries)}",
        f"Python syntax failures: {len(syntax_failures)}", "",
        "IMPORTANT:",
        "Do NOT publish the current scaffold as an exact reproducibility repository yet.",
        "Exact reproducibility requires reproducing paper tables/figures from frozen artifacts and checking expected numeric outputs.", "",
        "Candidate final scripts:",
    ]
    lines += [f"  {r.sha256 or 'NO_SHA'}  {r.relpath}" for r in scripts]
    lines += ["", "Unhashed large route artifacts:"]
    lines += [f"  {x}" for x in unhashed_large[:300]]
    (outdir / "REPRO_INVENTORY_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n===== FINAL =====")
    print("candidate_final_scripts =", len(scripts))
    print("protocol_or_result_sidecars =", len(sidecars))
    print("route_binary_or_large_artifacts =", len(binaries))
    print("syntax_failures =", len(syntax_failures))
    print("Decision=INVENTORY_READY_FOR_EXACT_REPRO_REPO_CONSTRUCTION")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
