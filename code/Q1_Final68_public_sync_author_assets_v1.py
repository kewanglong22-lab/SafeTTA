#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path

EXPECTED = {
    "controller": (
        Path(r"outputs\Q1_Final68_E1B1_source_tristate_controller_polypgen_pregt_scoring_v1\FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib"),
        "8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4",
        Path(r"artifacts\final68\FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib"),
    ),
    "target68": (
        Path(r"outputs\Q1_Final68_E1B0_polypgen_unseen_memo_final68_exact_build_v1_fix2\FINAL68_E1B0_POLYPGEN_DEEPLAB_MEMO_68D_PREOUTCOME.npy"),
        "0640a87e087398ff68a79229509da614451310c924c270ca718e0a235c9a88ab",
        Path(r"artifacts\final68\FINAL68_E1B0_POLYPGEN_DEEPLAB_MEMO_68D_PREOUTCOME.npy"),
    ),
}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=r"F:\MEDSEG_SAFETTA")
    ap.add_argument("--repo", required=True, help="Local clone of SafeTTA final68 sync branch")
    ap.add_argument("--include-target68", action="store_true",
                    help="Also copy target68 NPY. Omit to publish only the 5-KB controller.")
    args = ap.parse_args()

    workspace = Path(args.workspace)
    repo = Path(args.repo)
    selected = ["controller"] + (["target68"] if args.include_target68 else [])

    print("=" * 100)
    print("SafeTTA Final68 exact author-asset sync")
    print("Workspace:", workspace)
    print("Repo:", repo)
    print("Selected:", selected)
    print("=" * 100)

    for key in selected:
        rel_src, exp_sha, rel_dst = EXPECTED[key]
        src = workspace / rel_src
        dst = repo / rel_dst
        if not src.is_file():
            raise FileNotFoundError(src)
        got = sha256(src)
        print(f"{key}: source_sha={got}")
        if got != exp_sha:
            raise RuntimeError(f"{key} SHA mismatch: expected {exp_sha}, got {got}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied = sha256(dst)
        if copied != exp_sha:
            raise RuntimeError(f"{key} copied SHA mismatch")
        print(f"{key}: COPY=PASS -> {dst}")

    print("GATE=PASS_FINAL68_AUTHOR_BINARY_ASSET_SYNC")

if __name__ == "__main__":
    main()
