#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

VERSION = "2026-09-20-Final68-v7-fix11-release-asset-audit-v1"

EXPECTED_SHA = {
    "FINAL68_TABLE4_PAIRED_BOOTSTRAP_SUMMARY.csv": "8fd796f7465b14bdadef2dc3ed9cea62f528ed5c4493e39f58a5d195ea8dd919",
    "FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv": "6891aed13b71289afd5f3fcf4e07f6d372677970c7bc0843e9950bac7ddf0b07",
    "Q1_Final68_Table4_paired_bootstrap_ci_v1.py": "aca0defdfb6aa598e4142bd6617ab86810d8640d199849fbbe0812892ea56c9a",
    "V7_FIX11_FINAL_MIA_CLOSURE_AUDIT.md": "1f392be4394b07de550dd97e24643f92ea75c7ad5469e392b544344d9feca0d2",
}

EXPECTED_TABLE4 = {
    "Final68 - Geometry4 H-v-B AUROC": {
        "raw_point_difference": 0.117610,
        "bootstrap_reps_requested": 2000,
        "bootstrap_reps_valid": 2000,
        "ci95_low": 0.067823,
        "ci95_high": 0.165476,
        "ci_excludes_zero": "True",
    },
    "Final68 - Transition64 H-v-B AUROC": {
        "raw_point_difference": 0.059111,
        "bootstrap_reps_requested": 2000,
        "bootstrap_reps_valid": 2000,
        "ci95_low": 0.030953,
        "ci95_high": 0.087875,
        "ci_excludes_zero": "True",
    },
}

EXPECTED_E4E = {
    0.01: (0.722411, 0.716600, 0.716243, 0.695829, "sensitivity"),
    0.02: (0.760884, 0.737825, 0.710086, 0.742179, "primary"),
    0.03: (0.770854, 0.737988, 0.693338, 0.773107, "sensitivity"),
    0.05: (0.782102, 0.740094, 0.673712, 0.798075, "sensitivity"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = root / "reproducibility" / "final68_20260919"
    print("=" * 100)
    print("SafeTTA Final68 fix11 release-asset audit")
    print("Version:", VERSION)
    print("Root:", root)
    print("=" * 100)

    for name, exp in EXPECTED_SHA.items():
        p = base / name
        assert p.is_file(), p
        got = sha256(p)
        print(f"ASSET_SHA {name}: {'PASS' if got == exp else 'FAIL'} {got}")
        assert got == exp, (name, got, exp)

    t4 = base / "FINAL68_TABLE4_PAIRED_BOOTSTRAP_SUMMARY.csv"
    with t4.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    got_by_name = {r["comparison"]: r for r in rows}
    assert set(got_by_name) == set(EXPECTED_TABLE4), got_by_name.keys()
    for name, exp in EXPECTED_TABLE4.items():
        r = got_by_name[name]
        assert close(float(r["raw_point_difference"]), exp["raw_point_difference"])
        assert int(r["bootstrap_reps_requested"]) == exp["bootstrap_reps_requested"]
        assert int(r["bootstrap_reps_valid"]) == exp["bootstrap_reps_valid"]
        assert close(float(r["ci95_low"]), exp["ci95_low"])
        assert close(float(r["ci95_high"]), exp["ci95_high"])
        assert r["ci_excludes_zero"] == exp["ci_excludes_zero"]
    print("TABLE4_PAIRED_CI_BINDING=PASS")

    e4e = base / "FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv"
    with e4e.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    for r in rows:
        m = float(r["dice_margin"])
        assert m in EXPECTED_E4E, m
        exp = EXPECTED_E4E[m]
        got = (
            float(r["hvb_auroc"]),
            float(r["hvb_auprc"]),
            float(r["harm_rollback"]),
            float(r["benefit_accepted"]),
            r["role"],
        )
        for gv, ev in zip(got[:4], exp[:4]):
            assert close(gv, ev), (m, got, exp)
        assert got[4] == exp[4], (m, got, exp)
    print("E4E_MARGIN_SUMMARY_BINDING=PASS")

    print("GATE=PASS_FINAL68_V7_FIX11_RELEASE_ASSET_AUDIT")


if __name__ == "__main__":
    main()
