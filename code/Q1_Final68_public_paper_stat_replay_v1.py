#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

VERSION = "2026-09-20-Final68-v7-public-replay-v1"

EXPECTED = {
    ("E1B2", "PolypGen_Final68", "harm_auroc"): 0.57979958,
    ("E1B2", "PolypGen_Final68", "harm_auprc"): 0.26827043,
    ("E1B2", "PolypGen_Final68", "hvb_auroc"): 0.68864073,
    ("E1B2", "PolypGen_Final68", "hvb_auprc"): 0.53383230,
    ("E1B2", "PolypGen_Final68", "deployed_dice"): 0.75691941,
    ("E1B2", "PolypGen_Final68", "deployed_minus_source"): 0.00202374,
    ("E2", "Full68", "hvb_auroc"): 0.6886407,
    ("E2", "Full68", "deployed_dice"): 0.7569194,
    ("E3B", "QCResUNet", "deployed_dice_50pct"): 0.75841653,
    ("E3B", "TEGDA_ADIC", "harm_auroc"): 0.7497391,
    ("E3B", "TEGDA_ADIC", "benefit_accepted_50pct"): 0.060823,
    ("E3B", "Full68", "benefit_accepted_50pct"): 0.744186,
    ("E4D", "TENT1", "hvb_auroc"): 0.817440,
    ("E4D", "TENT1", "deployed_minus_source"): 0.005709,
    ("E4D", "PL_CONF90", "hvb_auroc"): 0.729658,
    ("E4D", "PL_CONF90", "deployed_minus_source"): 0.011781,
    ("E4D", "MEMO", "hvb_auroc"): 0.735553,
    ("E4E", "margin_0.02", "hvb_auroc"): 0.760884,
    ("E4E", "margin_0.02", "hvb_auprc"): 0.737825,
    ("E4E", "margin_0.02", "harm_rollback"): 0.710086,
    ("E4E", "margin_0.02", "benefit_accept"): 0.742179,
    ("PROMISE12_B2", "PROMISE12_Final68", "hvb_auroc"): 0.770840,
    ("PROMISE12_B2", "PROMISE12_Final68", "deployed_minus_source"): 0.006098,
    ("E5", "tau_+0.00", "accept_fraction"): 0.224326,
    ("E5A", "near_zero_audit", "utility_neg005_to_0_fraction"): 0.691471,
    ("E6B", "controller_only", "batch1_mean_ms"): 0.086914,
    ("E6B", "controller_only", "controller_bytes"): 5188.0,
}

EXPECTED_FIGURE_SHA = {
    "fig2_e4d_action_shift_frozen_points.csv": "36cb4a79697bd3aeffc63eaafaf97b6e72ba998ad6fc23425c9e942007af7558",
    "figS1_margin_sensitivity_frozen_points.csv": "96b2a58cffa3c799619e097a86f6165009b6ef01b954b5bcc60637dd14638fef",
}

UNSAFE_PUBLIC_ASSETS = {
    "fig2_coverage_utility_frozen_points.csv",
    "fig3_action_shift_loao_frozen_points.csv",
}

FORBIDDEN_ACTIVE_GROUPS = {"E4C", "E7"}
FORBIDDEN_LEGACY_SIGNATURES = {"0.743238", "0.255458"}

EXPECTED_CONTROLLER_SHA = "8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4"
EXPECTED_TARGET68_SHA = "0640a87e087398ff68a79229509da614451310c924c270ca718e0a235c9a88ab"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = root / "reproducibility" / "final68_20260919"
    anchors = base / "FINAL68_NUMERIC_ANCHORS.csv"
    locks = base / "FINAL68_LOCKS.json"
    figdir = base / "figure_data"

    print("=" * 100)
    print("SafeTTA Final68 v7 corrected public paper-stat replay")
    print("Version:", VERSION)
    print("Root:", root)
    print("=" * 100)

    assert anchors.is_file(), anchors
    assert locks.is_file(), locks
    assert figdir.is_dir(), figdir

    with anchors.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    registry = {
        (r["group"], r["context"], r["metric"]): float(r["value"])
        for r in rows
    }
    active_groups = {r["group"] for r in rows}

    missing = []
    bad = []
    for key, exp in EXPECTED.items():
        if key not in registry:
            missing.append(key)
            continue
        got = registry[key]
        if not math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-12):
            bad.append((key, got, exp))

    print(f"Anchor rows: {len(rows)}")
    print(f"Required anchors: {len(EXPECTED)}")
    print("Missing:", len(missing))
    print("Mismatched:", len(bad))
    assert not missing, missing
    assert not bad, bad

    leaked_groups = sorted(active_groups & FORBIDDEN_ACTIVE_GROUPS)
    print("Forbidden active groups:", leaked_groups)
    assert not leaked_groups, leaked_groups

    for name, exp_sha in EXPECTED_FIGURE_SHA.items():
        p = figdir / name
        assert p.is_file(), p
        got = sha256(p)
        print(f"FIGURE_DATA {name}: {'PASS' if got == exp_sha else 'FAIL'} {got}")
        assert got == exp_sha, (name, got, exp_sha)

    for name in sorted(UNSAFE_PUBLIC_ASSETS):
        p = figdir / name
        print(f"UNSAFE_ASSET_EXCLUDED {name}: {'PASS' if not p.exists() else 'FAIL'}")
        assert not p.exists(), p

    text = anchors.read_text(encoding="utf-8")
    for sig in FORBIDDEN_LEGACY_SIGNATURES:
        assert sig not in text, f"legacy 130-D signature leaked into Final68 registry: {sig}"

    lock_data = json.loads(locks.read_text(encoding="utf-8"))
    frozen = lock_data["frozen_artifacts"]

    assert frozen["controller_joblib_sha256"] == EXPECTED_CONTROLLER_SHA
    assert frozen["target68_npy_sha256"] == EXPECTED_TARGET68_SHA

    controller_path = root / frozen["controller_joblib_path"]
    assert controller_path.is_file(), controller_path
    controller_actual_sha = sha256(controller_path)
    print(
        "FINAL68_CONTROLLER_ACTUAL_FILE_SHA="
        + ("PASS" if controller_actual_sha == EXPECTED_CONTROLLER_SHA else "FAIL")
        + f" {controller_actual_sha}"
    )
    assert controller_actual_sha == EXPECTED_CONTROLLER_SHA

    print("FINAL68_TARGET68_LOCK_SHA=PASS")
    print("LEGACY_SIGNATURE_EXCLUSION=PASS")
    print("GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY")


if __name__ == "__main__":
    main()
