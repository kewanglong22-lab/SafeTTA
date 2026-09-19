#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path

VERSION = "2026-09-19-Final68-public-replay-v1"

EXPECTED = {
    ("E1B2","PolypGen_Final68","harm_auroc"): 0.57979958,
    ("E1B2","PolypGen_Final68","harm_auprc"): 0.26827043,
    ("E1B2","PolypGen_Final68","hvb_auroc"): 0.68864073,
    ("E1B2","PolypGen_Final68","hvb_auprc"): 0.53383230,
    ("E1B2","PolypGen_Final68","deployed_dice"): 0.75691941,
    ("E1B2","PolypGen_Final68","deployed_minus_source"): 0.00202374,
    ("E2","Full68","hvb_auroc"): 0.6886407,
    ("E2","Full68","deployed_dice"): 0.7569194,
    ("E3B","QCResUNet","deployed_dice_50pct"): 0.75841653,
    ("E3B","TEGDA_ADIC","harm_auroc"): 0.7497391,
    ("E3B","TEGDA_ADIC","benefit_accepted_50pct"): 0.060823,
    ("E3B","Full68","benefit_accepted_50pct"): 0.744186,
    ("E4C","TENT1","hvb_auroc"): 0.84081,
    ("E4C","PL_CONF90","hvb_auroc"): 0.77743,
    ("E4C","MEMO","hvb_auroc"): 0.83393,
    ("PROMISE12_B2","PROMISE12_Final68","hvb_auroc"): 0.770840,
    ("PROMISE12_B2","PROMISE12_Final68","deployed_minus_source"): 0.006098,
    ("E5","tau_+0.00","accept_fraction"): 0.224326,
    ("E5A","near_zero_audit","utility_neg005_to_0_fraction"): 0.691471,
    ("E6B","controller_only","batch1_mean_ms"): 0.086914,
    ("E6B","controller_only","controller_bytes"): 5188.0,
    ("E7","coverage_10","deployed_dice"): 0.759488,
    ("E7","coverage_20","deployed_dice"): 0.759401,
    ("E7","coverage_50","benefit_accept"): 0.744186,
}

EXPECTED_FIGURE_SHA = {
    "fig2_coverage_utility_frozen_points.csv": "b47c30e3cf56173e7b075b5876c5364720759cd9e9dda49e55bd419b39ae198e",
    "fig3_action_shift_loao_frozen_points.csv": "68790ec7cb6a17cdf73932b87d34f3946f7689bdb245a63c39cb7ba90d422225",
    "figS1_margin_sensitivity_frozen_points.csv": "96b2a58cffa3c799619e097a86f6165009b6ef01b954b5bcc60637dd14638fef",
}

FORBIDDEN_LEGACY_SIGNATURES = {"0.743238", "0.255458"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    base = root / "reproducibility" / "final68_20260919"
    anchors = base / "FINAL68_NUMERIC_ANCHORS.csv"
    locks = base / "FINAL68_LOCKS.json"
    figdir = base / "figure_data"

    print("=" * 100)
    print("SafeTTA Final68 public paper-stat replay")
    print("Version:", VERSION)
    print("Root:", root)
    print("=" * 100)

    assert anchors.is_file(), anchors
    assert locks.is_file(), locks

    with anchors.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    registry = {
        (r["group"], r["context"], r["metric"]): float(r["value"])
        for r in rows
    }

    missing = []
    bad = []
    for k, exp in EXPECTED.items():
        if k not in registry:
            missing.append(k)
            continue
        got = registry[k]
        if not math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-12):
            bad.append((k, got, exp))

    print(f"Anchor rows: {len(rows)}")
    print(f"Required anchors: {len(EXPECTED)}")
    print("Missing:", len(missing))
    print("Mismatched:", len(bad))
    assert not missing, missing
    assert not bad, bad

    fig_status = []
    for name, exp_sha in EXPECTED_FIGURE_SHA.items():
        p = figdir / name
        assert p.is_file(), p
        got = sha256(p)
        ok = got == exp_sha
        fig_status.append(ok)
        print(f"FIGURE_DATA {name}: {'PASS' if ok else 'FAIL'} {got}")
    assert all(fig_status)

    text = anchors.read_text(encoding="utf-8")
    for sig in FORBIDDEN_LEGACY_SIGNATURES:
        assert sig not in text, f"legacy 130-D signature leaked into Final68 registry: {sig}"

    lock_data = json.loads(locks.read_text(encoding="utf-8"))
    controller_sha = lock_data["frozen_artifacts"]["controller_joblib_sha256"]
    target68_sha = lock_data["frozen_artifacts"]["target68_npy_sha256"]
    assert controller_sha == "8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4"
    assert target68_sha == "0640a87e087398ff68a79229509da614451310c924c270ca718e0a235c9a88ab"

    print("FINAL68_CONTROLLER_SHA=PASS")
    print("FINAL68_TARGET68_SHA=PASS")
    print("LEGACY_SIGNATURE_EXCLUSION=PASS")
    print("GATE=PASS_FINAL68_PUBLIC_PAPER_STAT_REPLAY")

if __name__ == "__main__":
    main()
