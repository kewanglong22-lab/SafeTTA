#!/usr/bin/env python3
"""Portable reviewer-facing replay for frozen R31-R33 manuscript statistics.

This script validates the released compact outputs against frozen numeric anchors.
It does not retrain models, run TTA inference, access datasets/GT, calibrate on the
target domain, or regenerate DINO features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ABS_TOL = 1e-9

EXPECTED_LOAO = {
    "Random": (0.5, 0.5, 0.5, 0.5, 0.16713),
    "SOURCE semantic64 only": (0.666059, 0.619009, 0.658999, 0.648022, 0.224168),
    "SOURCE Q66 only": (0.727547, 0.669616, 0.787407, 0.72819, 0.295683),
    "DeltaSemantic64 only": (0.663397, 0.655501, 0.764159, 0.694352, 0.455586),
    "Action-conditioned Q66+DeltaS": (0.725173, 0.712228, 0.774016, 0.737139, 0.381348),
    "Shared Q66+DeltaS": (0.71796, 0.736321, 0.770519, 0.7416, 0.415859),
    "Separate-action reference": (0.808604, 0.758185, 0.907669, 0.824819, 0.503236),
}

EXPECTED_R33 = {
    "SafeTTA-Q66+DeltaS": (0.7432377136, 0.2554579725, 0.06549173194, 3.900614092),
    "TEGDA-ADIC": (0.749739, 0.154403, 0.06549173194, 2.357595),
    "MC-dropout": (0.616843, 0.098194, 0.06549173194, 1.49933),
    "SicTTA-CCD": (0.607297, 0.083324, 0.06549173194, 1.272283),
}

EXPECTED_DELTAS = {
    ("SicTTA-CCD", "AUROC"): (0.135941, 0.072382, 0.19382),
    ("SicTTA-CCD", "AUPRC"): (0.172134, 0.125599, 0.228141),
    ("TEGDA-ADIC", "AUROC"): (-0.006501, -0.043762, 0.028323),
    ("TEGDA-ADIC", "AUPRC"): (0.101055, 0.055974, 0.149594),
    ("MC-dropout", "AUROC"): (0.126395, 0.083388, 0.168958),
    ("MC-dropout", "AUPRC"): (0.157264, 0.114321, 0.210393),
}

EXPECTED_CLAIM_LOCK = "d6b2fd30114c3b9d81eb32712cd5958d09251059c2b86a07e7acff682a865521"


def close(actual: str | float, expected: float) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=ABS_TOL)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_loao(path: Path) -> None:
    rows = read_csv(path)
    require(len(rows) == len(EXPECTED_LOAO), f"R32A row count mismatch: {len(rows)}")
    by_name = {r["representation"]: r for r in rows}
    require(set(by_name) == set(EXPECTED_LOAO), "R32A representation set mismatch")
    columns = ("memo_auroc", "pl_auroc", "tent_auroc", "macro_auroc", "macro_auprc")
    for name, expected in EXPECTED_LOAO.items():
        row = by_name[name]
        for col, exp in zip(columns, expected):
            require(close(row[col], exp), f"R32A mismatch: {name} / {col}: {row[col]} != {exp}")


def validate_r33_primary(path: Path) -> None:
    rows = read_csv(path)
    require(len(rows) == len(EXPECTED_R33), f"R33 primary row count mismatch: {len(rows)}")
    by_name = {r["method"]: r for r in rows}
    require(set(by_name) == set(EXPECTED_R33), "R33 primary method set mismatch")
    columns = ("auroc", "auprc", "harm_prevalence", "auprc_lift")
    for name, expected in EXPECTED_R33.items():
        row = by_name[name]
        for col, exp in zip(columns, expected):
            require(close(row[col], exp), f"R33 primary mismatch: {name} / {col}: {row[col]} != {exp}")


def validate_r33_deltas(path: Path) -> None:
    rows = read_csv(path)
    require(len(rows) == len(EXPECTED_DELTAS), f"R33 delta row count mismatch: {len(rows)}")
    by_key = {(r["comparator"], r["metric"]): r for r in rows}
    require(set(by_key) == set(EXPECTED_DELTAS), "R33 paired-delta key set mismatch")
    delta_col = "delta_safettta_minus_comparator"
    columns = (delta_col, "ci95_low", "ci95_high")
    for key, expected in EXPECTED_DELTAS.items():
        row = by_key[key]
        for col, exp in zip(columns, expected):
            require(close(row[col], exp), f"R33 delta mismatch: {key} / {col}: {row[col]} != {exp}")


def validate_claim_lock(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("r33a4_final_lock_sha256") == EXPECTED_CLAIM_LOCK, "R33 claim-lock SHA mismatch")
    require(payload.get("pristine_prospective_external_claim_allowed") is False, "R33 prospective-claim guard mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="SafeTTA repository root")
    args = parser.parse_args()
    out = args.root.resolve() / "outputs" / "R31_R33"

    paths = {
        "loao": out / "r32a_three_action_loao_summary.csv",
        "primary": out / "r33_polypgen_memo_primary_metrics.csv",
        "deltas": out / "r33_polypgen_memo_paired_deltas.csv",
        "claim": out / "r33_claim_freeze_summary.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required R31-R33 replay assets:\n" + "\n".join(missing))

    validate_loao(paths["loao"])
    print("R32A LOAO rows: 7/7 PASS")
    validate_r33_primary(paths["primary"])
    print("R33 primary rows: 4/4 PASS")
    validate_r33_deltas(paths["deltas"])
    print("R33 paired deltas: 6/6 PASS")
    validate_claim_lock(paths["claim"])
    print("R33 claim lock: PASS")
    print("GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY")


if __name__ == "__main__":
    main()
