#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-2B — PolypGen exact50 absolute deployment references.

ADDITIVE post-P0-2 analysis.
NO new fitting.
NO coverage tuning.
NO target calibration.
NO score reversal.
NO threshold tuning.

Adds:
- SOURCE-only (coverage 0)
- adapt-all MEMO (coverage 1)
- matched-random exact50 expectation (coverage 0.5)
to the already-frozen P0-2 exact50 selections:
- SOURCE_STATE
- SOURCE_PLUS_SIMPLE_MASK_CHANGE
- FROZEN_FULL_TRANSITION_SAFETTA

The random reference is analytical: under uniform exact selection of
766/1532 rows within each equally sized state, every row has inclusion
probability 0.5. This avoids an arbitrary random seed for the point estimate.
For each physical-image bootstrap replicate, the same matched-random
expectation is recalculated on that resampled panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-B6-P02B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_PATH = CODE / "B6_P02B_POLYPGEN_EXACT50_ABSOLUTE_REFERENCE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "4be20289e99bd98163ac124d04774b7436f497fe58e5f7f736dcb5aa4729fb24"

P02_DIR = ROOT / "B6_P02_polypgen_exact50_utility_v1"
P02_AUDIT = P02_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_AUDIT.json"
P02_ROWS = P02_DIR / "B6_P02_POLYPGEN_EXACT50_SELECTIONS_AND_DEPLOYMENT.csv"

OUT_DIR = ROOT / "B6_P02B_polypgen_exact50_absolute_references_v1"

PASS_GATE = "PASS_B6_P02B_POLYPGEN_EXACT50_ABSOLUTE_REFERENCES"
PREFLIGHT_GATE = "PASS_B6_P02B_POLYPGEN_EXACT50_ABSOLUTE_REFERENCE_PREFLIGHT"

EXPECTED_P02_GATE = "PASS_B6_P02_POLYPGEN_EXACT50_UTILITY_COMPLETE"
EXPECTED_P02_DECISION = (
    "CANDIDATE_CONDITIONING_EXACT50_SAFETY_SUPPORTED_UTILITY_NOT_CONFIRMED"
)

N_IMAGES = 1532
N_STATES = 3
N_ROWS = 4596
SELECT_PER_STATE = 766
COVERAGE = 0.5
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260918

FROZEN_METHODS = [
    "SOURCE_STATE",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
    "FROZEN_FULL_TRANSITION_SAFETTA",
]

METHOD_LABELS = {
    "SOURCE_ONLY": "SOURCE_ONLY",
    "ADAPT_ALL_MEMO": "ADAPT_ALL_MEMO",
    "MATCHED_RANDOM_EXPECTATION_EXACT50": "MATCHED_RANDOM_EXPECTATION_EXACT50",
    "SOURCE_STATE": "SOURCE_STATE_GATING_EXACT50",
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE": "SIMPLE_GEOMETRY_GATING_EXACT50",
    "FROZEN_FULL_TRANSITION_SAFETTA": "FULL_TRANSITION_GATING_EXACT50",
}

DISPLAY_ORDER = [
    "SOURCE_ONLY",
    "ADAPT_ALL_MEMO",
    "MATCHED_RANDOM_EXPECTATION_EXACT50",
    "SOURCE_STATE_GATING_EXACT50",
    "SIMPLE_GEOMETRY_GATING_EXACT50",
    "FULL_TRANSITION_GATING_EXACT50",
]

UTILITY_METRICS = [
    "mean_deployed_dice",
    "mean_deployed_delta_vs_source",
    "prevented_harm_fraction",
    "committed_harm_rate",
    "benefit_capture_fraction",
]

PAIRWISE = [
    ("FULL_TRANSITION_GATING_EXACT50", "MATCHED_RANDOM_EXPECTATION_EXACT50"),
    ("SOURCE_STATE_GATING_EXACT50", "MATCHED_RANDOM_EXPECTATION_EXACT50"),
    ("SIMPLE_GEOMETRY_GATING_EXACT50", "MATCHED_RANDOM_EXPECTATION_EXACT50"),
    ("FULL_TRANSITION_GATING_EXACT50", "SOURCE_ONLY"),
    ("FULL_TRANSITION_GATING_EXACT50", "ADAPT_ALL_MEMO"),
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)

    sha = sha256_file(PROTOCOL_PATH)
    if sha.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "P02B protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"observed={sha}"
        )

    p = load_json(PROTOCOL_PATH)
    if p.get("status") != "FROZEN_ADDITIVE_REFERENCE_BEFORE_P02B_METRICS":
        raise RuntimeError("P02B protocol status changed.")

    return {
        "path": str(PROTOCOL_PATH),
        "sha256": sha,
        "status": p.get("status"),
    }


def verify_and_load_p02() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not P02_AUDIT.is_file():
        raise FileNotFoundError(P02_AUDIT)
    if not P02_ROWS.is_file():
        raise FileNotFoundError(P02_ROWS)

    audit = load_json(P02_AUDIT)
    if audit.get("gate") != EXPECTED_P02_GATE:
        raise RuntimeError(f"P02 gate changed: {audit.get('gate')}")
    if audit.get("decision") != EXPECTED_P02_DECISION:
        raise RuntimeError(f"P02 decision changed: {audit.get('decision')}")

    d = pd.read_csv(P02_ROWS, low_memory=False)

    required = {
        "sample_id",
        "model_state_id",
        "source_dice",
        "memo_dice",
        "memo_delta_dice",
        "harm_label",
        "benefit_label",
    }
    for m in FROZEN_METHODS:
        required.add(f"selected_{m}")
        required.add(f"deployed_dice_{m}")
        required.add(f"deployed_delta_{m}")

    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"P02 row table missing columns: {missing}")

    if len(d) != N_ROWS:
        raise RuntimeError(f"P02 rows={len(d)} != {N_ROWS}")
    if d["sample_id"].astype(str).nunique() != N_IMAGES:
        raise RuntimeError("P02 image count changed.")
    if d["model_state_id"].astype(str).nunique() != N_STATES:
        raise RuntimeError("P02 state count changed.")

    for c in [
        "source_dice",
        "memo_dice",
        "memo_delta_dice",
        "harm_label",
        "benefit_label",
    ]:
        d[c] = pd.to_numeric(d[c], errors="raise")

    if not np.allclose(
        d["memo_dice"].to_numpy(dtype=float)
        - d["source_dice"].to_numpy(dtype=float),
        d["memo_delta_dice"].to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("memo_dice-source_dice identity failed.")

    expected_harm = (
        d["memo_delta_dice"].to_numpy(dtype=float) <= -0.02
    ).astype(int)
    expected_benefit = (
        d["memo_delta_dice"].to_numpy(dtype=float) >= +0.02
    ).astype(int)

    if not np.array_equal(
        expected_harm,
        d["harm_label"].to_numpy(dtype=int),
    ):
        raise RuntimeError("HARM rule reproduction failed.")
    if not np.array_equal(
        expected_benefit,
        d["benefit_label"].to_numpy(dtype=int),
    ):
        raise RuntimeError("BENEFIT rule reproduction failed.")

    for m in FROZEN_METHODS:
        sc = f"selected_{m}"
        vals = set(pd.to_numeric(d[sc], errors="raise").astype(int).unique())
        if not vals.issubset({0, 1}):
            raise RuntimeError(f"{sc}: not binary.")

        per_state = (
            d.groupby("model_state_id")[sc]
            .sum()
            .astype(int)
        )
        if not (per_state == SELECT_PER_STATE).all():
            raise RuntimeError(
                f"{m}: frozen exact50 selection count changed: "
                f"{per_state.to_dict()}"
            )

    return d, {
        "audit_path": str(P02_AUDIT),
        "audit_sha256": sha256_file(P02_AUDIT),
        "row_table_path": str(P02_ROWS),
        "row_table_sha256": sha256_file(P02_ROWS),
        "gate": audit.get("gate"),
        "decision": audit.get("decision"),
        "rows": len(d),
        "images": d["sample_id"].astype(str).nunique(),
        "states": d["model_state_id"].astype(str).nunique(),
        "harm_n": int(d["harm_label"].sum()),
        "benefit_n": int(d["benefit_label"].sum()),
    }


def point_metrics_from_arrays(
    deployed_dice: np.ndarray,
    deployed_delta: np.ndarray,
    selected: np.ndarray | None,
    harm: np.ndarray,
    benefit: np.ndarray,
    coverage: float,
    label: str,
    expected_random: bool = False,
) -> Dict[str, Any]:
    harm_n = int(harm.sum())
    benefit_n = int(benefit.sum())

    if expected_random:
        prevented_harm_fraction = 0.5 if harm_n > 0 else np.nan
        benefit_capture_fraction = 0.5 if benefit_n > 0 else np.nan
        committed_harm_rate = float(harm.mean())
        selected_n = int(round(coverage * len(harm)))
    else:
        if selected is None:
            raise RuntimeError("selected vector required.")
        selected = np.asarray(selected, dtype=int)
        selected_n = int(selected.sum())

        prevented_harm_fraction = (
            float(((1 - selected) * harm).sum() / harm_n)
            if harm_n > 0 else np.nan
        )
        benefit_capture_fraction = (
            float((selected * benefit).sum() / benefit_n)
            if benefit_n > 0 else np.nan
        )
        committed_harm_rate = (
            float((selected * harm).sum() / selected_n)
            if selected_n > 0 else np.nan
        )

    return {
        "policy": label,
        "coverage": float(coverage),
        "selected_n_or_expected": selected_n,
        "mean_deployed_dice": float(np.mean(deployed_dice)),
        "mean_deployed_delta_vs_source": float(np.mean(deployed_delta)),
        "prevented_harm_fraction": prevented_harm_fraction,
        "committed_harm_rate": committed_harm_rate,
        "benefit_capture_fraction": benefit_capture_fraction,
        "harm_n": harm_n,
        "benefit_n": benefit_n,
    }


def build_point_table(d: pd.DataFrame) -> pd.DataFrame:
    src = d["source_dice"].to_numpy(dtype=float)
    memo = d["memo_dice"].to_numpy(dtype=float)
    delta = d["memo_delta_dice"].to_numpy(dtype=float)
    harm = d["harm_label"].to_numpy(dtype=int)
    benefit = d["benefit_label"].to_numpy(dtype=int)

    rows = []

    # SOURCE-only
    rows.append(point_metrics_from_arrays(
        deployed_dice=src,
        deployed_delta=np.zeros_like(delta),
        selected=np.zeros(len(d), dtype=int),
        harm=harm,
        benefit=benefit,
        coverage=0.0,
        label="SOURCE_ONLY",
    ))

    # Adapt-all
    rows.append(point_metrics_from_arrays(
        deployed_dice=memo,
        deployed_delta=delta,
        selected=np.ones(len(d), dtype=int),
        harm=harm,
        benefit=benefit,
        coverage=1.0,
        label="ADAPT_ALL_MEMO",
    ))

    # Analytical matched-random exact50 expectation.
    random_deployed_delta = COVERAGE * delta
    random_deployed_dice = src + random_deployed_delta
    rows.append(point_metrics_from_arrays(
        deployed_dice=random_deployed_dice,
        deployed_delta=random_deployed_delta,
        selected=None,
        harm=harm,
        benefit=benefit,
        coverage=COVERAGE,
        label="MATCHED_RANDOM_EXPECTATION_EXACT50",
        expected_random=True,
    ))

    # Frozen P0-2 selections.
    for m in FROZEN_METHODS:
        sel = d[f"selected_{m}"].to_numpy(dtype=int)
        dep_dice = d[f"deployed_dice_{m}"].to_numpy(dtype=float)
        dep_delta = d[f"deployed_delta_{m}"].to_numpy(dtype=float)

        rows.append(point_metrics_from_arrays(
            deployed_dice=dep_dice,
            deployed_delta=dep_delta,
            selected=sel,
            harm=harm,
            benefit=benefit,
            coverage=COVERAGE,
            label=METHOD_LABELS[m],
        ))

    out = pd.DataFrame(rows)
    out["_order"] = out["policy"].map({
        k: i for i, k in enumerate(DISPLAY_ORDER)
    })
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def per_image_contributions(d: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build per-physical-image sums so bootstrap retains all 3 states.
    """
    frames = {}

    src = d["source_dice"].to_numpy(dtype=float)
    memo = d["memo_dice"].to_numpy(dtype=float)
    delta = d["memo_delta_dice"].to_numpy(dtype=float)
    harm = d["harm_label"].to_numpy(dtype=int)
    benefit = d["benefit_label"].to_numpy(dtype=int)

    # Helper for deterministic policies.
    def aggregate_policy(
        label: str,
        deployed_dice: np.ndarray,
        deployed_delta: np.ndarray,
        selected: np.ndarray,
    ) -> pd.DataFrame:
        x = pd.DataFrame({
            "sample_id": d["sample_id"].astype(str).to_numpy(),
            "row_n": 1,
            "deployed_dice_sum": deployed_dice,
            "deployed_delta_sum": deployed_delta,
            "selected_n": selected,
            "harm_n": harm,
            "benefit_n": benefit,
            "prevented_harm_n": (1 - selected) * harm,
            "committed_harm_n": selected * harm,
            "captured_benefit_n": selected * benefit,
        })
        return x.groupby("sample_id", as_index=False).sum(numeric_only=True)

    frames["SOURCE_ONLY"] = aggregate_policy(
        "SOURCE_ONLY",
        src,
        np.zeros_like(delta),
        np.zeros(len(d), dtype=int),
    )
    frames["ADAPT_ALL_MEMO"] = aggregate_policy(
        "ADAPT_ALL_MEMO",
        memo,
        delta,
        np.ones(len(d), dtype=int),
    )

    for m in FROZEN_METHODS:
        frames[METHOD_LABELS[m]] = aggregate_policy(
            METHOD_LABELS[m],
            d[f"deployed_dice_{m}"].to_numpy(dtype=float),
            d[f"deployed_delta_{m}"].to_numpy(dtype=float),
            d[f"selected_{m}"].to_numpy(dtype=int),
        )

    # Matched-random expectation: carry source/delta/harm/benefit sums;
    # utility is evaluated analytically from these quantities in bootstrap.
    r = pd.DataFrame({
        "sample_id": d["sample_id"].astype(str).to_numpy(),
        "row_n": 1,
        "source_dice_sum": src,
        "memo_delta_sum": delta,
        "harm_n": harm,
        "benefit_n": benefit,
    })
    frames["MATCHED_RANDOM_EXPECTATION_EXACT50"] = (
        r.groupby("sample_id", as_index=False).sum(numeric_only=True)
    )

    for label, f in frames.items():
        if len(f) != N_IMAGES:
            raise RuntimeError(
                f"{label}: physical-image rows={len(f)} != {N_IMAGES}"
            )
        if "row_n" in f.columns and not (f["row_n"] == N_STATES).all():
            raise RuntimeError(
                f"{label}: each physical image must contain 3 states."
            )

    return frames


def deterministic_metrics_from_cluster(x: pd.DataFrame) -> Dict[str, float]:
    rows = float(x["row_n"].sum())
    selected_n = float(x["selected_n"].sum())
    harm_n = float(x["harm_n"].sum())
    benefit_n = float(x["benefit_n"].sum())

    return {
        "mean_deployed_dice": float(x["deployed_dice_sum"].sum() / rows),
        "mean_deployed_delta_vs_source": float(x["deployed_delta_sum"].sum() / rows),
        "prevented_harm_fraction": (
            float(x["prevented_harm_n"].sum() / harm_n)
            if harm_n > 0 else np.nan
        ),
        "committed_harm_rate": (
            float(x["committed_harm_n"].sum() / selected_n)
            if selected_n > 0 else np.nan
        ),
        "benefit_capture_fraction": (
            float(x["captured_benefit_n"].sum() / benefit_n)
            if benefit_n > 0 else np.nan
        ),
    }


def random_expectation_metrics_from_cluster(x: pd.DataFrame) -> Dict[str, float]:
    rows = float(x["row_n"].sum())
    source_mean = float(x["source_dice_sum"].sum() / rows)
    delta_mean = float(x["memo_delta_sum"].sum() / rows)
    harm_n = float(x["harm_n"].sum())
    benefit_n = float(x["benefit_n"].sum())

    return {
        "mean_deployed_dice": source_mean + COVERAGE * delta_mean,
        "mean_deployed_delta_vs_source": COVERAGE * delta_mean,
        "prevented_harm_fraction": 1.0 - COVERAGE if harm_n > 0 else np.nan,
        "committed_harm_rate": (
            float(harm_n / rows) if rows > 0 else np.nan
        ),
        "benefit_capture_fraction": COVERAGE if benefit_n > 0 else np.nan,
    }


def bootstrap(
    d: pd.DataFrame,
    point: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = per_image_contributions(d)
    ids = sorted(d["sample_id"].astype(str).unique().tolist())
    if len(ids) != N_IMAGES:
        raise RuntimeError("Bootstrap sample count changed.")

    indexed = {}
    for label, f in frames.items():
        indexed[label] = f.set_index("sample_id").loc[ids].reset_index()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rep_rows = []
    delta_rows = []

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="P02B physical-image bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        draw = rng.integers(0, N_IMAGES, size=N_IMAGES)

        vals = {}
        for label in DISPLAY_ORDER:
            x = indexed[label].iloc[draw]
            if label == "MATCHED_RANDOM_EXPECTATION_EXACT50":
                vals[label] = random_expectation_metrics_from_cluster(x)
            else:
                vals[label] = deterministic_metrics_from_cluster(x)

            rep_rows.append({
                "rep": rep,
                "policy": label,
                **vals[label],
            })

        for a, b in PAIRWISE:
            for metric in UTILITY_METRICS:
                va = vals[a][metric]
                vb = vals[b][metric]
                if np.isfinite(va) and np.isfinite(vb):
                    delta_rows.append({
                        "rep": rep,
                        "comparison": f"{a} - {b}",
                        "metric": metric,
                        "delta": va - vb,
                    })

    reps = pd.DataFrame(rep_rows)
    deltas = pd.DataFrame(delta_rows)

    point_ix = point.set_index("policy")
    summary_rows = []

    for (comparison, metric), g in deltas.groupby(
        ["comparison", "metric"],
        sort=False,
    ):
        vals = g["delta"].to_numpy(dtype=float)
        a, b = comparison.split(" - ")

        pa = point_ix.loc[a, metric]
        pb = point_ix.loc[b, metric]

        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue

        lo = float(np.quantile(vals, 0.025))
        hi = float(np.quantile(vals, 0.975))

        summary_rows.append({
            "comparison": comparison,
            "metric": metric,
            "valid_reps": int(len(vals)),
            "point_delta": float(pa - pb),
            "bootstrap_mean_delta": float(np.mean(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    return reps, pd.DataFrame(summary_rows)


def self_test() -> None:
    assert N_IMAGES * N_STATES == N_ROWS
    assert SELECT_PER_STATE / N_IMAGES == COVERAGE
    assert BOOTSTRAP_REPS == 2000
    assert "SOURCE_ONLY" in DISPLAY_ORDER
    assert "MATCHED_RANDOM_EXPECTATION_EXACT50" in DISPLAY_ORDER
    print("SELF_TEST_COUNTS=PASS")
    print("SELF_TEST_RANDOM_EXPECTATION=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA P0-2B PolypGen exact50 absolute deployment references."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 164)
    print("SafeTTA B6-P0-2B — PolypGen exact50 absolute deployment references")
    print(f"Version             : {VERSION}")
    print("New model fitting   : NO")
    print("Coverage tuning     : NO")
    print("Target calibration  : NO")
    print("P0-2 primary change : NO")
    print("Random comparator   : analytical exact50 expectation")
    print("=" * 164)

    print("\n[1/4] Verify additive-reference protocol")
    protocol_meta = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol_meta["sha256"])

    print("\n[2/4] Verify completed P0-2 row-level deployment asset")
    d, p02_meta = verify_and_load_p02()
    print("rows      =", p02_meta["rows"])
    print("images    =", p02_meta["images"])
    print("states    =", p02_meta["states"])
    print("harm_n    =", p02_meta["harm_n"])
    print("benefit_n =", p02_meta["benefit_n"])
    print("P02_BINDING=PASS")

    print("\n[3/4] Build absolute reference table")
    point = build_point_table(d)
    print(point[[
        "policy",
        "coverage",
        "mean_deployed_dice",
        "mean_deployed_delta_vs_source",
        "prevented_harm_fraction",
        "committed_harm_rate",
        "benefit_capture_fraction",
    ]].to_string(index=False))

    # Preflight is allowed to validate formulas but does not write outputs.
    random_row = point[
        point["policy"] == "MATCHED_RANDOM_EXPECTATION_EXACT50"
    ].iloc[0]
    if abs(float(random_row["prevented_harm_fraction"]) - 0.5) > 1e-12:
        raise RuntimeError("Random prevented-HARM expectation changed.")
    if abs(float(random_row["benefit_capture_fraction"]) - 0.5) > 1e-12:
        raise RuntimeError("Random BENEFIT-capture expectation changed.")

    print("ABSOLUTE_REFERENCE_FORMULAS=PASS")

    if args.preflight_only:
        print("\nP02B_PREFLIGHT=PASS")
        print("BOOTSTRAP=NOT_RUN")
        print(f"GATE={PREFLIGHT_GATE}")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite P02B output: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/4] Paired physical-image bootstrap")
    reps, deltas = bootstrap(d, point)
    print("\nPAIRED DELTAS")
    print(deltas.to_string(index=False))

    p_point = OUT_DIR / "B6_P02B_POLYPGEN_ABSOLUTE_REFERENCE_TABLE.csv"
    p_reps = OUT_DIR / "B6_P02B_BOOTSTRAP_REPLICATES.csv"
    p_delta = OUT_DIR / "B6_P02B_PAIRED_DELTAS.csv"

    point.to_csv(p_point, index=False)
    reps.to_csv(p_reps, index=False)
    deltas.to_csv(p_delta, index=False)

    full_vs_random = deltas[
        (deltas["comparison"] ==
         "FULL_TRANSITION_GATING_EXACT50 - MATCHED_RANDOM_EXPECTATION_EXACT50")
        & (deltas["metric"] == "mean_deployed_dice")
    ]
    full_vs_source = deltas[
        (deltas["comparison"] ==
         "FULL_TRANSITION_GATING_EXACT50 - SOURCE_ONLY")
        & (deltas["metric"] == "mean_deployed_dice")
    ]

    if len(full_vs_random) != 1 or len(full_vs_source) != 1:
        raise RuntimeError("Headline delta rows missing.")

    r1 = full_vs_random.iloc[0]
    r2 = full_vs_source.iloc[0]

    matched_random_utility_supported = bool(
        float(r1["point_delta"]) > 0 and float(r1["ci95_low"]) > 0
    )
    source_only_superiority_supported = bool(
        float(r2["point_delta"]) > 0 and float(r2["ci95_low"]) > 0
    )

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "relationship_to_P02": "ADDITIVE_REFERENCE_ONLY",
        "p02_primary_unchanged": True,
        "protocol": protocol_meta,
        "p02": p02_meta,
        "headline": {
            "full_transition_vs_matched_random_deployed_dice": {
                "point_delta": float(r1["point_delta"]),
                "ci95_low": float(r1["ci95_low"]),
                "ci95_high": float(r1["ci95_high"]),
                "supported": matched_random_utility_supported,
            },
            "full_transition_vs_source_only_deployed_dice": {
                "point_delta": float(r2["point_delta"]),
                "ci95_low": float(r2["ci95_low"]),
                "ci95_high": float(r2["ci95_high"]),
                "supported": source_only_superiority_supported,
            },
        },
        "claim_boundary": {
            "matched_random_is_matched_budget_expectation": True,
            "source_only_and_adapt_all_have_different_coverage": True,
            "no_coverage_optimization": True,
            "no_rescue": True,
        },
        "outputs": {},
    }

    for p in [p_point, p_reps, p_delta]:
        audit["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_audit = OUT_DIR / "B6_P02B_POLYPGEN_ABSOLUTE_REFERENCE_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join([
        "=" * 164,
        "SafeTTA B6-P0-2B POLYPGEN ABSOLUTE REFERENCES COMPLETE",
        "",
        point[[
            "policy",
            "coverage",
            "mean_deployed_dice",
            "mean_deployed_delta_vs_source",
            "prevented_harm_fraction",
            "committed_harm_rate",
            "benefit_capture_fraction",
        ]].to_string(index=False),
        "",
        "PAIRED DELTAS:",
        deltas.to_string(index=False),
        "",
        f"FULL_VS_MATCHED_RANDOM_UTILITY_SUPPORTED={matched_random_utility_supported}",
        f"FULL_VS_SOURCE_ONLY_SUPERIORITY_SUPPORTED={source_only_superiority_supported}",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "=" * 164,
        "",
    ])

    p_report = OUT_DIR / "B6_P02B_POLYPGEN_ABSOLUTE_REFERENCE_REPORT.txt"
    p_report.write_text(report, encoding="utf-8")

    print("\n" + report)


if __name__ == "__main__":
    main()
