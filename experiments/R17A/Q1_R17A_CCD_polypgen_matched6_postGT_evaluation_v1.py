#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_polypgen_matched6_postGT_evaluation_v1.py

SafeTTA R17A — post-GT evaluation of the faithful PolypGen SicTTA-CCD
baseline on the exact 6-state subset for which historical frozen SOURCE logits
exist.

WHY 6 STATES ONLY
-----------------
The newly regenerated 3-state SegFormer soft logits do not reproduce the
historical frozen SOURCE hard prediction sufficiently closely:
  exact historical hard-mask samples: 56 / 4596
  mismatch samples:                  4540 / 4596
  total mismatch pixels:             5,578,077

Therefore the regenerated SegFormer CCD scores are NOT joined to the historical
HARM outcomes. They are quarantined from the primary R17A comparison.

The valid matched panel is:
  DeepLabV3-R50 seeds 20260817/18/19
  PraNet         seeds 20260817/18/19
  1532 PolypGen images x 6 states = 9192 rows

PRE-GT score artifact
---------------------
R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv
SHA256:
85a2e5d913ddbb88a88c8df09b7ef88822963ea9a1451eda6c6254f95b151786

POST-GT evaluation
------------------
Join the already-locked CCD score to the already-frozen R10L3C PolypGen
evaluation panel on exact model-case identity, then compare against the frozen
SafeTTA score on the SAME rows.

Metrics:
  - pooled HARM AUROC / AUPRC
  - per-family AUROC / AUPRC
  - macro-over-2-family AUROC / AUPRC
  - paired physical-image bootstrap:
        Delta = SafeTTA - CCD
    for pooled and macro-2 metrics

Bootstrap:
  - 2000 reps
  - cluster = sample_id
  - center-stratified
  - all six model-state rows for a sampled physical image travel together

NO RETUNING
-----------
CCD score orientation: higher CCD = higher HARM risk (pre-registered)
SafeTTA score orientation: frozen, unchanged
HARM definition: frozen R10L3C harm_label
No threshold fitting, recalibration, feature selection, score reversal, or
target subset selection based on results.

This is a MATCHED-SUBSET published-baseline audit, not a replacement for the
paper's original 9-state PolypGen primary result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-09-Q1-R17A-CCD-POLYPGEN-MATCHED6-POSTGT-EVALUATION-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_polypgen_matched6_postGT_evaluation_v1")

CCD_REL = Path(
    r"outputs\Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1"
    r"\R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv"
)
CCD_SHA256 = "85a2e5d913ddbb88a88c8df09b7ef88822963ea9a1451eda6c6254f95b151786"

R10L3C_DIR_REL = Path(
    r"outputs\Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)
R10L3C_PANEL_NAME = "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
R10L3C_LOCK_NAME = "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"

EXPECTED_SAMPLES = 1532
EXPECTED_STATES = 6
EXPECTED_ROWS = EXPECTED_SAMPLES * EXPECTED_STATES

EXPECTED_FAMILIES = {
    "DeepLabV3-R50": {20260817, 20260818, 20260819},
    "PraNet": {20260817, 20260818, 20260819},
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260909


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
    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA mismatch\npath={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def load_r10l3c_panel(root: Path) -> Tuple[pd.DataFrame, dict, str]:
    d = root / R10L3C_DIR_REL
    panel_path = d / R10L3C_PANEL_NAME
    lock_path = d / R10L3C_LOCK_NAME

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    artifacts = lock.get("artifacts", {})
    meta = artifacts.get(R10L3C_PANEL_NAME)

    if not isinstance(meta, dict) or not meta.get("sha256"):
        raise RuntimeError(
            "R10L3C lock does not contain SHA metadata for the frozen evaluation panel."
        )

    panel_sha = validate_sha(
        panel_path,
        str(meta["sha256"]),
        "Frozen R10L3C evaluation panel",
    )

    df = pd.read_csv(panel_path, low_memory=False)

    required = {
        "sample_id",
        "center",
        "model_family",
        "model_state_id",
        "training_seed",
        "harm_label",
        "delta_dice",
        "frozen_safety_probability",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"R10L3C panel missing required columns: {missing}")

    if len(df) != EXPECTED_SAMPLES * 9:
        raise RuntimeError(
            f"Full frozen R10L3C rows={len(df)}, expected={EXPECTED_SAMPLES * 9}"
        )

    return df, lock, panel_sha


def validate_ccd_table(df: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "ccd_risk",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"CCD table missing required columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"CCD rows={len(df)}, expected={EXPECTED_ROWS}")

    if df["sample_id"].astype(str).nunique() != EXPECTED_SAMPLES:
        raise RuntimeError("CCD unique sample count mismatch.")

    states = (
        df[["model_family", "training_seed", "model_state_id"]]
        .drop_duplicates()
        .copy()
    )
    if len(states) != EXPECTED_STATES:
        raise RuntimeError(f"CCD states={len(states)}, expected={EXPECTED_STATES}")

    for family, seeds in EXPECTED_FAMILIES.items():
        sub = states[states["model_family"].astype(str) == family]
        observed = set(sub["training_seed"].astype(int).tolist())
        if observed != seeds:
            raise RuntimeError(
                f"CCD state mismatch for {family}: observed={sorted(observed)} "
                f"expected={sorted(seeds)}"
            )

    if (states["model_family"].astype(str) == "SegFormer-B0").any():
        raise RuntimeError("SegFormer must not appear in faithful matched-6 CCD table.")

    if df.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate CCD sample/model-state keys.")

    scores = df["ccd_risk"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError("CCD scores contain NaN/Inf.")


def build_matched_panel(ccd: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    allowed = []
    for family, seeds in EXPECTED_FAMILIES.items():
        for seed in seeds:
            allowed.append((family, seed))

    f = frozen[
        frozen.apply(
            lambda r: (str(r["model_family"]), int(r["training_seed"])) in allowed,
            axis=1,
        )
    ].copy()

    if len(f) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Frozen matched-6 rows={len(f)}, expected={EXPECTED_ROWS}"
        )

    if f.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Frozen matched-6 sample/model-state keys are not unique.")

    keys = ["sample_id", "model_family", "model_state_id", "training_seed"]

    merged = ccd.merge(
        f[
            keys
            + [
                "center",
                "harm_label",
                "delta_dice",
                "frozen_safety_probability",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Matched join rows={len(merged)}, expected={EXPECTED_ROWS}"
        )

    if merged["sample_id"].astype(str).nunique() != EXPECTED_SAMPLES:
        raise RuntimeError("Matched panel physical sample count mismatch.")

    # Every physical image must have exactly 6 rows.
    counts = merged.groupby("sample_id").size()
    if not (counts == EXPECTED_STATES).all():
        raise RuntimeError(
            "Not every PolypGen physical image has exactly six matched states."
        )

    # One center per physical image.
    center_counts = merged.groupby("sample_id")["center"].nunique()
    if not (center_counts == 1).all():
        raise RuntimeError("A physical image maps to multiple centers.")

    y = merged["harm_label"].to_numpy(dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError("harm_label is not binary.")
    if len(np.unique(y)) != 2:
        raise RuntimeError("Matched panel HARM labels contain only one class.")

    return merged


def metric_pair(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def family_metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for family in ["DeepLabV3-R50", "PraNet"]:
        sub = df[df["model_family"].astype(str) == family]
        m = metric_pair(
            sub["harm_label"].to_numpy(dtype=int),
            sub[score_col].to_numpy(dtype=float),
        )
        rows.append({
            "model_family": family,
            "rows": len(sub),
            "harm_rows": int(sub["harm_label"].sum()),
            **m,
        })
    return pd.DataFrame(rows)


def summary_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    pooled = metric_pair(
        df["harm_label"].to_numpy(dtype=int),
        df[score_col].to_numpy(dtype=float),
    )
    fam = family_metrics(df, score_col)

    return {
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
        "macro2_auroc": float(fam["auroc"].mean()),
        "macro2_auprc": float(fam["auprc"].mean()),
    }


def bootstrap_indices_by_center(df: pd.DataFrame, rng: np.random.Generator) -> List[str]:
    sample_center = (
        df[["sample_id", "center"]]
        .drop_duplicates()
        .sort_values(["center", "sample_id"])
    )

    sampled_ids = []
    for center, group in sample_center.groupby("center", sort=True):
        ids = group["sample_id"].astype(str).to_numpy()
        draw = rng.choice(ids, size=len(ids), replace=True)
        sampled_ids.extend(draw.tolist())

    return sampled_ids


def materialize_cluster_bootstrap(df: pd.DataFrame, sampled_ids: List[str]) -> pd.DataFrame:
    groups = {
        str(sid): g.copy()
        for sid, g in df.groupby("sample_id", sort=False)
    }

    parts = []
    for j, sid in enumerate(sampled_ids):
        g = groups[str(sid)].copy()
        # Duplicate sample clusters need a bootstrap-instance ID so downstream
        # grouping treats each draw as a distinct physical-cluster replicate.
        g["_bootstrap_cluster"] = f"{j:06d}::{sid}"
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def bootstrap_deltas(df: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    iterator = range(reps)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=reps,
            desc="Matched-6 paired center-stratified bootstrap",
            unit="rep",
            dynamic_ncols=True,
        )

    for rep in iterator:
        sampled_ids = bootstrap_indices_by_center(df, rng)
        b = materialize_cluster_bootstrap(df, sampled_ids)

        safe = summary_metrics(b, "frozen_safety_probability")
        ccd = summary_metrics(b, "ccd_risk")

        row = {"rep": rep}
        for metric in [
            "pooled_auroc",
            "pooled_auprc",
            "macro2_auroc",
            "macro2_auprc",
        ]:
            row[f"safetta_{metric}"] = safe[metric]
            row[f"ccd_{metric}"] = ccd[metric]
            row[f"delta_safetta_minus_ccd_{metric}"] = safe[metric] - ccd[metric]

        rows.append(row)

    return pd.DataFrame(rows)


def ci95(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bootstrap_reps", type=int, default=BOOTSTRAP_REPS)
    ap.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A POLYPGEN MATCHED-6 CCD POST-GT EVALUATION =====")
    print("Version:", VERSION)
    print("Dataset: PolypGen")
    print("Families: DeepLabV3-R50 + PraNet")
    print("States: 6/9 matched historical-soft-logit states")
    print("Expected rows:", EXPECTED_ROWS)
    print("CCD orientation: higher = higher HARM risk")
    print("SafeTTA orientation: frozen")
    print("Target tuning/recalibration: NO")
    print("SegFormer regenerated CCD included: NO")
    print()

    ccd_path = root / CCD_REL
    ccd_sha = validate_sha(ccd_path, CCD_SHA256, "Frozen 6-state CCD score table")
    ccd = pd.read_csv(ccd_path, low_memory=False)
    validate_ccd_table(ccd)

    frozen, r10_lock, frozen_panel_sha = load_r10l3c_panel(root)
    matched = build_matched_panel(ccd, frozen)

    # Lock exact joined input before computing/reporting metrics.
    joined_path = out_dir / "R17A_POLYPGEN_MATCHED6_LOCKED_JOIN_PANEL.csv"
    matched.to_csv(joined_path, index=False)
    joined_sha = sha256_file(joined_path)

    # Point metrics.
    safe_point = summary_metrics(matched, "frozen_safety_probability")
    ccd_point = summary_metrics(matched, "ccd_risk")

    point_rows = []
    for method, m in [("SafeTTA", safe_point), ("SicTTA-CCD", ccd_point)]:
        point_rows.append({
            "method": method,
            "rows": len(matched),
            "physical_samples": matched["sample_id"].nunique(),
            "families": 2,
            **m,
        })
    point = pd.DataFrame(point_rows)

    per_family = []
    for method, score_col in [
        ("SafeTTA", "frozen_safety_probability"),
        ("SicTTA-CCD", "ccd_risk"),
    ]:
        f = family_metrics(matched, score_col)
        f.insert(0, "method", method)
        per_family.append(f)
    per_family = pd.concat(per_family, ignore_index=True)

    # Paired physical-image bootstrap.
    boot = bootstrap_deltas(
        matched,
        reps=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed),
    )

    delta_rows = []
    for metric in [
        "pooled_auroc",
        "pooled_auprc",
        "macro2_auroc",
        "macro2_auprc",
    ]:
        col = f"delta_safetta_minus_ccd_{metric}"
        vals = boot[col].to_numpy(dtype=float)
        lo, hi = ci95(vals)
        point_delta = safe_point[metric] - ccd_point[metric]
        delta_rows.append({
            "metric": metric,
            "point_delta_safetta_minus_ccd": point_delta,
            "ci95_low": lo,
            "ci95_high": hi,
            "safetta_significantly_better_95ci": bool(lo > 0),
            "ccd_significantly_better_95ci": bool(hi < 0),
            "bootstrap_reps": int(args.bootstrap_reps),
            "bootstrap_seed": int(args.bootstrap_seed),
            "cluster": "sample_id",
            "center_stratified": True,
        })
    delta = pd.DataFrame(delta_rows)

    point_path = out_dir / "R17A_POLYPGEN_MATCHED6_POINT_METRICS.csv"
    family_path = out_dir / "R17A_POLYPGEN_MATCHED6_PER_FAMILY_METRICS.csv"
    boot_path = out_dir / "R17A_POLYPGEN_MATCHED6_PAIRED_BOOTSTRAP.csv"
    delta_path = out_dir / "R17A_POLYPGEN_MATCHED6_DELTA_CI95.csv"

    point.to_csv(point_path, index=False)
    per_family.to_csv(family_path, index=False)
    boot.to_csv(boot_path, index=False)
    delta.to_csv(delta_path, index=False)

    lock = {
        "status": "PASS",
        "decision": "PASS_R17A_POLYPGEN_MATCHED6_CCD_POSTGT_EVALUATION",
        "version": VERSION,
        "dataset": "PolypGen",
        "scope": {
            "families": ["DeepLabV3-R50", "PraNet"],
            "states": 6,
            "full_paper_states": 9,
            "rows": len(matched),
            "physical_samples": int(matched["sample_id"].nunique()),
            "segformer_included": False,
            "reason_segformer_excluded": (
                "regenerated SOURCE predictions materially mismatched historical "
                "frozen SOURCE state; historical soft logits unavailable"
            ),
        },
        "frozen_inputs": {
            "ccd_score_table": str(CCD_REL).replace("\\", "/"),
            "ccd_score_sha256": ccd_sha,
            "r10l3c_panel": str(
                (R10L3C_DIR_REL / R10L3C_PANEL_NAME)
            ).replace("\\", "/"),
            "r10l3c_panel_sha256": frozen_panel_sha,
            "matched_join_panel": joined_path.name,
            "matched_join_panel_sha256": joined_sha,
        },
        "evaluation": {
            "harm_label": "frozen R10L3C harm_label",
            "ccd_orientation": "higher = higher HARM risk",
            "safetta_orientation": "frozen higher = higher HARM risk",
            "target_recalibration": False,
            "target_threshold_selection": False,
            "target_feature_selection": False,
            "target_score_reversal": False,
            "paired_bootstrap_cluster": "sample_id",
            "paired_bootstrap_center_stratified": True,
            "bootstrap_reps": int(args.bootstrap_reps),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "artifacts": {
            point_path.name: sha256_file(point_path),
            family_path.name: sha256_file(family_path),
            boot_path.name: sha256_file(boot_path),
            delta_path.name: sha256_file(delta_path),
        },
        "interpretation": (
            "Matched-subset published-baseline audit only; do not substitute its "
            "2-family metrics for the manuscript's original 3-family/9-state "
            "PolypGen primary SafeTTA result."
        ),
    }

    lock_path = out_dir / "R17A_POLYPGEN_MATCHED6_EVALUATION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Human-readable report.
    safe = point[point["method"] == "SafeTTA"].iloc[0]
    ccdm = point[point["method"] == "SicTTA-CCD"].iloc[0]

    lines = [
        "===== R17A POLYPGEN MATCHED-6 CCD POST-GT EVALUATION REPORT =====",
        f"Version: {VERSION}",
        "Dataset: PolypGen",
        "Scope: DeepLabV3-R50 + PraNet only",
        f"Rows: {len(matched)}",
        f"Physical samples: {matched['sample_id'].nunique()}",
        "States: 6/9",
        "SegFormer included: NO",
        "",
        "WHY SEGFOMER IS EXCLUDED:",
        "- Historical SegFormer soft logits are unavailable.",
        "- New SegFormer regeneration mismatched the historical hard SOURCE state",
        "  on 4540/4596 state-image pairs with 5,578,077 differing pixels.",
        "- Therefore its regenerated CCD is not joined to historical HARM outcomes.",
        "",
        "POINT METRICS:",
        f"SafeTTA pooled AUROC={safe['pooled_auroc']:.9f} AUPRC={safe['pooled_auprc']:.9f}",
        f"CCD     pooled AUROC={ccdm['pooled_auroc']:.9f} AUPRC={ccdm['pooled_auprc']:.9f}",
        f"SafeTTA macro2 AUROC={safe['macro2_auroc']:.9f} AUPRC={safe['macro2_auprc']:.9f}",
        f"CCD     macro2 AUROC={ccdm['macro2_auroc']:.9f} AUPRC={ccdm['macro2_auprc']:.9f}",
        "",
        "PAIRED CLUSTERED BOOTSTRAP DELTAS (SafeTTA - CCD):",
    ]

    for r in delta.itertuples(index=False):
        lines.append(
            f"{r.metric}: delta={r.point_delta_safetta_minus_ccd:+.9f} "
            f"95%CI=[{r.ci95_low:+.9f},{r.ci95_high:+.9f}] "
            f"SafeTTA_better={r.safetta_significantly_better_95ci}"
        )

    lines += [
        "",
        "INTERPRETATION:",
        "- This is a faithful matched-6-state baseline comparison.",
        "- It is NOT the paper's original full 9-state PolypGen primary result.",
        "- No target tuning/recalibration/score reversal was performed.",
        "",
        "GATE=PASS_R17A_POLYPGEN_MATCHED6_CCD_POSTGT_EVALUATION",
        "NEXT=REVIEW_CCD_EFFECT_SIZE_THEN_DECIDE_WHETHER_TO_ADD_AS_PUBLISHED_BASELINE_SENSITIVITY",
    ]

    report_path = out_dir / "R17A_POLYPGEN_MATCHED6_CCD_EVALUATION_REPORT.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    for line in lines[-18:]:
        print(line)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
