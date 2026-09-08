#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_polypgen_deeplab3_fourway_postGT_evaluation_v1.py

SafeTTA R17A — final apples-to-apples PolypGen DeepLabV3-R50 3-state
published-reliability baseline comparison.

IDENTICAL ROWS FOR ALL METHODS
------------------------------
PolypGen:
  1532 physical images
  DeepLabV3-R50 seeds 20260817/18/19
  4596 model-case rows

Methods:
  1) SafeTTA                    frozen paper score
  2) SicTTA-CCD                 frozen historical SOURCE-logit baseline
  3) TEGDA-ADIC                 faithful native-dropout baseline
  4) MC-dropout predictive entropy
                               same 10 stochastic passes as ADIC

WHY THIS IS THE PRIMARY R17A FOUR-WAY SENSITIVITY PANEL
-------------------------------------------------------
All four methods are evaluated on the exact same model states and physical
images. No regenerated SegFormer state is used, and PraNet is excluded from
ADIC/MC because the exact frozen PraNet model has no native torch.nn.Dropout.

FROZEN PRE-GT INPUTS
--------------------
CCD 6-state score table:
  SHA256 = 85a2e5d913ddbb88a88c8df09b7ef88822963ea9a1451eda6c6254f95b151786
  This script subsets only DeepLabV3-R50 seeds 20260817/18/19.

ADIC/MC DeepLab3 score table:
  SHA256 = 223f1496e9cc56736d752f57f38d8edae64f20c47bcb6d1b48c243775081cd52

Frozen post-GT outcome/SafeTTA panel:
  R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv
  Its SHA is read from the frozen R10L3C lock and checked before use.

NO RETUNING
-----------
No target recalibration
No threshold fitting
No feature selection
No target score reversal
No subset selection based on outcomes

FIXED SCORE DIRECTIONS
----------------------
SafeTTA: higher = higher HARM risk
CCD:     higher = higher HARM risk
ADIC:    adic_harm_risk = -adic_quality; higher = higher HARM risk
MC:      higher predictive entropy = higher HARM risk

METRICS
-------
For each method:
  - pooled AUROC / AUPRC over 4596 rows
  - macro-over-3-seed AUROC / AUPRC
  - per-seed AUROC / AUPRC

Paired bootstrap:
  - 2000 reps
  - physical cluster = sample_id
  - center-stratified
  - all three DeepLab state rows for a sampled image move together
  - delta = SafeTTA - comparator
  - 95% percentile CI

This is a sensitivity / published-baseline defense panel. It does NOT replace
the manuscript's original full 9-state PolypGen primary result.
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


VERSION = "2026-09-09-Q1-R17A-POLYPGEN-DEEPLAB3-FOURWAY-POSTGT-EVALUATION-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_polypgen_deeplab3_fourway_postGT_evaluation_v1")

CCD_REL = Path(
    r"outputs\Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1"
    r"\R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv"
)
CCD_SHA256 = "85a2e5d913ddbb88a88c8df09b7ef88822963ea9a1451eda6c6254f95b151786"

ADIC_REL = Path(
    r"outputs\Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2"
    r"\R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK.csv"
)
ADIC_SHA256 = "223f1496e9cc56736d752f57f38d8edae64f20c47bcb6d1b48c243775081cd52"

R10L3C_DIR_REL = Path(
    r"outputs\Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
)
R10L3C_PANEL_NAME = "R10L3C_LOCKED_SCORE_GT_EVALUATION_PANEL.csv"
R10L3C_LOCK_NAME = "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"

FAMILY = "DeepLabV3-R50"
SEEDS = (20260817, 20260818, 20260819)
EXPECTED_SAMPLES = 1532
EXPECTED_STATES = 3
EXPECTED_ROWS = EXPECTED_SAMPLES * EXPECTED_STATES

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260909

METHODS = {
    "SafeTTA": "frozen_safety_probability",
    "SicTTA-CCD": "ccd_risk",
    "TEGDA-ADIC": "adic_harm_risk",
    "MC-dropout": "mc_predictive_entropy_risk",
}


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


def load_frozen_r10_panel(root: Path) -> Tuple[pd.DataFrame, str]:
    d = root / R10L3C_DIR_REL
    panel_path = d / R10L3C_PANEL_NAME
    lock_path = d / R10L3C_LOCK_NAME

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    meta = lock.get("artifacts", {}).get(R10L3C_PANEL_NAME)
    if not isinstance(meta, dict) or not meta.get("sha256"):
        raise RuntimeError("Frozen R10L3C lock lacks panel SHA metadata.")

    panel_sha = validate_sha(
        panel_path,
        str(meta["sha256"]),
        "Frozen R10L3C panel",
    )
    df = pd.read_csv(panel_path, low_memory=False)

    required = {
        "sample_id", "center", "model_family", "model_state_id",
        "training_seed", "harm_label", "delta_dice",
        "frozen_safety_probability",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"R10L3C panel missing columns: {missing}")

    if len(df) != EXPECTED_SAMPLES * 9:
        raise RuntimeError(
            f"R10L3C full rows={len(df)}, expected={EXPECTED_SAMPLES * 9}"
        )

    sub = df[
        (df["model_family"].astype(str) == FAMILY)
        & (df["training_seed"].astype(int).isin(SEEDS))
    ].copy()

    if len(sub) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Frozen DeepLab3 rows={len(sub)}, expected={EXPECTED_ROWS}"
        )
    return sub, panel_sha


def validate_and_subset_ccd(root: Path) -> Tuple[pd.DataFrame, str]:
    p = root / CCD_REL
    sha = validate_sha(p, CCD_SHA256, "Frozen CCD score table")
    df = pd.read_csv(p, low_memory=False)

    required = {
        "sample_id", "center", "model_family", "model_state_id",
        "training_seed", "ccd_risk",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"CCD table missing columns: {missing}")

    sub = df[
        (df["model_family"].astype(str) == FAMILY)
        & (df["training_seed"].astype(int).isin(SEEDS))
    ].copy()

    if len(sub) != EXPECTED_ROWS:
        raise RuntimeError(f"CCD DeepLab3 rows={len(sub)}, expected={EXPECTED_ROWS}")
    if sub.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("CCD DeepLab3 duplicate sample/model-state keys.")
    return sub, sha


def validate_adic(root: Path) -> Tuple[pd.DataFrame, str]:
    p = root / ADIC_REL
    sha = validate_sha(p, ADIC_SHA256, "Frozen ADIC/MC score table")
    df = pd.read_csv(p, low_memory=False)

    required = {
        "sample_id", "center", "model_family", "model_state_id",
        "training_seed", "adic_quality", "adic_harm_risk",
        "mc_predictive_entropy_risk",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"ADIC/MC table missing columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"ADIC/MC rows={len(df)}, expected={EXPECTED_ROWS}")
    if df["sample_id"].astype(str).nunique() != EXPECTED_SAMPLES:
        raise RuntimeError("ADIC/MC unique sample count mismatch.")
    if set(df["training_seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("ADIC/MC seed set mismatch.")
    if set(df["model_family"].astype(str)) != {FAMILY}:
        raise RuntimeError("ADIC/MC contains unexpected model family.")
    if df.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("ADIC/MC duplicate sample/model-state keys.")

    # Exact orientation identity, no target data involved.
    if not np.array_equal(
        df["adic_harm_risk"].to_numpy(dtype=float),
        -df["adic_quality"].to_numpy(dtype=float),
    ):
        raise RuntimeError("ADIC harm-risk orientation identity failed.")

    return df, sha


def merge_fourway(
    frozen: pd.DataFrame,
    ccd: pd.DataFrame,
    adic: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["sample_id", "model_family", "model_state_id", "training_seed"]

    m = frozen[
        keys + [
            "center", "harm_label", "delta_dice", "frozen_safety_probability"
        ]
    ].merge(
        ccd[keys + ["center", "ccd_risk"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_frozen", "_ccd"),
    )

    if len(m) != EXPECTED_ROWS:
        raise RuntimeError(f"Frozen+CCD join rows={len(m)}, expected={EXPECTED_ROWS}")

    if not np.array_equal(
        m["center_frozen"].astype(str).str.strip().to_numpy(),
        m["center_ccd"].astype(str).str.strip().to_numpy(),
    ):
        raise RuntimeError("Frozen-vs-CCD center mismatch.")
    m["center"] = m["center_frozen"].astype(str).str.strip()
    m = m.drop(columns=["center_frozen", "center_ccd"])

    m = m.merge(
        adic[
            keys + [
                "center", "adic_harm_risk",
                "mc_predictive_entropy_risk",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_adic"),
    )

    if len(m) != EXPECTED_ROWS:
        raise RuntimeError(f"Four-way join rows={len(m)}, expected={EXPECTED_ROWS}")

    if "center_adic" not in m.columns:
        raise RuntimeError("Expected center_adic after ADIC merge.")

    if not np.array_equal(
        m["center"].astype(str).str.strip().to_numpy(),
        m["center_adic"].astype(str).str.strip().to_numpy(),
    ):
        raise RuntimeError("Frozen-vs-ADIC center mismatch.")
    m = m.drop(columns=["center_adic"])

    if m["sample_id"].astype(str).nunique() != EXPECTED_SAMPLES:
        raise RuntimeError("Four-way unique physical sample count mismatch.")

    per_sample_rows = m.groupby("sample_id").size()
    if not (per_sample_rows == EXPECTED_STATES).all():
        raise RuntimeError("Each physical image must have exactly 3 DeepLab rows.")

    if not np.isin(m["harm_label"].to_numpy(dtype=int), [0, 1]).all():
        raise RuntimeError("harm_label is not binary.")
    if len(np.unique(m["harm_label"].to_numpy(dtype=int))) != 2:
        raise RuntimeError("HARM outcome has only one class.")

    for col in METHODS.values():
        x = m[col].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite method score: {col}")

    return m


def metric_pair(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def per_seed_metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sub = df[df["training_seed"].astype(int) == seed]
        m = metric_pair(
            sub["harm_label"].to_numpy(dtype=int),
            sub[score_col].to_numpy(dtype=float),
        )
        rows.append({
            "training_seed": seed,
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
    seed = per_seed_metrics(df, score_col)
    return {
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
        "macro3_auroc": float(seed["auroc"].mean()),
        "macro3_auprc": float(seed["auprc"].mean()),
    }


def sample_ids_by_center(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sc = (
        df[["sample_id", "center"]]
        .drop_duplicates()
        .sort_values(["center", "sample_id"])
    )
    out = {}
    for center, g in sc.groupby("center", sort=True):
        out[str(center)] = g["sample_id"].astype(str).to_numpy()
    return out


def bootstrap_panel(
    df: pd.DataFrame,
    center_ids: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = {
        str(sid): g.copy()
        for sid, g in df.groupby("sample_id", sort=False)
    }

    parts = []
    draw_index = 0
    for center in sorted(center_ids):
        ids = center_ids[center]
        draw = rng.choice(ids, size=len(ids), replace=True)
        for sid in draw:
            g = groups[str(sid)].copy()
            g["_bootstrap_cluster"] = f"{draw_index:06d}::{sid}"
            parts.append(g)
            draw_index += 1

    return pd.concat(parts, ignore_index=True)


def bootstrap_deltas(
    df: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = sample_ids_by_center(df)

    rows = []
    it = range(reps)
    if tqdm is not None:
        it = tqdm(
            it,
            total=reps,
            desc="DeepLab3 four-way paired center-stratified bootstrap",
            unit="rep",
            dynamic_ncols=True,
        )

    for rep in it:
        b = bootstrap_panel(df, centers, rng)
        metrics = {
            method: summary_metrics(b, col)
            for method, col in METHODS.items()
        }

        row = {"rep": rep}
        for method, vals in metrics.items():
            prefix = method.lower().replace("-", "_")
            for metric, val in vals.items():
                row[f"{prefix}_{metric}"] = val

        safe = metrics["SafeTTA"]
        for comparator in ["SicTTA-CCD", "TEGDA-ADIC", "MC-dropout"]:
            comp = metrics[comparator]
            cprefix = comparator.lower().replace("-", "_")
            for metric in [
                "pooled_auroc", "pooled_auprc",
                "macro3_auroc", "macro3_auprc",
            ]:
                row[
                    f"delta_safetta_minus_{cprefix}_{metric}"
                ] = safe[metric] - comp[metric]

        rows.append(row)

    return pd.DataFrame(rows)


def ci95(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


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

    print("===== R17A POLYPGEN DEEPLAB3 FOUR-WAY POST-GT EVALUATION =====")
    print("Version:", VERSION)
    print("Dataset: PolypGen")
    print("Family: DeepLabV3-R50")
    print("States: 3")
    print("Rows:", EXPECTED_ROWS)
    print("Methods: SafeTTA / SicTTA-CCD / TEGDA-ADIC / MC-dropout")
    print("Target tuning/recalibration: NO")
    print("Score reversal after GT reveal: NO")
    print()

    print("[1/5] Validate/load frozen pre-GT score artifacts...")
    ccd, ccd_sha = validate_and_subset_ccd(root)
    adic, adic_sha = validate_adic(root)

    print("[2/5] Validate/load frozen post-GT SafeTTA/HARM panel...")
    frozen, frozen_sha = load_frozen_r10_panel(root)

    print("[3/5] Build exact 4596-row four-way matched panel...")
    panel = merge_fourway(frozen, ccd, adic)

    panel_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_MATCHED_PANEL.csv"
    panel.to_csv(panel_path, index=False)
    panel_sha = sha256_file(panel_path)

    print("[4/5] Compute point/per-seed metrics and paired bootstrap...")
    point_rows = []
    seed_rows = []

    for method, score_col in METHODS.items():
        sm = summary_metrics(panel, score_col)
        point_rows.append({
            "method": method,
            "rows": len(panel),
            "physical_samples": panel["sample_id"].nunique(),
            "states": EXPECTED_STATES,
            **sm,
        })
        p = per_seed_metrics(panel, score_col)
        p.insert(0, "method", method)
        seed_rows.append(p)

    point = pd.DataFrame(point_rows)
    per_seed = pd.concat(seed_rows, ignore_index=True)

    boot = bootstrap_deltas(
        panel,
        reps=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed),
    )

    print("[5/5] Freeze paired SafeTTA-vs-baseline delta CIs...")
    point_lookup = {
        r["method"]: r for _, r in point.iterrows()
    }

    delta_rows = []
    for comparator in ["SicTTA-CCD", "TEGDA-ADIC", "MC-dropout"]:
        cprefix = comparator.lower().replace("-", "_")
        for metric in [
            "pooled_auroc", "pooled_auprc",
            "macro3_auroc", "macro3_auprc",
        ]:
            col = f"delta_safetta_minus_{cprefix}_{metric}"
            vals = boot[col].to_numpy(dtype=float)
            lo, hi = ci95(vals)
            point_delta = (
                float(point_lookup["SafeTTA"][metric])
                - float(point_lookup[comparator][metric])
            )
            delta_rows.append({
                "comparator": comparator,
                "metric": metric,
                "point_delta_safetta_minus_comparator": point_delta,
                "ci95_low": lo,
                "ci95_high": hi,
                "safetta_significantly_better_95ci": bool(lo > 0),
                "comparator_significantly_better_95ci": bool(hi < 0),
                "bootstrap_reps": int(args.bootstrap_reps),
                "bootstrap_seed": int(args.bootstrap_seed),
                "cluster": "sample_id",
                "center_stratified": True,
            })

    delta = pd.DataFrame(delta_rows)

    point_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_POINT_METRICS.csv"
    seed_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_PER_SEED_METRICS.csv"
    boot_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_PAIRED_BOOTSTRAP.csv"
    delta_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_DELTA_CI95.csv"

    point.to_csv(point_path, index=False)
    per_seed.to_csv(seed_path, index=False)
    boot.to_csv(boot_path, index=False)
    delta.to_csv(delta_path, index=False)

    lock = {
        "status": "PASS",
        "decision": "PASS_R17A_POLYPGEN_DEEPLAB3_FOURWAY_POSTGT_EVALUATION",
        "version": VERSION,
        "dataset": "PolypGen",
        "scope": {
            "model_family": FAMILY,
            "training_seeds": list(SEEDS),
            "states": EXPECTED_STATES,
            "physical_samples": EXPECTED_SAMPLES,
            "rows": EXPECTED_ROWS,
            "methods": list(METHODS.keys()),
            "full_9state_primary_result_replaced": False,
        },
        "frozen_inputs": {
            "ccd_score_table": str(CCD_REL).replace("\\", "/"),
            "ccd_score_sha256": ccd_sha,
            "adic_mc_score_table": str(ADIC_REL).replace("\\", "/"),
            "adic_mc_score_sha256": adic_sha,
            "r10l3c_panel": str(
                R10L3C_DIR_REL / R10L3C_PANEL_NAME
            ).replace("\\", "/"),
            "r10l3c_panel_sha256": frozen_sha,
            "fourway_matched_panel": panel_path.name,
            "fourway_matched_panel_sha256": panel_sha,
        },
        "evaluation": {
            "harm_label": "frozen R10L3C harm_label",
            "fixed_directions": {
                "SafeTTA": "higher = higher HARM risk",
                "SicTTA-CCD": "higher = higher HARM risk",
                "TEGDA-ADIC": "adic_harm_risk=-adic_quality; higher = higher HARM risk",
                "MC-dropout": "higher predictive entropy = higher HARM risk",
            },
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
            seed_path.name: sha256_file(seed_path),
            boot_path.name: sha256_file(boot_path),
            delta_path.name: sha256_file(delta_path),
        },
        "interpretation": (
            "Published-baseline sensitivity panel on the exact common "
            "DeepLabV3-R50 3-state subset. Do not replace the manuscript's "
            "original 9-state PolypGen primary result."
        ),
    }

    lock_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_EVALUATION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A POLYPGEN DEEPLAB3 FOUR-WAY EVALUATION REPORT =====",
        f"Version: {VERSION}",
        "Dataset: PolypGen",
        "Family: DeepLabV3-R50",
        f"Rows: {EXPECTED_ROWS}",
        f"Physical samples: {EXPECTED_SAMPLES}",
        "States: 3/3",
        "",
        "POINT METRICS:",
    ]

    for _, r in point.iterrows():
        lines.append(
            f"{r['method']}: "
            f"pooled AUROC={r['pooled_auroc']:.9f} "
            f"AUPRC={r['pooled_auprc']:.9f} | "
            f"macro3 AUROC={r['macro3_auroc']:.9f} "
            f"AUPRC={r['macro3_auprc']:.9f}"
        )

    lines += [
        "",
        "PAIRED CLUSTERED BOOTSTRAP DELTAS (SafeTTA - comparator):",
    ]

    for r in delta.itertuples(index=False):
        lines.append(
            f"{r.comparator} / {r.metric}: "
            f"delta={r.point_delta_safetta_minus_comparator:+.9f} "
            f"95%CI=[{r.ci95_low:+.9f},{r.ci95_high:+.9f}] "
            f"SafeTTA_better={r.safetta_significantly_better_95ci}"
        )

    lines += [
        "",
        "INTERPRETATION:",
        "- All four methods use exactly the same 4596 model-case rows.",
        "- No target tuning/recalibration/score reversal was performed.",
        "- This is a published-baseline sensitivity panel.",
        "- It does NOT replace the original full 9-state PolypGen primary result.",
        "",
        "GATE=PASS_R17A_POLYPGEN_DEEPLAB3_FOURWAY_POSTGT_EVALUATION",
        "NEXT=REVIEW_FOURWAY_EFFECT_SIZES_AND_FREEZE_R17A_MANUSCRIPT_UPDATE_DECISION",
    ]

    report_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_EVALUATION_REPORT.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    for line in lines[-28:]:
        print(line)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
