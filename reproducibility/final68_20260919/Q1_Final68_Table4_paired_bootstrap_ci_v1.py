#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

VERSION = "2026-09-20-Final68-Table4-paired-bootstrap-v1"
EXPECTED = {
    "Geometry4": 0.5710304,
    "Transition64": 0.6295295,
    "Final68": 0.6886407,
}

CLUSTER_CANDIDATES = [
    "physical_image_id", "image_id", "sample_id", "frame_id", "case_id",
    "image_key", "sample_key", "frame_key", "case_key",
]
DELTA_CANDIDATES = ["delta_dice", "true_delta_dice", "delta", "dice_delta"]
LABEL_CANDIDATES = ["outcome", "outcome_label", "tristate", "label", "class_label"]
REP_CANDIDATES = ["representation", "variant", "method", "model"]
SCORE_GENERIC_CANDIDATES = ["risk_score", "score", "harm_risk", "risk", "utility"]


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum() or ch == "_")


def find_first(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cmap = {norm(c): c for c in cols}
    for cand in candidates:
        nc = norm(cand)
        if nc in cmap:
            return cmap[nc]
    for c in cols:
        lc = norm(c)
        for cand in candidates:
            if norm(cand) in lc:
                return c
    return None


def classify_score_cols(cols: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in cols:
        lc = norm(c)
        if any(k in lc for k in ["geometry4", "geometry_4", "geom4"]):
            if any(k in lc for k in ["score", "risk", "utility", "prob", "margin"]):
                out.setdefault("Geometry4", c)
        if any(k in lc for k in ["transition64", "transition_64", "semantictransition"]):
            if any(k in lc for k in ["score", "risk", "utility", "prob", "margin"]):
                out.setdefault("Transition64", c)
        if any(k in lc for k in ["final68", "full68", "geometry4_transition64"]):
            if any(k in lc for k in ["score", "risk", "utility", "prob", "margin"]):
                out.setdefault("Final68", c)
    return out


def derive_binary_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mask and y where y=1 is HARM, y=0 is BENEFIT."""
    delta_col = find_first(df.columns, DELTA_CANDIDATES)
    if delta_col:
        d = pd.to_numeric(df[delta_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(d) & ((d <= -0.02) | (d >= 0.02))
        y = np.full(len(df), -1, dtype=int)
        y[d <= -0.02] = 1
        y[d >= 0.02] = 0
        return mask, y

    label_col = find_first(df.columns, LABEL_CANDIDATES)
    if label_col:
        s = df[label_col].astype(str).str.upper().str.strip()
        mask = s.isin(["HARM", "BENEFIT", "BEN", "2", "0"]).to_numpy()
        y = np.full(len(df), -1, dtype=int)
        y[s.eq("HARM").to_numpy()] = 1
        y[(s.eq("BENEFIT") | s.eq("BEN")).to_numpy()] = 0
        # Numeric labels are ambiguous; only accept if text labels are absent.
        if not np.any(y >= 0):
            vals = pd.to_numeric(df[label_col], errors="coerce")
            u = set(vals.dropna().astype(int).unique().tolist())
            if {0, 2}.issubset(u):
                y[vals.eq(0).to_numpy()] = 1
                y[vals.eq(2).to_numpy()] = 0
                mask = (y >= 0)
        return mask, y

    raise ValueError("Could not identify delta_dice or HARM/BENEFIT label column.")


def choose_orientation(y: np.ndarray, x: np.ndarray, expected: float, name: str) -> Tuple[np.ndarray, float, int]:
    good = np.isfinite(x)
    if good.sum() < 2 or len(np.unique(y[good])) < 2:
        raise ValueError(f"{name}: insufficient finite decisive rows")
    a1 = roc_auc_score(y[good], x[good])
    a2 = roc_auc_score(y[good], -x[good])
    if abs(a1 - expected) <= abs(a2 - expected):
        chosen, sign = a1, +1
        xx = x
    else:
        chosen, sign = a2, -1
        xx = -x
    if abs(chosen - expected) > 0.03:
        raise ValueError(
            f"{name}: neither score orientation reproduces frozen AUROC anchor. "
            f"got {a1:.6f}/{a2:.6f}, expected {expected:.6f}"
        )
    return xx, chosen, sign


def load_wide(path: Path) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    cluster = find_first(df.columns, CLUSTER_CANDIDATES)
    scores = classify_score_cols(df.columns)
    if cluster and len(scores) == 3:
        return df, cluster, scores

    # Long format attempt.
    rep_col = find_first(df.columns, REP_CANDIDATES)
    score_col = find_first(df.columns, SCORE_GENERIC_CANDIDATES)
    if cluster and rep_col and score_col:
        base_cols = [cluster]
        for c in DELTA_CANDIDATES + LABEL_CANDIDATES:
            fc = find_first(df.columns, [c])
            if fc and fc not in base_cols:
                base_cols.append(fc)
        tmp = df[base_cols + [rep_col, score_col]].copy()
        tmp["__rep__"] = tmp[rep_col].astype(str).str.upper()
        def rep_name(v: str) -> Optional[str]:
            if "GEOMETRY4_TRANSITION64" in v or "FINAL68" in v or "FULL68" in v:
                return "Final68"
            if "TRANSITION64" in v or "SEMANTIC_TRANSITION" in v:
                return "Transition64"
            if "GEOMETRY4" in v:
                return "Geometry4"
            return None
        tmp["__name__"] = tmp["__rep__"].map(rep_name)
        tmp = tmp[tmp["__name__"].notna()].copy()
        # Need a row identity in addition to cluster to pivot safely.
        id_candidates = ["row_id", "model_case_id", "modelcase_id", "sample_index", "row_index", "index"]
        rid = find_first(df.columns, id_candidates)
        if not rid:
            raise ValueError("Long-format candidate found but no row identity column for pivot.")
        tmp[rid] = df.loc[tmp.index, rid]
        value = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp["__score__"] = value
        piv = tmp.pivot_table(index=[cluster, rid] + [c for c in base_cols if c != cluster], columns="__name__", values="__score__", aggfunc="first").reset_index()
        scores = {k: k for k in ["Geometry4", "Transition64", "Final68"] if k in piv.columns}
        if len(scores) == 3:
            return piv, cluster, scores

    raise ValueError("Could not identify cluster + Geometry4/Transition64/Final68 score columns.")


def discover(root: Path, max_files: int = 50000) -> List[Path]:
    candidates: List[Path] = []
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip obvious environments/caches.
        dirnames[:] = [d for d in dirnames if d.lower() not in {".git", "__pycache__", "site-packages", "node_modules"}]
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                files.append(Path(dirpath) / fn)
                if len(files) >= max_files:
                    break
        if len(files) >= max_files:
            break
    for p in tqdm(files, desc="Scanning CSV headers", unit="file"):
        try:
            hdr = pd.read_csv(p, nrows=0)
            cols = list(hdr.columns)
            if find_first(cols, CLUSTER_CANDIDATES) and len(classify_score_cols(cols)) == 3:
                candidates.append(p)
                continue
            # Long-format rough screen.
            if find_first(cols, CLUSTER_CANDIDATES) and find_first(cols, REP_CANDIDATES) and find_first(cols, SCORE_GENERIC_CANDIDATES):
                candidates.append(p)
        except Exception:
            pass
    return candidates


def bootstrap(df: pd.DataFrame, cluster_col: str, score_cols: Dict[str, str], reps: int, seed: int) -> pd.DataFrame:
    mask, yy = derive_binary_labels(df)
    d = df.loc[mask].copy().reset_index(drop=True)
    y = yy[mask]
    clusters = d[cluster_col].astype(str).to_numpy()

    oriented: Dict[str, np.ndarray] = {}
    points: Dict[str, float] = {}
    for name in ["Geometry4", "Transition64", "Final68"]:
        x = pd.to_numeric(d[score_cols[name]], errors="coerce").to_numpy(float)
        x, auc, sign = choose_orientation(y, x, EXPECTED[name], name)
        oriented[name] = x
        points[name] = auc
        print(f"{name}: point AUROC={auc:.9f}, orientation_sign={sign:+d}, expected={EXPECTED[name]:.9f}")

    unique_clusters = np.array(pd.unique(clusters))
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique_clusters}
    rng = np.random.default_rng(seed)
    diffs_g: List[float] = []
    diffs_t: List[float] = []

    for _ in tqdm(range(reps), desc="Clustered bootstrap", unit="rep"):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in sampled])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            a_f = roc_auc_score(yb, oriented["Final68"][idx])
            a_g = roc_auc_score(yb, oriented["Geometry4"][idx])
            a_t = roc_auc_score(yb, oriented["Transition64"][idx])
        except Exception:
            continue
        diffs_g.append(a_f - a_g)
        diffs_t.append(a_f - a_t)

    def row(name: str, raw: float, arr: Sequence[float]) -> dict:
        a = np.asarray(arr, float)
        return {
            "comparison": name,
            "raw_point_difference": raw,
            "bootstrap_reps_requested": reps,
            "bootstrap_reps_valid": int(len(a)),
            "bootstrap_mean_difference": float(np.mean(a)),
            "ci95_low": float(np.quantile(a, 0.025)),
            "ci95_high": float(np.quantile(a, 0.975)),
            "ci_excludes_zero": bool((np.quantile(a, 0.025) > 0) or (np.quantile(a, 0.975) < 0)),
        }

    out = pd.DataFrame([
        row("Final68 - Geometry4 H-v-B AUROC", points["Final68"] - points["Geometry4"], diffs_g),
        row("Final68 - Transition64 H-v-B AUROC", points["Final68"] - points["Transition64"], diffs_t),
    ])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired physical-image-clustered bootstrap for SafeTTA Table 4 H-v-B AUROC contrasts.")
    ap.add_argument("--input", type=Path, default=None, help="Frozen per-case component-score CSV. If omitted, use --root discovery.")
    ap.add_argument("--root", type=Path, default=Path(r"F:\MEDSEG_SAFETTA"), help="Root to scan when --input is omitted.")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", type=Path, default=Path("Q1_Final68_Table4_paired_bootstrap_ci_v1.csv"))
    args = ap.parse_args()

    print("=" * 110)
    print("SafeTTA Table 4 paired component bootstrap")
    print("Version:", VERSION)
    print("No model fitting. No threshold tuning. No score reversal based on target outcome beyond anchor-verified sign binding.")
    print("=" * 110)

    if args.input is None:
        print("Discovering candidate CSVs under:", args.root)
        cands = discover(args.root)
        if not cands:
            print("NO_CANDIDATE_SCORE_CSV_FOUND")
            sys.exit(2)
        print("Candidate files:")
        for i, p in enumerate(cands, 1):
            print(f"  [{i}] {p}")
        valid = []
        for p in cands:
            try:
                df, cluster, scores = load_wide(p)
                mask, y = derive_binary_labels(df)
                if mask.sum() >= 100 and len(np.unique(y[mask])) == 2:
                    valid.append((p, df, cluster, scores))
            except Exception:
                pass
        if len(valid) != 1:
            print(f"VALID_CANDIDATES={len(valid)}")
            print("Please rerun with --input <exact CSV> using one frozen per-case component-score table.")
            for p, _, cluster, scores in valid:
                print("VALID:", p, "cluster=", cluster, "scores=", scores)
            sys.exit(3)
        path, df, cluster, scores = valid[0]
    else:
        path = args.input
        df, cluster, scores = load_wide(path)

    print("INPUT:", path)
    print("ROWS:", len(df))
    print("CLUSTER_COL:", cluster)
    print("SCORE_COLS:", scores)
    out = bootstrap(df, cluster, scores, args.reps, args.seed)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print("OUTPUT:", args.output.resolve())
    print("GATE=PASS_FINAL68_TABLE4_PAIRED_COMPONENT_BOOTSTRAP")


if __name__ == "__main__":
    main()
