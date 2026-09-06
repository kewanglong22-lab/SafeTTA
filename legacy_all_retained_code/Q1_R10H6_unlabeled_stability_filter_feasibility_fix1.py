
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
N_SPLITS = 5
KS_THRESHOLD = 0.05

DEFAULT_SOURCE = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10A0_invariant_feature_table_v1/"
    "invariant_feature_table.csv"
)
DEFAULT_EXTERNAL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H2_NeoPolyp_external_labeled_panel_fix1_v1/"
    "neopolyp_external_evaluation_labeled_panel.csv"
)
DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H6_unlabeled_stability_filter_feasibility_fix1_v1"
)

RAW18 = [
    "source_entropy_mean","source_entropy_std","source_entropy_q10",
    "source_entropy_q50","source_entropy_q90","source_prob_mean",
    "source_prob_std","source_prob_q10","source_prob_q50","source_prob_q90",
    "source_confidence_mean","source_confidence_std","source_fg_fraction",
    "source_boundary_density","source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050","source_logit_abs_mean",
    "source_logit_abs_std",
]
RUR12 = [
    "rur_source_entropy_mean_z","rur_source_entropy_mean_mad",
    "rur_source_entropy_mean_tail","rur_source_prob_mean_z",
    "rur_source_prob_mean_mad","rur_source_prob_mean_tail",
    "rur_source_confidence_mean_z","rur_source_confidence_mean_mad",
    "rur_source_confidence_mean_tail","rur_source_logit_abs_mean_z",
    "rur_source_logit_abs_mean_mad","rur_source_logit_abs_mean_tail",
]
CANDIDATE30 = RAW18 + RUR12
STRUCTURE2 = ["source_fg_fraction", "source_boundary_density"]
SHAPE12 = [
    "source_entropy_std","source_entropy_q10","source_entropy_q50",
    "source_entropy_q90","source_prob_std","source_prob_q10",
    "source_prob_q50","source_prob_q90","source_confidence_std",
    "source_uncertain_fraction_040_060","source_high_entropy_fraction_050",
    "source_logit_abs_std",
]

def norm_sid(s):
    return s.astype(str).str.strip().str.replace("\\","/",regex=False).str.lower()

def model(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed
        )),
    ])

def mat(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols
    ])

def ks_stat(a, b):
    a = np.sort(np.asarray(a, float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, float)[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return 1.0
    v = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, v, side="right") / len(a)
    cb = np.searchsorted(b, v, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))

def metrics(y, p):
    return float(roc_auc_score(y,p)), float(average_precision_score(y,p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H6 UNLABELED STABILITY FILTER FEASIBILITY FIX1 =====")
    print("STATUS: FEASIBILITY / ROUTE SELECTION")
    print("GROUP KEY: sample_id")
    print("OUTER FOLDS:", N_SPLITS)
    print("SEEDS:", SEEDS)
    print("CANDIDATE UNIQUE FEATURES:", len(CANDIDATE30))
    print("DUPLICATED struct_* ALIASES INCLUDED: NO")
    print("PRIMARY FILTER: KS <=", KS_THRESHOLD)
    print("EXTERNAL LABELS USED FOR FEATURE SELECTION: NO")
    print("EXTERNAL LABELS USED FOR FITTING: NO")
    print("HELD-OUT EXTERNAL FEATURES USED FOR FEATURE SELECTION: NO")
    print("THRESHOLD TUNING: NONE\n")

    src = pd.read_csv(args.source, low_memory=False)
    ext = pd.read_csv(args.external, low_memory=False)
    src["_sid"] = norm_sid(src["sample_id"])
    ext["_sid"] = norm_sid(ext["sample_id"])

    if set(src["_sid"]) != set(ext["_sid"]):
        raise AssertionError("Case pools differ")
    if not (src["_sid"].value_counts() == 9).all():
        raise AssertionError("Source multiplicity != 9")
    if not (ext["_sid"].value_counts() == 9).all():
        raise AssertionError("External multiplicity != 9")

    for c in CANDIDATE30:
        if c not in src.columns or c not in ext.columns:
            raise AssertionError(f"Missing frozen candidate feature: {c}")

    ysrc = src["outcome"].astype(str).str.upper().eq("HARM").astype(int).to_numpy()
    yext = ext["harm_label"].astype(int).to_numpy()
    gsrc = src["_sid"].to_numpy()
    gext = ext["_sid"].to_numpy()

    methods = ["full_30","structure_2","shape12_posthoc_diagnostic","KS_stability_filter"]
    result_rows, select_rows = [], []

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed)
        print(f"\n######## SEED {seed} ########")
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros((len(src),1)), ysrc, gsrc))
        ps = {m: np.full(len(src), np.nan) for m in methods}
        pe = {m: np.full(len(ext), np.nan) for m in methods}

        for fold, (tr, te_s) in enumerate(splits):
            train_cases, test_cases = set(gsrc[tr]), set(gsrc[te_s])
            if train_cases & test_cases:
                raise AssertionError("Outer group leakage")
            tr_e = np.flatnonzero(np.isin(gext, list(train_cases)))
            te_e = np.flatnonzero(np.isin(gext, list(test_cases)))
            if set(gext[te_e]) != test_cases:
                raise AssertionError("External held-out case mismatch")

            selected = []
            for c in CANDIDATE30:
                ks = ks_stat(
                    pd.to_numeric(src.iloc[tr][c], errors="coerce").to_numpy(float),
                    pd.to_numeric(ext.iloc[tr_e][c], errors="coerce").to_numpy(float),
                )
                keep = ks <= KS_THRESHOLD
                selected.append(c) if keep else None
                select_rows.append({
                    "seed":seed,"fold":fold,"feature":c,"KS":ks,"selected":keep
                })

            print(
                f"fold={fold} train_cases={len(train_cases)} test_cases={len(test_cases)} "
                f"selected={len(selected)}/30"
            )
            print(" selected:", selected)
            if not selected:
                raise AssertionError("KS filter selected no features")

            panels = {
                "full_30": CANDIDATE30,
                "structure_2": STRUCTURE2,
                "shape12_posthoc_diagnostic": SHAPE12,
                "KS_stability_filter": selected,
            }

            for name, cols in panels.items():
                m = model(seed + fold)
                m.fit(mat(src.iloc[tr], cols), ysrc[tr])
                ps[name][te_s] = m.predict_proba(mat(src.iloc[te_s], cols))[:,1]
                pe[name][te_e] = m.predict_proba(mat(ext.iloc[te_e], cols))[:,1]

        for name in methods:
            if np.isnan(ps[name]).any() or np.isnan(pe[name]).any():
                raise AssertionError(f"Incomplete predictions: {name}")
            sauc,spr = metrics(ysrc, ps[name])
            eauc,epr = metrics(yext, pe[name])
            print(
                f"{name:28s} SRC_AUROC={sauc:.6f} SRC_AUPRC={spr:.6f} "
                f"EXT_AUROC={eauc:.6f} EXT_AUPRC={epr:.6f}"
            )
            result_rows.append({
                "seed":seed,"method":name,
                "source_AUROC":sauc,"source_AUPRC":spr,
                "external_AUROC":eauc,"external_AUPRC":epr,
            })

    res = pd.DataFrame(result_rows)
    sel = pd.DataFrame(select_rows)

    summary = res.groupby("method", sort=False).agg(
        seeds=("seed","count"),
        source_AUROC_mean=("source_AUROC","mean"),
        source_AUROC_std=("source_AUROC","std"),
        source_AUPRC_mean=("source_AUPRC","mean"),
        source_AUPRC_std=("source_AUPRC","std"),
        external_AUROC_mean=("external_AUROC","mean"),
        external_AUROC_std=("external_AUROC","std"),
        external_AUPRC_mean=("external_AUPRC","mean"),
        external_AUPRC_std=("external_AUPRC","std"),
    ).reset_index()

    sel_summary = sel.groupby("feature", sort=False).agg(
        evaluations=("selected","count"),
        selected_count=("selected","sum"),
        selected_fraction=("selected","mean"),
        KS_mean=("KS","mean"),
        KS_max=("KS","max"),
    ).reset_index().sort_values(
        ["selected_fraction","KS_mean"], ascending=[False,True]
    )

    print("\n===== FINAL SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== FEATURE STABILITY =====")
    print(sel_summary.to_string(index=False))

    ks = summary[summary.method=="KS_stability_filter"].iloc[0]
    full = summary[summary.method=="full_30"].iloc[0]
    st = summary[summary.method=="structure_2"].iloc[0]

    print("\n===== ROUTE DECISION =====")
    print("KS-filter external AUROC:", float(ks.external_AUROC_mean))
    print("Gain vs full30:", float(ks.external_AUROC_mean-full.external_AUROC_mean))
    print("Gain vs structure2:", float(ks.external_AUROC_mean-st.external_AUROC_mean))

    if ks.external_AUROC_mean >= 0.80 and ks.external_AUROC_mean >= st.external_AUROC_mean + 0.02:
        decision = "STABILITY_FILTER_ROUTE_PROMISING"
    elif ks.external_AUROC_mean >= st.external_AUROC_mean:
        decision = "STABILITY_FILTER_ROUTE_PARTIALLY_SUPPORTED"
    else:
        decision = "STABILITY_FILTER_ROUTE_NOT_SUPPORTED"
    print("DECISION:", decision)

    res.to_csv(out/"R10H6_per_seed_metrics.csv", index=False)
    summary.to_csv(out/"R10H6_summary.csv", index=False)
    sel.to_csv(out/"R10H6_feature_stability_selection.csv", index=False)
    sel_summary.to_csv(out/"R10H6_feature_stability_selection_summary.csv", index=False)
    print("\nOutput:", out)
    print("PASS")

if __name__ == "__main__":
    main()
