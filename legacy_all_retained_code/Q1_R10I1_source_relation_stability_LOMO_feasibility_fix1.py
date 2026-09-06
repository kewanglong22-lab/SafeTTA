
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10I1_source_relation_stability_LOMO_feasibility_fix1.py

Purpose
-------
R10I0 showed that hard marginal-distribution stability screening (KS<=0.05)
does NOT generalize across model families:
- it degraded DeepLabV3-R50 transfer,
- underperformed structure-only for PraNet,
- and selected zero features for SegFormer-B0.

This experiment tests the next mechanistic hypothesis:

    Safety transfer depends more on STABILITY OF FEATURE-LABEL RELATIONSHIPS
    across source environments than on marginal feature-distribution stability.

Protocol
--------
Leave-One-Model-Family-Out (LOMO):
- target family = held-out model family
- other two families = labeled source environments
- 5-fold StratifiedGroupKFold by sample_id
- target labels are evaluation-only

Feature selection for RELATION_STABLE_FILTER:
For each outer fold and each candidate feature:
1) compute univariate AUROC separately in the two SOURCE families,
   using source-training cases only;
2) retain the feature iff BOTH source-family AUROCs show the same direction
   and each has at least MIN_SIGNAL=0.05 distance from random:
       both >= 0.55  OR  both <= 0.45
3) target-family features and labels are NOT used for selection.

This is a post-R10I0 feasibility experiment, not confirmatory evidence.

Methods
-------
- full_30
- structure_2
- relation_stable_filter

Candidate panel:
- 18 raw descriptors
- 12 RUR descriptors
- duplicated struct_* aliases excluded

Additional diagnostic:
Target-train marginal KS is printed for selected features, but is NOT used for
selection or fitting. This tests whether conditional/relation stability can
survive even when marginal distributions shift.
"""

import argparse
import random
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
MIN_SIGNAL = 0.05

DEFAULT_PANEL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10A0_invariant_feature_table_v1/"
    "invariant_feature_table.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10I1_source_relation_stability_LOMO_feasibility_fix1_v1"
)

RAW18 = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_entropy_q10",
    "source_entropy_q50",
    "source_entropy_q90",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_confidence_mean",
    "source_confidence_std",
    "source_fg_fraction",
    "source_boundary_density",
    "source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050",
    "source_logit_abs_mean",
    "source_logit_abs_std",
]

RUR12 = [
    "rur_source_entropy_mean_z",
    "rur_source_entropy_mean_mad",
    "rur_source_entropy_mean_tail",
    "rur_source_prob_mean_z",
    "rur_source_prob_mean_mad",
    "rur_source_prob_mean_tail",
    "rur_source_confidence_mean_z",
    "rur_source_confidence_mean_mad",
    "rur_source_confidence_mean_tail",
    "rur_source_logit_abs_mean_z",
    "rur_source_logit_abs_mean_mad",
    "rur_source_logit_abs_mean_tail",
]

CANDIDATE30 = RAW18 + RUR12

STRUCTURE2 = [
    "source_fg_fraction",
    "source_boundary_density",
]

FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def build_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def matrix(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in cols
    ])


def safe_univariate_auc(y, x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)

    if finite.sum() == 0:
        return np.nan

    med = np.nanmedian(x[finite])
    x = np.where(np.isfinite(x), x, med)

    if len(np.unique(y)) < 2:
        return np.nan

    return float(roc_auc_score(y, x))


def ks_statistic(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])

    if len(a) == 0 or len(b) == 0:
        return np.nan

    values = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, values, side="right") / len(a)
    cb = np.searchsorted(b, values, side="right") / len(b)

    return float(np.max(np.abs(ca - cb)))


def same_relation_direction(auc1, auc2):
    if not np.isfinite(auc1) or not np.isfinite(auc2):
        return False

    high = 0.5 + MIN_SIGNAL
    low = 0.5 - MIN_SIGNAL

    return (auc1 >= high and auc2 >= high) or (auc1 <= low and auc2 <= low)


def eval_metrics(y, p):
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10I1 SOURCE-RELATION-STABILITY LOMO FEASIBILITY FIX1 =====")
    print("STATUS: POST-R10I0 FEASIBILITY / ROUTE SELECTION")
    print("DOMAIN AXIS: model_family")
    print("GROUP KEY: sample_id")
    print("SOURCE ENVIRONMENTS PER TARGET: 2 model families")
    print("RELATION SIGNAL THRESHOLD:", MIN_SIGNAL)
    print("SELECTION RULE: both source-family AUROC >=0.55 OR both <=0.45")
    print("TARGET LABELS USED FOR FOLD CONSTRUCTION: NO")
    print("TARGET LABELS USED FOR FEATURE SELECTION: NO")
    print("TARGET FEATURES USED FOR FEATURE SELECTION: NO")
    print("TARGET LABELS USED FOR MODEL FITTING: NO")
    print("TARGET LABELS USED FOR EVALUATION ONLY: YES")
    print("TARGET MARGINAL KS: DIAGNOSTIC ONLY, NOT USED FOR SELECTION\n")

    panel_path = Path(args.panel)
    print("panel exists:", panel_path.exists(), "path:", panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    df = pd.read_csv(panel_path, low_memory=False)

    required = ["sample_id", "model_family", "outcome"] + CANDIDATE30
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

    df["_sid"] = norm_sid(df["sample_id"])
    df["_harm"] = (
        df["outcome"]
        .astype(str)
        .str.upper()
        .eq("HARM")
        .astype(int)
    )

    detected = set(df["model_family"].astype(str).unique())
    if detected != set(FAMILIES):
        raise AssertionError(
            f"Expected model families {FAMILIES}, got {sorted(detected)}"
        )

    per_case_family = (
        df.groupby(["_sid", "model_family"])
        .size()
        .reset_index(name="n")
    )
    if not (per_case_family["n"] == 3).all():
        raise AssertionError("Expected exactly 3 states/case/family.")

    print("Rows:", len(df))
    print("Cases:", df["_sid"].nunique())
    print("Families:", FAMILIES)

    result_rows = []
    selection_rows = []
    fold_rows = []

    for target_family in FAMILIES:
        source_families = [f for f in FAMILIES if f != target_family]

        source = df[df["model_family"].isin(source_families)].copy()
        target = df[df["model_family"] == target_family].copy()

        if set(source["_sid"]) != set(target["_sid"]):
            raise AssertionError("Source/target case pools differ.")

        y_source = source["_harm"].to_numpy(int)
        y_target = target["_harm"].to_numpy(int)
        g_source = source["_sid"].to_numpy()
        g_target = target["_sid"].to_numpy()

        print("\n\n============================================================")
        print("TARGET FAMILY:", target_family)
        print("SOURCE FAMILIES:", source_families)
        print("Source rows:", len(source))
        print("Target rows:", len(target))
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            print(f"\n######## TARGET={target_family} SEED={seed} ########")

            cv = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed,
            )

            splits = list(cv.split(
                np.zeros((len(source), 1)),
                y_source,
                g_source,
            ))

            methods = [
                "full_30",
                "structure_2",
                "relation_stable_filter",
            ]

            p_source = {
                m: np.full(len(source), np.nan, dtype=float)
                for m in methods
            }
            p_target = {
                m: np.full(len(target), np.nan, dtype=float)
                for m in methods
            }

            for fold, (tr_s, te_s) in enumerate(splits):
                train_cases = set(g_source[tr_s])
                test_cases = set(g_source[te_s])

                if train_cases & test_cases:
                    raise AssertionError("Outer group leakage.")

                te_t = np.flatnonzero(np.isin(g_target, list(test_cases)))
                tr_t = np.flatnonzero(np.isin(g_target, list(train_cases)))

                if set(g_target[te_t]) != test_cases:
                    raise AssertionError("Target-test case mismatch.")
                if set(g_target[tr_t]) != train_cases:
                    raise AssertionError("Target-train case mismatch.")

                source_train = source.iloc[tr_s]

                selected = []

                for c in CANDIDATE30:
                    fam_aucs = {}

                    for sf in source_families:
                        g = source_train[
                            source_train["model_family"] == sf
                        ]

                        y = g["_harm"].to_numpy(int)
                        x = pd.to_numeric(
                            g[c],
                            errors="coerce",
                        ).to_numpy(float)

                        fam_aucs[sf] = safe_univariate_auc(y, x)

                    auc1 = fam_aucs[source_families[0]]
                    auc2 = fam_aucs[source_families[1]]
                    keep = same_relation_direction(auc1, auc2)

                    # Diagnostic only: marginal source-vs-target KS.
                    src_vals = pd.to_numeric(
                        source_train[c],
                        errors="coerce",
                    ).to_numpy(float)
                    tgt_vals = pd.to_numeric(
                        target.iloc[tr_t][c],
                        errors="coerce",
                    ).to_numpy(float)
                    target_ks = ks_statistic(src_vals, tgt_vals)

                    if keep:
                        selected.append(c)

                    selection_rows.append({
                        "target_family": target_family,
                        "source_family_1": source_families[0],
                        "source_family_2": source_families[1],
                        "seed": seed,
                        "fold": fold,
                        "feature": c,
                        "source_family_1_AUROC": auc1,
                        "source_family_2_AUROC": auc2,
                        "same_relation_direction": bool(keep),
                        "selected": bool(keep),
                        "target_train_marginal_KS_diagnostic": target_ks,
                        "target_features_used_for_selection": "NO",
                        "target_labels_used_for_selection": "NO",
                    })

                print(
                    f"fold={fold} "
                    f"train_cases={len(train_cases)} "
                    f"test_cases={len(test_cases)} "
                    f"selected={len(selected)}/30"
                )
                print(" selected:", selected)

                if len(selected) == 0:
                    raise AssertionError(
                        f"{target_family} seed={seed} fold={fold}: "
                        "relation-stable filter selected zero features."
                    )

                panels = {
                    "full_30": CANDIDATE30,
                    "structure_2": STRUCTURE2,
                    "relation_stable_filter": selected,
                }

                for method, cols in panels.items():
                    clf = build_model(seed + 1000 * FAMILIES.index(target_family) + fold)
                    clf.fit(
                        matrix(source.iloc[tr_s], cols),
                        y_source[tr_s],
                    )

                    p_source[method][te_s] = clf.predict_proba(
                        matrix(source.iloc[te_s], cols)
                    )[:, 1]

                    p_target[method][te_t] = clf.predict_proba(
                        matrix(target.iloc[te_t], cols)
                    )[:, 1]

                fold_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "fold": fold,
                    "train_cases": len(train_cases),
                    "test_cases": len(test_cases),
                    "selected_feature_count": len(selected),
                    "selected_features": ";".join(selected),
                })

            for method in methods:
                if np.isnan(p_source[method]).any():
                    raise AssertionError(
                        f"Incomplete source predictions: "
                        f"{target_family}/{seed}/{method}"
                    )
                if np.isnan(p_target[method]).any():
                    raise AssertionError(
                        f"Incomplete target predictions: "
                        f"{target_family}/{seed}/{method}"
                    )

                s_auc, s_pr = eval_metrics(
                    y_source,
                    p_source[method],
                )
                t_auc, t_pr = eval_metrics(
                    y_target,
                    p_target[method],
                )

                print(
                    f"{method:24s} "
                    f"SOURCE_AUROC={s_auc:.6f} "
                    f"SOURCE_AUPRC={s_pr:.6f} "
                    f"TARGET_AUROC={t_auc:.6f} "
                    f"TARGET_AUPRC={t_pr:.6f}"
                )

                result_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "method": method,
                    "source_AUROC": s_auc,
                    "source_AUPRC": s_pr,
                    "target_AUROC": t_auc,
                    "target_AUPRC": t_pr,
                    "target_labels_used_for_split": "NO",
                    "target_features_used_for_selection": "NO",
                    "target_labels_used_for_selection": "NO",
                    "target_labels_used_for_fit": "NO",
                })

    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    folds = pd.DataFrame(fold_rows)

    by_family = (
        results.groupby(["target_family", "method"], sort=False)
        .agg(
            seeds=("seed", "count"),
            source_AUROC_mean=("source_AUROC", "mean"),
            source_AUROC_std=("source_AUROC", "std"),
            target_AUROC_mean=("target_AUROC", "mean"),
            target_AUROC_std=("target_AUROC", "std"),
            target_AUPRC_mean=("target_AUPRC", "mean"),
            target_AUPRC_std=("target_AUPRC", "std"),
        )
        .reset_index()
    )

    macro_seed = (
        results.groupby(["method", "seed"], sort=False)
        .agg(
            macro_target_AUROC=("target_AUROC", "mean"),
            macro_target_AUPRC=("target_AUPRC", "mean"),
        )
        .reset_index()
    )

    macro = (
        macro_seed.groupby("method", sort=False)
        .agg(
            seeds=("seed", "count"),
            macro_target_AUROC_mean=("macro_target_AUROC", "mean"),
            macro_target_AUROC_std=("macro_target_AUROC", "std"),
            macro_target_AUPRC_mean=("macro_target_AUPRC", "mean"),
            macro_target_AUPRC_std=("macro_target_AUPRC", "std"),
        )
        .reset_index()
    )

    selection_summary = (
        selections.groupby(["target_family", "feature"], sort=False)
        .agg(
            evaluations=("selected", "count"),
            selected_count=("selected", "sum"),
            selected_fraction=("selected", "mean"),
            source_family_1_AUROC_mean=("source_family_1_AUROC", "mean"),
            source_family_2_AUROC_mean=("source_family_2_AUROC", "mean"),
            target_KS_diagnostic_mean=(
                "target_train_marginal_KS_diagnostic",
                "mean",
            ),
            target_KS_diagnostic_max=(
                "target_train_marginal_KS_diagnostic",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            ["target_family", "selected_fraction"],
            ascending=[True, False],
        )
    )

    print("\n\n===== R10I1 BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10I1 MACRO SUMMARY =====")
    print(macro.to_string(index=False))

    rel = macro[
        macro["method"] == "relation_stable_filter"
    ].iloc[0]
    full = macro[
        macro["method"] == "full_30"
    ].iloc[0]
    structure = macro[
        macro["method"] == "structure_2"
    ].iloc[0]

    rel_family = by_family[
        by_family["method"] == "relation_stable_filter"
    ]
    worst = float(rel_family["target_AUROC_mean"].min())

    print("\n===== ROUTE DECISION =====")
    print(
        "Relation-stable macro target AUROC:",
        float(rel["macro_target_AUROC_mean"]),
    )
    print(
        "Gain vs full30:",
        float(
            rel["macro_target_AUROC_mean"]
            - full["macro_target_AUROC_mean"]
        ),
    )
    print(
        "Gain vs structure2:",
        float(
            rel["macro_target_AUROC_mean"]
            - structure["macro_target_AUROC_mean"]
        ),
    )
    print("Worst held-out-family AUROC:", worst)

    if (
        rel["macro_target_AUROC_mean"] >= 0.78
        and worst >= 0.72
        and rel["macro_target_AUROC_mean"]
            >= full["macro_target_AUROC_mean"] + 0.02
    ):
        decision = "RELATION_STABILITY_ROUTE_PROMISING"
    elif (
        rel["macro_target_AUROC_mean"]
        >= full["macro_target_AUROC_mean"]
    ):
        decision = "RELATION_STABILITY_ROUTE_PARTIALLY_SUPPORTED"
    else:
        decision = "RELATION_STABILITY_ROUTE_NOT_SUPPORTED"

    print("DECISION:", decision)

    results.to_csv(
        out / "R10I1_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10I1_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10I1_macro_summary.csv",
        index=False,
    )
    selections.to_csv(
        out / "R10I1_relation_selection.csv",
        index=False,
    )
    selection_summary.to_csv(
        out / "R10I1_relation_selection_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10I1_fold_audit.csv",
        index=False,
    )

    print("\nOutputs:")
    print(out / "R10I1_per_seed_family_metrics.csv")
    print(out / "R10I1_by_family_summary.csv")
    print(out / "R10I1_macro_summary.csv")
    print(out / "R10I1_relation_selection_summary.csv")
    print(out / "R10I1_fold_audit.csv")
    print("PASS")


if __name__ == "__main__":
    main()
