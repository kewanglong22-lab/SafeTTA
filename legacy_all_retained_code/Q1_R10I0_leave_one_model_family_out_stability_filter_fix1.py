
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10I0_leave_one_model_family_out_stability_filter_fix1.py

Purpose
-------
Test whether the unlabeled KS stability filter found in R10H6 generalizes under
a GENUINE model-family shift, rather than the paired-representation identity
setting in which the selected H6 features were numerically preserved.

Protocol
--------
For each held-out target model family:
    DeepLabV3-R50 / PraNet / SegFormer-B0

1. The other two model families are labeled SOURCE domains.
2. The held-out family is the unlabeled TARGET domain for feature selection.
3. Cases are split with StratifiedGroupKFold using SOURCE labels only.
4. In every outer fold:
   - source-train cases: labeled, used for model fitting;
   - target-train cases: FEATURES ONLY, used for KS stability screening;
   - target-test cases: completely held out from feature selection and fitting.
5. No target-family labels are used for:
   - fold construction,
   - feature selection,
   - model fitting,
   - hyperparameter tuning.
6. Target labels are used only after prediction for AUROC/AUPRC evaluation.

Methods
-------
- full_30
- structure_2
- KS_stability_filter

Candidate panel:
- 18 raw descriptors
- 12 RUR descriptors
- duplicated struct_* aliases excluded

Primary decision is based on macro-average held-out-family AUROC.

This is a feasibility experiment for cross-model-family safety generalization,
not an independent external-dataset validation.
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
KS_THRESHOLD = 0.05

DEFAULT_PANEL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10A0_invariant_feature_table_v1/"
    "invariant_feature_table.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10I0_leave_one_model_family_out_stability_filter_fix1_v1"
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

EXPECTED_FAMILIES = [
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


def numeric_matrix(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in cols
    ])


def ks_statistic(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])

    if len(a) == 0 or len(b) == 0:
        return 1.0

    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)

    return float(np.max(np.abs(cdf_a - cdf_b)))


def safe_metrics(y, p):
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10I0 LEAVE-ONE-MODEL-FAMILY-OUT STABILITY FILTER FIX1 =====")
    print("STATUS: FEASIBILITY / ROUTE SELECTION")
    print("DOMAIN AXIS: model_family")
    print("TARGET FAMILY LABELS USED FOR FOLD CONSTRUCTION: NO")
    print("TARGET FAMILY LABELS USED FOR FEATURE SELECTION: NO")
    print("TARGET FAMILY LABELS USED FOR MODEL FITTING: NO")
    print("TARGET FAMILY HELD-OUT FEATURES USED FOR SELECTION: NO")
    print("TARGET FAMILY LABELS USED FOR EVALUATION ONLY: YES")
    print("GROUP KEY: sample_id")
    print("KS THRESHOLD:", KS_THRESHOLD)
    print("CANDIDATE UNIQUE FEATURES:", len(CANDIDATE30))
    print("DUPLICATED struct_* ALIASES INCLUDED: NO\n")

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

    families = list(df["model_family"].dropna().astype(str).unique())
    print("Detected families:", families)
    print("Rows:", len(df))
    print("Unique cases:", df["_sid"].nunique())
    print("Rows per family:")
    print(df["model_family"].value_counts().to_string())

    if set(families) != set(EXPECTED_FAMILIES):
        raise AssertionError(
            f"Expected families {EXPECTED_FAMILIES}, got {families}"
        )

    # Audit every case contributes exactly 3 states per family.
    per_case_family = (
        df.groupby(["_sid", "model_family"])
        .size()
        .reset_index(name="n")
    )
    if not (per_case_family["n"] == 3).all():
        bad = per_case_family[per_case_family["n"] != 3].head(20)
        raise AssertionError(
            "Expected exactly 3 states per case per model family.\n"
            + bad.to_string(index=False)
        )

    result_rows = []
    selection_rows = []
    fold_rows = []

    for target_family in EXPECTED_FAMILIES:
        source_families = [f for f in EXPECTED_FAMILIES if f != target_family]

        source = df[df["model_family"].isin(source_families)].copy()
        target = df[df["model_family"] == target_family].copy()

        if set(source["_sid"]) != set(target["_sid"]):
            raise AssertionError(
                f"Case pools differ for target family {target_family}"
            )

        y_source = source["_harm"].to_numpy(int)
        y_target = target["_harm"].to_numpy(int)
        g_source = source["_sid"].to_numpy()
        g_target = target["_sid"].to_numpy()

        print("\n\n============================================================")
        print("TARGET FAMILY:", target_family)
        print("SOURCE FAMILIES:", source_families)
        print("Source rows:", len(source))
        print("Target rows:", len(target))
        print("Source HARM:", int(y_source.sum()))
        print("Target HARM:", int(y_target.sum()))
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            print(f"\n######## TARGET={target_family} SEED={seed} ########")

            # IMPORTANT: split construction uses SOURCE labels only.
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
                "KS_stability_filter",
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
                    raise AssertionError("Outer source case leakage.")

                # Target TRAIN FEATURES may be used unlabeled for KS selection.
                tr_t = np.flatnonzero(np.isin(g_target, list(train_cases)))
                te_t = np.flatnonzero(np.isin(g_target, list(test_cases)))

                if set(g_target[tr_t]) != train_cases:
                    raise AssertionError("Target-train case set mismatch.")
                if set(g_target[te_t]) != test_cases:
                    raise AssertionError("Target-test case set mismatch.")
                if set(g_target[te_t]) & train_cases:
                    raise AssertionError("Target held-out case leakage.")

                selected = []

                for c in CANDIDATE30:
                    a = pd.to_numeric(
                        source.iloc[tr_s][c],
                        errors="coerce",
                    ).to_numpy(float)
                    b = pd.to_numeric(
                        target.iloc[tr_t][c],
                        errors="coerce",
                    ).to_numpy(float)

                    ks = ks_statistic(a, b)
                    keep = ks <= KS_THRESHOLD

                    if keep:
                        selected.append(c)

                    selection_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "fold": fold,
                        "feature": c,
                        "KS_sourceTrain_vs_targetTrain": ks,
                        "selected": bool(keep),
                        "target_labels_used_for_selection": "NO",
                        "heldout_target_features_used_for_selection": "NO",
                    })

                if len(selected) == 0:
                    raise AssertionError(
                        f"{target_family} seed={seed} fold={fold}: "
                        "KS filter selected zero features."
                    )

                print(
                    f"fold={fold} "
                    f"train_cases={len(train_cases)} "
                    f"test_cases={len(test_cases)} "
                    f"source_train_rows={len(tr_s)} "
                    f"target_train_unlabeled_rows={len(tr_t)} "
                    f"target_test_rows={len(te_t)} "
                    f"selected={len(selected)}/30"
                )
                print(" selected:", selected)

                panels = {
                    "full_30": CANDIDATE30,
                    "structure_2": STRUCTURE2,
                    "KS_stability_filter": selected,
                }

                for method, cols in panels.items():
                    clf = build_model(seed + 1000 * EXPECTED_FAMILIES.index(target_family) + fold)

                    Xtr = numeric_matrix(source.iloc[tr_s], cols)
                    Xte_source = numeric_matrix(source.iloc[te_s], cols)
                    Xte_target = numeric_matrix(target.iloc[te_t], cols)

                    clf.fit(Xtr, y_source[tr_s])

                    p_source[method][te_s] = clf.predict_proba(Xte_source)[:, 1]
                    p_target[method][te_t] = clf.predict_proba(Xte_target)[:, 1]

                fold_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "fold": fold,
                    "train_cases": len(train_cases),
                    "test_cases": len(test_cases),
                    "source_train_rows": len(tr_s),
                    "source_test_rows": len(te_s),
                    "target_train_unlabeled_rows": len(tr_t),
                    "target_test_rows": len(te_t),
                    "selected_feature_count": len(selected),
                    "selected_features": ";".join(selected),
                })

            for method in methods:
                if np.isnan(p_source[method]).any():
                    raise AssertionError(
                        f"Incomplete source predictions: {target_family}/{seed}/{method}"
                    )
                if np.isnan(p_target[method]).any():
                    raise AssertionError(
                        f"Incomplete target predictions: {target_family}/{seed}/{method}"
                    )

                source_auc, source_pr = safe_metrics(
                    y_source, p_source[method]
                )
                target_auc, target_pr = safe_metrics(
                    y_target, p_target[method]
                )

                print(
                    f"{method:22s} "
                    f"SOURCE_AUROC={source_auc:.6f} "
                    f"SOURCE_AUPRC={source_pr:.6f} "
                    f"TARGET_AUROC={target_auc:.6f} "
                    f"TARGET_AUPRC={target_pr:.6f}"
                )

                result_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "method": method,
                    "source_group_oof_AUROC": source_auc,
                    "source_group_oof_AUPRC": source_pr,
                    "target_family_AUROC": target_auc,
                    "target_family_AUPRC": target_pr,
                    "target_labels_used_for_split": "NO",
                    "target_labels_used_for_selection": "NO",
                    "target_labels_used_for_fit": "NO",
                    "target_labels_used_for_evaluation_only": "YES",
                })

    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    folds = pd.DataFrame(fold_rows)

    by_family = (
        results.groupby(["target_family", "method"], sort=False)
        .agg(
            seeds=("seed", "count"),
            source_AUROC_mean=("source_group_oof_AUROC", "mean"),
            source_AUROC_std=("source_group_oof_AUROC", "std"),
            target_AUROC_mean=("target_family_AUROC", "mean"),
            target_AUROC_std=("target_family_AUROC", "std"),
            target_AUPRC_mean=("target_family_AUPRC", "mean"),
            target_AUPRC_std=("target_family_AUPRC", "std"),
        )
        .reset_index()
    )

    macro = (
        results.groupby(["method", "seed"], sort=False)
        .agg(
            macro_target_AUROC=("target_family_AUROC", "mean"),
            macro_target_AUPRC=("target_family_AUPRC", "mean"),
        )
        .reset_index()
        .groupby("method", sort=False)
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
            KS_mean=("KS_sourceTrain_vs_targetTrain", "mean"),
            KS_max=("KS_sourceTrain_vs_targetTrain", "max"),
        )
        .reset_index()
        .sort_values(
            ["target_family", "selected_fraction", "KS_mean"],
            ascending=[True, False, True],
        )
    )

    print("\n\n===== R10I0 BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10I0 MACRO HELD-OUT-FAMILY SUMMARY =====")
    print(macro.to_string(index=False))

    ks = macro[macro["method"] == "KS_stability_filter"].iloc[0]
    full = macro[macro["method"] == "full_30"].iloc[0]
    structure = macro[macro["method"] == "structure_2"].iloc[0]

    print("\n===== ROUTE DECISION =====")
    print("KS-filter macro target AUROC:", float(ks["macro_target_AUROC_mean"]))
    print(
        "Gain vs full30:",
        float(
            ks["macro_target_AUROC_mean"]
            - full["macro_target_AUROC_mean"]
        ),
    )
    print(
        "Gain vs structure2:",
        float(
            ks["macro_target_AUROC_mean"]
            - structure["macro_target_AUROC_mean"]
        ),
    )

    family_ks = by_family[
        by_family["method"] == "KS_stability_filter"
    ]
    worst_family_auc = float(family_ks["target_AUROC_mean"].min())
    print("Worst held-out-family KS AUROC:", worst_family_auc)

    if (
        ks["macro_target_AUROC_mean"] >= 0.75
        and worst_family_auc >= 0.70
        and ks["macro_target_AUROC_mean"]
            >= full["macro_target_AUROC_mean"] + 0.02
    ):
        decision = "CROSS_MODEL_STABILITY_FILTER_SUPPORTED"
    elif (
        ks["macro_target_AUROC_mean"]
        >= full["macro_target_AUROC_mean"]
    ):
        decision = "CROSS_MODEL_STABILITY_FILTER_PARTIALLY_SUPPORTED"
    else:
        decision = "CROSS_MODEL_STABILITY_FILTER_NOT_SUPPORTED"

    print("DECISION:", decision)

    results.to_csv(
        out / "R10I0_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10I0_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10I0_macro_summary.csv",
        index=False,
    )
    selections.to_csv(
        out / "R10I0_feature_selection.csv",
        index=False,
    )
    selection_summary.to_csv(
        out / "R10I0_feature_selection_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10I0_fold_audit.csv",
        index=False,
    )

    print("\nOutputs:")
    print(out / "R10I0_per_seed_family_metrics.csv")
    print(out / "R10I0_by_family_summary.csv")
    print(out / "R10I0_macro_summary.csv")
    print(out / "R10I0_feature_selection_summary.csv")
    print(out / "R10I0_fold_audit.csv")
    print("PASS")


if __name__ == "__main__":
    main()
