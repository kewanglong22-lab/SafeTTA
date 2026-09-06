
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10J2_richer_morphology_LOMO_confirmatory_fix1.py

Confirmatory comparison of the frozen 2-feature morphology backbone versus the
frozen R10J1 11-feature pre-adaptation morphology panel.

Information boundary
--------------------
- Features were frozen before downstream outcome evaluation in R10J1.
- Target-family labels are evaluation-only.
- Outer folds use source-family labels + sample_id groups only.
- Source-only inner grouped OOF selects the Recall>=0.90 operating threshold.
- No target labels are used for fitting, threshold selection, or tuning.

Methods
-------
M2:
    morph_fg_fraction
    morph_boundary_density

M11:
    frozen 11-feature R10J1 morphology panel

Evaluation
----------
Leave-one-model-family-out:
    DeepLabV3-R50
    PraNet
    SegFormer-B0

Metrics:
    AUROC
    AUPRC
    transferred Recall/FPR/Specificity/EmpiricalPPV/PPV@1%
    target-oracle R90 FPR/PPV@1% (diagnostic only)

Five split seeds:
    20260816..20260820
"""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
OUTER_FOLDS = 5
INNER_FOLDS = 4
TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01

FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]

M2 = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

M11 = [
    "morph_fg_fraction",
    "morph_boundary_density",
    "morph_component_count_8",
    "morph_largest_component_ratio",
    "morph_largest_component_extent",
    "morph_largest_component_eccentricity",
    "morph_largest_component_circularity",
    "morph_largest_component_perimeter_pixels",
    "morph_largest_component_perimeter_area_ratio",
    "morph_hole_count_4",
    "morph_euler_number_8_4",
]

DEFAULT_PANEL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1/"
    "source_mask_morphology_features_labeled.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J2_richer_morphology_LOMO_confirmatory_fix1_v1"
)


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


def threshold_for_recall(y, p, target_recall):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    positive_scores = p[y == 1]
    if positive_scores.size == 0:
        raise AssertionError("No positive samples for threshold selection.")

    scores = np.sort(positive_scores)[::-1]
    k = int(np.ceil(target_recall * len(scores)))
    k = min(max(k, 1), len(scores))

    return float(scores[k - 1])


def grouped_oof_predictions(X, y, groups, seed):
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    p = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Inner grouped split leakage.")

        clf = build_model(seed + fold)
        clf.fit(X[tr], y[tr])
        p[va] = clf.predict_proba(X[va])[:, 1]

    if np.isnan(p).any():
        raise AssertionError("Incomplete inner OOF predictions.")

    return p


def operating_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    empirical_ppv = tp / (tp + fp) if (tp + fp) else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "EmpiricalPPV": float(empirical_ppv),
        "PPV1pct": float(ppv1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10J2 RICHER MORPHOLOGY LOMO CONFIRMATORY FIX1 =====")
    print("STATUS: CONFIRMATORY REPRESENTATION COMPARISON")
    print("FEATURE PANEL CHANGED AFTER R10J1 OUTCOME INSPECTION: NO")
    print("METHODS: M2 vs M11")
    print("GROUP KEY: sample_id")
    print("OUTER FOLDS:", OUTER_FOLDS)
    print("INNER FOLDS:", INNER_FOLDS)
    print("SOURCE-ONLY THRESHOLD TARGET RECALL:", TARGET_RECALL)
    print("TARGET LABELS USED FOR OUTER FOLD CONSTRUCTION: NO")
    print("TARGET LABELS USED FOR FITTING: NO")
    print("TARGET LABELS USED FOR THRESHOLD SELECTION: NO")
    print("TARGET LABELS USED FOR HYPERPARAMETER TUNING: NO")
    print("TARGET LABELS USED FOR EVALUATION ONLY: YES\n")

    panel_path = Path(args.panel)
    print("panel exists:", panel_path.exists(), "path:", panel_path)

    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    df = pd.read_csv(panel_path, low_memory=False)

    required = [
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
    ] + M11

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

    df["_sid"] = norm_sid(df["sample_id"])
    df["_harm"] = pd.to_numeric(
        df["harm_label"], errors="raise"
    ).astype(int)

    print("Rows:", len(df))
    print("Cases:", df["_sid"].nunique())
    print("States:", df["model_state_id"].nunique())
    print("Families:", sorted(df["model_family"].astype(str).unique().tolist()))

    if len(df) != 9000:
        raise AssertionError("Expected 9000 rows.")
    if df["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 cases.")
    if df["model_state_id"].nunique() != 9:
        raise AssertionError("Expected 9 model states.")
    if set(df["model_family"].astype(str).unique()) != set(FAMILIES):
        raise AssertionError("Unexpected model-family set.")

    per_case_family = (
        df.groupby(["_sid", "model_family"])
        .size()
        .reset_index(name="n")
    )
    if not (per_case_family["n"] == 3).all():
        raise AssertionError("Expected 3 rows/case/family.")

    for c in M11:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            raise AssertionError(f"Non-finite frozen morphology feature: {c}")

    methods = {
        "M2_backbone": M2,
        "M11_richer_morphology": M11,
    }

    result_rows = []
    fold_rows = []
    pred_rows = []

    for target_family in FAMILIES:
        source_families = [f for f in FAMILIES if f != target_family]

        source = df[df["model_family"].isin(source_families)].copy()
        target = df[df["model_family"] == target_family].copy()

        if set(source["_sid"]) != set(target["_sid"]):
            raise AssertionError(
                f"Case pools differ for target family {target_family}."
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
        print("Source HARM/NON-HARM:", int(y_source.sum()), int((y_source == 0).sum()))
        print("Target HARM/NON-HARM:", int(y_target.sum()), int((y_target == 0).sum()))
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            print(f"\n######## TARGET={target_family} SEED={seed} ########")

            outer = StratifiedGroupKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=seed,
            )

            splits = list(
                outer.split(
                    np.zeros((len(source), 1)),
                    y_source,
                    g_source,
                )
            )

            for method_name, cols in methods.items():
                p_source = np.full(len(source), np.nan, dtype=float)
                p_target = np.full(len(target), np.nan, dtype=float)
                target_threshold = np.full(len(target), np.nan, dtype=float)

                for fold, (tr_s, te_s) in enumerate(splits):
                    train_cases = set(g_source[tr_s])
                    test_cases = set(g_source[te_s])

                    if train_cases & test_cases:
                        raise AssertionError("Outer group leakage.")

                    te_t = np.flatnonzero(
                        np.isin(g_target, list(test_cases))
                    )

                    if set(g_target[te_t]) != test_cases:
                        raise AssertionError(
                            "Target held-out case-set mismatch."
                        )

                    X_train = matrix(source.iloc[tr_s], cols)
                    y_train = y_source[tr_s]
                    g_train = g_source[tr_s]

                    inner_p = grouped_oof_predictions(
                        X_train,
                        y_train,
                        g_train,
                        seed + 1000 + fold,
                    )

                    threshold = threshold_for_recall(
                        y_train,
                        inner_p,
                        TARGET_RECALL,
                    )

                    clf = build_model(seed + 10000 + fold)
                    clf.fit(X_train, y_train)

                    p_source[te_s] = clf.predict_proba(
                        matrix(source.iloc[te_s], cols)
                    )[:, 1]

                    p_target[te_t] = clf.predict_proba(
                        matrix(target.iloc[te_t], cols)
                    )[:, 1]

                    target_threshold[te_t] = threshold

                    fold_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "method": method_name,
                        "fold": fold,
                        "train_cases": len(train_cases),
                        "test_cases": len(test_cases),
                        "source_train_rows": len(tr_s),
                        "source_test_rows": len(te_s),
                        "target_test_rows": len(te_t),
                        "source_inner_threshold_R90": threshold,
                    })

                if (
                    np.isnan(p_source).any()
                    or np.isnan(p_target).any()
                    or np.isnan(target_threshold).any()
                ):
                    raise AssertionError(
                        f"Incomplete predictions: {target_family}/{seed}/{method_name}"
                    )

                source_auc = float(roc_auc_score(y_source, p_source))
                source_pr = float(average_precision_score(y_source, p_source))
                target_auc = float(roc_auc_score(y_target, p_target))
                target_pr = float(average_precision_score(y_target, p_target))

                # Fold-specific source-only thresholds transferred to target.
                pred_target = (p_target >= target_threshold).astype(int)

                tp = int(np.sum((pred_target == 1) & (y_target == 1)))
                fp = int(np.sum((pred_target == 1) & (y_target == 0)))
                tn = int(np.sum((pred_target == 0) & (y_target == 0)))
                fn = int(np.sum((pred_target == 0) & (y_target == 1)))

                transfer_recall = tp / (tp + fn) if (tp + fn) else np.nan
                transfer_fpr = fp / (fp + tn) if (fp + tn) else np.nan
                transfer_specificity = tn / (tn + fp) if (tn + fp) else np.nan
                transfer_emp_ppv = tp / (tp + fp) if (tp + fp) else np.nan

                pi = PPV_PREVALENCE
                denom = transfer_recall * pi + transfer_fpr * (1 - pi)
                transfer_ppv1 = (
                    transfer_recall * pi / denom
                    if denom > 0
                    else np.nan
                )

                # Target oracle diagnostic only.
                oracle_thr = threshold_for_recall(
                    y_target,
                    p_target,
                    TARGET_RECALL,
                )
                oracle = operating_metrics(
                    y_target,
                    p_target,
                    oracle_thr,
                )

                print(
                    f"{method_name:23s} "
                    f"TARGET_AUROC={target_auc:.6f} "
                    f"TARGET_AUPRC={target_pr:.6f} "
                    f"TRANSFER_RECALL={transfer_recall:.6f} "
                    f"TRANSFER_FPR={transfer_fpr:.6f} "
                    f"TRANSFER_PPV1%={transfer_ppv1:.6f} "
                    f"ORACLE_R90_FPR={oracle['FPR']:.6f}"
                )

                result_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "method": method_name,
                    "source_AUROC": source_auc,
                    "source_AUPRC": source_pr,
                    "target_AUROC": target_auc,
                    "target_AUPRC": target_pr,
                    "transfer_Recall": transfer_recall,
                    "transfer_FPR": transfer_fpr,
                    "transfer_Specificity": transfer_specificity,
                    "transfer_EmpiricalPPV": transfer_emp_ppv,
                    "transfer_PPV1pct": transfer_ppv1,
                    "oracle_target_R90_Recall": oracle["Recall"],
                    "oracle_target_R90_FPR": oracle["FPR"],
                    "oracle_target_R90_PPV1pct": oracle["PPV1pct"],
                })

                for i in range(len(target)):
                    pred_rows.append({
                        "target_family": target_family,
                        "seed": seed,
                        "method": method_name,
                        "sample_id": target.iloc[i]["sample_id"],
                        "model_state_id": target.iloc[i]["model_state_id"],
                        "harm_label": int(y_target[i]),
                        "probability": float(p_target[i]),
                        "source_only_threshold": float(target_threshold[i]),
                    })

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)

    by_family = (
        results.groupby(["target_family", "method"], sort=False)
        .agg(
            seeds=("seed", "count"),
            target_AUROC_mean=("target_AUROC", "mean"),
            target_AUROC_std=("target_AUROC", "std"),
            target_AUPRC_mean=("target_AUPRC", "mean"),
            target_AUPRC_std=("target_AUPRC", "std"),
            transfer_Recall_mean=("transfer_Recall", "mean"),
            transfer_Recall_std=("transfer_Recall", "std"),
            transfer_FPR_mean=("transfer_FPR", "mean"),
            transfer_FPR_std=("transfer_FPR", "std"),
            transfer_PPV1pct_mean=("transfer_PPV1pct", "mean"),
            transfer_PPV1pct_std=("transfer_PPV1pct", "std"),
            oracle_R90_FPR_mean=("oracle_target_R90_FPR", "mean"),
            oracle_R90_FPR_std=("oracle_target_R90_FPR", "std"),
        )
        .reset_index()
    )

    macro_seed = (
        results.groupby(["method", "seed"], sort=False)
        .agg(
            macro_target_AUROC=("target_AUROC", "mean"),
            macro_target_AUPRC=("target_AUPRC", "mean"),
            macro_transfer_Recall=("transfer_Recall", "mean"),
            macro_transfer_FPR=("transfer_FPR", "mean"),
            macro_transfer_PPV1pct=("transfer_PPV1pct", "mean"),
            macro_oracle_R90_FPR=("oracle_target_R90_FPR", "mean"),
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
            macro_transfer_Recall_mean=("macro_transfer_Recall", "mean"),
            macro_transfer_Recall_std=("macro_transfer_Recall", "std"),
            macro_transfer_FPR_mean=("macro_transfer_FPR", "mean"),
            macro_transfer_FPR_std=("macro_transfer_FPR", "std"),
            macro_transfer_PPV1pct_mean=("macro_transfer_PPV1pct", "mean"),
            macro_transfer_PPV1pct_std=("macro_transfer_PPV1pct", "std"),
            macro_oracle_R90_FPR_mean=("macro_oracle_R90_FPR", "mean"),
            macro_oracle_R90_FPR_std=("macro_oracle_R90_FPR", "std"),
        )
        .reset_index()
    )

    print("\n\n===== R10J2 BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10J2 MACRO SUMMARY =====")
    print(macro.to_string(index=False))

    m2 = macro[macro["method"] == "M2_backbone"].iloc[0]
    m11 = macro[macro["method"] == "M11_richer_morphology"].iloc[0]

    delta_auc = (
        float(m11["macro_target_AUROC_mean"])
        - float(m2["macro_target_AUROC_mean"])
    )
    delta_auprc = (
        float(m11["macro_target_AUPRC_mean"])
        - float(m2["macro_target_AUPRC_mean"])
    )
    delta_fpr = (
        float(m11["macro_transfer_FPR_mean"])
        - float(m2["macro_transfer_FPR_mean"])
    )
    delta_oracle_fpr = (
        float(m11["macro_oracle_R90_FPR_mean"])
        - float(m2["macro_oracle_R90_FPR_mean"])
    )

    m2_family = by_family[
        by_family["method"] == "M2_backbone"
    ]
    m11_family = by_family[
        by_family["method"] == "M11_richer_morphology"
    ]

    worst_m11_auc = float(
        m11_family["target_AUROC_mean"].min()
    )
    worst_delta_auc = float(
        (
            m11_family.set_index("target_family")["target_AUROC_mean"]
            - m2_family.set_index("target_family")["target_AUROC_mean"]
        ).min()
    )

    print("\n===== CONFIRMATORY DELTAS =====")
    print("Delta macro AUROC (M11-M2):", delta_auc)
    print("Delta macro AUPRC (M11-M2):", delta_auprc)
    print("Delta transferred FPR (M11-M2):", delta_fpr)
    print("Delta oracle R90 FPR (M11-M2):", delta_oracle_fpr)
    print("Worst M11 family AUROC:", worst_m11_auc)
    print("Worst family AUROC delta (M11-M2):", worst_delta_auc)

    # Frozen decision rule.
    if (
        delta_auc >= 0.02
        and delta_fpr <= -0.05
        and delta_oracle_fpr <= -0.04
        and float(m11["macro_transfer_Recall_mean"]) >= 0.85
        and worst_m11_auc >= 0.75
        and worst_delta_auc >= -0.01
    ):
        decision = "RICHER_MORPHOLOGY_STRONGLY_SUPPORTED"
    elif (
        delta_auc >= 0.01
        and delta_fpr <= -0.02
        and float(m11["macro_transfer_Recall_mean"]) >= 0.85
        and worst_delta_auc >= -0.02
    ):
        decision = "RICHER_MORPHOLOGY_PARTIALLY_SUPPORTED"
    else:
        decision = "RICHER_MORPHOLOGY_NOT_SUPPORTED"

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)

    results.to_csv(
        out / "R10J2_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10J2_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10J2_macro_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10J2_fold_audit.csv",
        index=False,
    )
    preds.to_csv(
        out / "R10J2_target_predictions.csv",
        index=False,
    )

    print("\nOutputs:")
    print(out / "R10J2_per_seed_family_metrics.csv")
    print(out / "R10J2_by_family_summary.csv")
    print(out / "R10J2_macro_summary.csv")
    print(out / "R10J2_fold_audit.csv")
    print(out / "R10J2_target_predictions.csv")
    print("PASS")


if __name__ == "__main__":
    main()
