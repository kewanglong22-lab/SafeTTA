
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H3_case_heldout_cross_domain_transfer_fix1.py

STRICT paired cross-representation / cross-domain transfer evaluation.

Scientific status
-----------------
NeoPolyp is NOT treated here as an independent external dataset because the
source-representation table and the recovered external-representation table
contain the same 1000 sample IDs.

Therefore this experiment is explicitly:
    case-held-out cross-domain / cross-representation transfer

NOT:
    independent external-dataset validation

Protocol
--------
- 1000 case groups, 9 rows/case.
- Same case IDs exist in source and external representations.
- Outer 5-fold StratifiedGroupKFold by sample_id.
- For every outer fold:
    * SGUC/baselines fit ONLY on source-representation rows from training cases.
    * Source outer-test rows and external rows from held-out cases are predicted.
    * No held-out case is present in fitting.
- Operating threshold:
    * learned ONLY from an inner grouped OOF on the source outer-training cases;
    * target Recall >= 0.90 on source inner-OOF predictions;
    * transferred unchanged to held-out external rows.
- External labels:
    * evaluation only;
    * never used for model fitting, feature selection, threshold selection,
      fold assignment, or hyperparameter tuning.
- Methods:
    uncertainty_only
    concat
    SGUC_fixed = [U, S, U * tanh(mean(S))]
- Five outer-split seeds: 20260816..20260820.
"""

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

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
TARGET_PREVALENCE = 0.01

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
    "Q1_R10H3_case_heldout_cross_domain_transfer_fix1_v1"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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


def build_method_matrix(df, u_cols, s_cols, method):
    U = df[u_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    S = df[s_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    if method == "uncertainty_only":
        return U
    if method == "concat":
        return np.concatenate([U, S], axis=1)
    if method == "SGUC_fixed":
        gate = np.tanh(np.nanmean(S, axis=1, keepdims=True))
        # preserve NaNs for imputer rather than silently zero-filling
        return np.concatenate([U, S, U * gate], axis=1)

    raise ValueError(method)


def threshold_for_recall(y, p, target=0.90):
    """
    Highest empirical threshold whose recall is >= target.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = p[y == 1]
    if len(pos) == 0:
        raise ValueError("No positives available for threshold estimation.")

    s = np.sort(pos)[::-1]
    k = int(np.ceil(target * len(s)))
    k = min(max(k, 1), len(s))
    return float(s[k - 1])


def binary_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    pi = TARGET_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv_1pct = recall * pi / denom if denom > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "Empirical_PPV": float(precision),
        "PPV_at_1pct": float(ppv_1pct),
    }


def grouped_oof_predictions(X, y, groups, seed, n_splits):
    """
    Grouped OOF on source data only.
    """
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    pred = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        tr_groups = set(groups[tr])
        va_groups = set(groups[va])
        if tr_groups & va_groups:
            raise AssertionError("Group leakage in inner OOF.")

        model = build_model(seed + fold)
        model.fit(X[tr], y[tr])
        pred[va] = model.predict_proba(X[va])[:, 1]

    if np.isnan(pred).any():
        raise AssertionError("Incomplete grouped OOF predictions.")

    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H3 CASE-HELD-OUT CROSS-DOMAIN TRANSFER FIX1 =====")
    print("INDEPENDENT EXTERNAL DATASET CLAIM: NO")
    print("EXPERIMENT TYPE: paired case-held-out cross-domain transfer")
    print("GROUP KEY: sample_id")
    print("OUTER FOLDS:", OUTER_FOLDS)
    print("INNER FOLDS FOR THRESHOLD:", INNER_FOLDS)
    print("TARGET SOURCE RECALL FOR THRESHOLD:", TARGET_RECALL)
    print("EXTERNAL LABELS USED FOR FITTING: NO")
    print("EXTERNAL LABELS USED FOR THRESHOLD SELECTION: NO")
    print("EXTERNAL LABELS USED FOR HYPERPARAMETER TUNING: NO\n")

    source_path = Path(args.source)
    ext_path = Path(args.external)

    print("source exists:", source_path.exists(), "path:", source_path)
    print("external exists:", ext_path.exists(), "path:", ext_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not ext_path.exists():
        raise FileNotFoundError(ext_path)

    src = pd.read_csv(source_path, low_memory=False)
    ext = pd.read_csv(ext_path, low_memory=False)

    if "sample_id" not in src.columns or "sample_id" not in ext.columns:
        raise AssertionError("sample_id missing.")
    if "outcome" not in src.columns:
        raise AssertionError("Source outcome missing.")
    if "harm_label" not in ext.columns:
        raise AssertionError("External harm_label missing.")

    src["_sid_norm"] = norm_sid(src["sample_id"])
    ext["_sid_norm"] = norm_sid(ext["sample_id"])

    src_ids = set(src["_sid_norm"])
    ext_ids = set(ext["_sid_norm"])

    print("\n===== CASE-POOL AUDIT =====")
    print("Source rows:", len(src))
    print("External rows:", len(ext))
    print("Source unique cases:", len(src_ids))
    print("External unique cases:", len(ext_ids))
    print("Case-ID intersection:", len(src_ids & ext_ids))
    print("Case-ID sets identical:", src_ids == ext_ids)

    if src_ids != ext_ids:
        raise AssertionError(
            "This protocol requires the paired source/external representations "
            "to have identical case pools."
        )

    src_mult = src["_sid_norm"].value_counts()
    ext_mult = ext["_sid_norm"].value_counts()
    print("Source multiplicity min/max:", int(src_mult.min()), int(src_mult.max()))
    print("External multiplicity min/max:", int(ext_mult.min()), int(ext_mult.max()))

    if not ((src_mult == 9).all() and (ext_mult == 9).all()):
        raise AssertionError("Expected exactly 9 rows per case in both representations.")

    y_src = (
        src["outcome"].astype(str).str.upper().eq("HARM").astype(int).to_numpy()
    )
    y_ext = ext["harm_label"].astype(int).to_numpy()

    print("Source HARM/NON-HARM:", int(y_src.sum()), int((y_src == 0).sum()))
    print("External HARM/NON-HARM:", int(y_ext.sum()), int((y_ext == 0).sum()))

    # Freeze feature names from SOURCE only, then require exact availability in external.
    u_cols = [
        c for c in src.columns
        if any(k in c.lower() for k in [
            "entropy", "prob", "confidence", "logit", "rur"
        ])
        and "outcome" not in c.lower()
        and "dice" not in c.lower()
    ]

    s_cols = [
        c for c in src.columns
        if (
            "struct" in c.lower()
            or "boundary_density" in c.lower()
            or "source_fg_fraction" == c.lower()
        )
        and "rur_" not in c.lower()
    ]

    # Preserve source column order while removing accidental duplicates.
    u_cols = list(dict.fromkeys(u_cols))
    s_cols = list(dict.fromkeys(s_cols))

    print("\n===== FROZEN FEATURE PANEL =====")
    print("Uncertainty features:", len(u_cols))
    for c in u_cols:
        print("  U:", c)
    print("Structure features:", len(s_cols))
    for c in s_cols:
        print("  S:", c)

    # The frozen R10G/H lineage should be exactly 27 + 4.
    if len(u_cols) != 27:
        raise AssertionError(
            f"Expected frozen 27 uncertainty features, found {len(u_cols)}."
        )
    if len(s_cols) != 4:
        raise AssertionError(
            f"Expected frozen 4 structure features, found {len(s_cols)}."
        )

    missing_ext = [c for c in u_cols + s_cols if c not in ext.columns]
    print("Missing frozen features in external:", missing_ext)
    if missing_ext:
        raise AssertionError("External feature panel does not match frozen source panel.")

    methods = ["uncertainty_only", "concat", "SGUC_fixed"]

    Xs = {m: build_method_matrix(src, u_cols, s_cols, m) for m in methods}
    Xe = {m: build_method_matrix(ext, u_cols, s_cols, m) for m in methods}

    groups_src = src["_sid_norm"].to_numpy()
    groups_ext = ext["_sid_norm"].to_numpy()

    per_seed_rows = []
    src_pred_rows = []
    ext_pred_rows = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n\n######## SEED {seed} ########")

        outer = StratifiedGroupKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        # Build outer folds from source groups. External rows are selected by
        # the held-out CASE IDs, never by row order.
        split_list = list(outer.split(
            np.zeros((len(src), 1)),
            y_src,
            groups_src,
        ))

        for method in methods:
            print(f"\n===== {method} =====")

            p_src = np.full(len(src), np.nan)
            p_ext = np.full(len(ext), np.nan)
            decision_ext = np.full(len(ext), -1, dtype=int)
            fold_threshold = np.full(len(ext), np.nan)

            for fold, (tr, te_src) in enumerate(split_list):
                train_cases = set(groups_src[tr])
                test_cases = set(groups_src[te_src])

                if train_cases & test_cases:
                    raise AssertionError("Outer source case leakage.")

                te_ext = np.flatnonzero(np.isin(groups_ext, list(test_cases)))

                if len(te_ext) == 0:
                    raise AssertionError("No paired external rows for outer test cases.")
                if set(groups_ext[te_ext]) != test_cases:
                    raise AssertionError("External held-out case set mismatch.")
                if set(groups_ext[te_ext]) & train_cases:
                    raise AssertionError("External test cases leaked into source training cases.")

                # Inner grouped OOF ONLY on outer source-training cases.
                inner_seed = seed + 1000 + fold
                inner_pred = grouped_oof_predictions(
                    Xs[method][tr],
                    y_src[tr],
                    groups_src[tr],
                    inner_seed,
                    INNER_FOLDS,
                )
                threshold = threshold_for_recall(
                    y_src[tr],
                    inner_pred,
                    TARGET_RECALL,
                )

                # Final outer-fold model: source training cases only.
                model = build_model(seed + 10000 + fold)
                model.fit(Xs[method][tr], y_src[tr])

                p_src[te_src] = model.predict_proba(Xs[method][te_src])[:, 1]
                p_ext[te_ext] = model.predict_proba(Xe[method][te_ext])[:, 1]

                decision_ext[te_ext] = (p_ext[te_ext] >= threshold).astype(int)
                fold_threshold[te_ext] = threshold

                print(
                    f"fold={fold} "
                    f"train_cases={len(train_cases)} "
                    f"test_cases={len(test_cases)} "
                    f"source_train_rows={len(tr)} "
                    f"source_test_rows={len(te_src)} "
                    f"external_test_rows={len(te_ext)} "
                    f"source_inner_threshold={threshold:.6f}"
                )

            if (
                np.isnan(p_src).any()
                or np.isnan(p_ext).any()
                or np.isnan(fold_threshold).any()
                or np.any(decision_ext < 0)
            ):
                raise AssertionError("Incomplete outer predictions.")

            # Source grouped OOF discrimination.
            src_auroc = roc_auc_score(y_src, p_src)
            src_auprc = average_precision_score(y_src, p_src)

            # External paired held-out discrimination.
            ext_auroc = roc_auc_score(y_ext, p_ext)
            ext_auprc = average_precision_score(y_ext, p_ext)

            # Transferred operating point: fold-specific source-only thresholds.
            tp = int(np.sum((decision_ext == 1) & (y_ext == 1)))
            fp = int(np.sum((decision_ext == 1) & (y_ext == 0)))
            tn = int(np.sum((decision_ext == 0) & (y_ext == 0)))
            fn = int(np.sum((decision_ext == 0) & (y_ext == 1)))

            ext_recall_transfer = tp / (tp + fn)
            ext_fpr_transfer = fp / (fp + tn)
            ext_specificity_transfer = tn / (tn + fp)
            ext_emp_ppv_transfer = tp / (tp + fp) if (tp + fp) else np.nan

            pi = TARGET_PREVALENCE
            denom = (
                ext_recall_transfer * pi
                + ext_fpr_transfer * (1.0 - pi)
            )
            ext_ppv1_transfer = (
                ext_recall_transfer * pi / denom if denom > 0 else np.nan
            )

            # Oracle target FPR@90 is a ranking diagnostic ONLY.
            oracle_thr = threshold_for_recall(y_ext, p_ext, TARGET_RECALL)
            oracle = binary_metrics(y_ext, p_ext, oracle_thr)

            print(
                f"SOURCE GROUP-OOF: "
                f"AUROC={src_auroc:.6f} "
                f"AUPRC={src_auprc:.6f}"
            )
            print(
                f"EXTERNAL CASE-HELD-OUT: "
                f"AUROC={ext_auroc:.6f} "
                f"AUPRC={ext_auprc:.6f}"
            )
            print(
                f"TRANSFERRED SOURCE THRESHOLD: "
                f"Recall={ext_recall_transfer:.6f} "
                f"FPR={ext_fpr_transfer:.6f} "
                f"Specificity={ext_specificity_transfer:.6f} "
                f"EmpPPV={ext_emp_ppv_transfer:.6f} "
                f"PPV@1%={ext_ppv1_transfer:.6f}"
            )
            print(
                f"ORACLE TARGET R>=0.90 DIAGNOSTIC ONLY: "
                f"Recall={oracle['Recall']:.6f} "
                f"FPR={oracle['FPR']:.6f} "
                f"PPV@1%={oracle['PPV_at_1pct']:.6f}"
            )

            per_seed_rows.append({
                "seed": seed,
                "method": method,
                "source_group_oof_AUROC": src_auroc,
                "source_group_oof_AUPRC": src_auprc,
                "external_caseheldout_AUROC": ext_auroc,
                "external_caseheldout_AUPRC": ext_auprc,
                "external_transfer_Recall": ext_recall_transfer,
                "external_transfer_FPR": ext_fpr_transfer,
                "external_transfer_Specificity": ext_specificity_transfer,
                "external_transfer_EmpiricalPPV": ext_emp_ppv_transfer,
                "external_transfer_PPV1pct": ext_ppv1_transfer,
                "external_oracle_R90_Recall": oracle["Recall"],
                "external_oracle_R90_FPR": oracle["FPR"],
                "external_oracle_R90_PPV1pct": oracle["PPV_at_1pct"],
                "independent_external_dataset": "NO",
                "external_labels_used_for_fit": "NO",
                "external_labels_used_for_threshold": "NO",
            })

            for i in range(len(src)):
                src_pred_rows.append({
                    "seed": seed,
                    "method": method,
                    "sample_id": src.iloc[i]["sample_id"],
                    "true_harm": int(y_src[i]),
                    "group_oof_probability": float(p_src[i]),
                })

            for i in range(len(ext)):
                ext_pred_rows.append({
                    "seed": seed,
                    "method": method,
                    "sample_id": ext.iloc[i]["sample_id"],
                    "model_state_id": (
                        ext.iloc[i]["model_state_id"]
                        if "model_state_id" in ext.columns else ""
                    ),
                    "model_family": (
                        ext.iloc[i]["model_family"]
                        if "model_family" in ext.columns else ""
                    ),
                    "true_harm": int(y_ext[i]),
                    "caseheldout_probability": float(p_ext[i]),
                    "source_only_fold_threshold": float(fold_threshold[i]),
                    "transferred_decision": int(decision_ext[i]),
                })

    metrics = pd.DataFrame(per_seed_rows)

    summary = (
        metrics.groupby("method", sort=False)
        .agg(
            seeds=("seed", "count"),
            source_AUROC_mean=("source_group_oof_AUROC", "mean"),
            source_AUROC_std=("source_group_oof_AUROC", "std"),
            source_AUPRC_mean=("source_group_oof_AUPRC", "mean"),
            source_AUPRC_std=("source_group_oof_AUPRC", "std"),
            external_AUROC_mean=("external_caseheldout_AUROC", "mean"),
            external_AUROC_std=("external_caseheldout_AUROC", "std"),
            external_AUPRC_mean=("external_caseheldout_AUPRC", "mean"),
            external_AUPRC_std=("external_caseheldout_AUPRC", "std"),
            transfer_Recall_mean=("external_transfer_Recall", "mean"),
            transfer_Recall_std=("external_transfer_Recall", "std"),
            transfer_FPR_mean=("external_transfer_FPR", "mean"),
            transfer_FPR_std=("external_transfer_FPR", "std"),
            transfer_PPV1pct_mean=("external_transfer_PPV1pct", "mean"),
            transfer_PPV1pct_std=("external_transfer_PPV1pct", "std"),
            oracle_R90_FPR_mean=("external_oracle_R90_FPR", "mean"),
            oracle_R90_FPR_std=("external_oracle_R90_FPR", "std"),
        )
        .reset_index()
    )

    print("\n\n===== R10H3 FINAL SUMMARY =====")
    print(summary.to_string(index=False))

    metrics_path = out / "R10H3_per_seed_metrics.csv"
    summary_path = out / "R10H3_summary.csv"
    src_pred_path = out / "R10H3_source_group_oof_predictions.csv"
    ext_pred_path = out / "R10H3_external_caseheldout_predictions.csv"

    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(src_pred_rows).to_csv(src_pred_path, index=False)
    pd.DataFrame(ext_pred_rows).to_csv(ext_pred_path, index=False)

    print("\n===== FINAL PROTOCOL STATUS =====")
    print("CASE LEAKAGE ACROSS OUTER FOLDS: NO")
    print("EXTERNAL LABELS USED FOR FITTING: NO")
    print("EXTERNAL LABELS USED FOR THRESHOLD: NO")
    print("INDEPENDENT EXTERNAL DATASET CLAIM: NO")
    print("VALID CLAIM: CASE-HELD-OUT CROSS-DOMAIN TRANSFER")

    print("\nOutputs:")
    print(metrics_path)
    print(summary_path)
    print(src_pred_path)
    print(ext_pred_path)
    print("PASS")


if __name__ == "__main__":
    main()
