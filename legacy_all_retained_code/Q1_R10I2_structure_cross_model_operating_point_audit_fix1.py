
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10I2_structure_cross_model_operating_point_audit_fix1.py

Purpose
-------
R10I1 showed that relation-stable feature screening only partially helps,
whereas the 2-feature morphology baseline is the strongest leave-one-model-
family-out result:

    macro target AUROC ~0.806

Before elevating morphology to the new paper backbone, this audit verifies:

1) The two structure descriptors genuinely vary across model families and are
   not another identity-preservation artifact.
2) Their safety relationship direction is stable across all three families.
3) A source-only high-recall operating threshold transfers to an unseen model
   family.
4) Correct high-recall metrics are reported.

Protocol
--------
- Families: DeepLabV3-R50, PraNet, SegFormer-B0
- Hold out one family as TARGET; other two are SOURCE.
- Outer 5-fold StratifiedGroupKFold by sample_id, source labels only.
- Inner 4-fold grouped OOF on source-training cases only to select threshold
  satisfying Recall >= 0.90.
- Target labels are evaluation-only.
- Features:
    source_fg_fraction
    source_boundary_density
- Five seeds: 20260816..20260820

No target labels are used for:
- fold construction
- fitting
- threshold selection
- hyperparameter tuning

This remains a route-validation experiment, not independent cohort validation.
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

FAMILIES = ["DeepLabV3-R50", "PraNet", "SegFormer-B0"]
STRUCTURE = ["source_fg_fraction", "source_boundary_density"]

DEFAULT_PANEL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10A0_invariant_feature_table_v1/"
    "invariant_feature_table.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10I2_structure_cross_model_operating_point_audit_fix1_v1"
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


def matrix(df):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in STRUCTURE
    ])


def threshold_for_recall(y, p, target=0.90):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    if len(pos) == 0:
        raise AssertionError("No positive samples for threshold selection.")
    s = np.sort(pos)[::-1]
    k = int(np.ceil(target * len(s)))
    k = min(max(k, 1), len(s))
    return float(s[k - 1])


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
    precision = tp / (tp + fp) if (tp + fp) else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1.0 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Recall": float(recall),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "EmpiricalPPV": float(precision),
        "PPV1pct": float(ppv1),
    }


def grouped_oof_predictions(X, y, groups, seed):
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    p = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Inner group leakage.")
        m = build_model(seed + fold)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]

    if np.isnan(p).any():
        raise AssertionError("Incomplete inner grouped OOF predictions.")
    return p


def safe_univariate_auc(y, x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.nan
    med = np.nanmedian(x[finite])
    x = np.where(np.isfinite(x), x, med)
    return float(roc_auc_score(y, x))


def feature_identity_audit(df):
    """
    Audit whether the structure vector is preserved across model families
    within the same case. No row matching across states is assumed.

    For each case and each family pair, compare the multiset of 3 structure
    vectors. If all 3 vectors are numerically identical as a multiset, mark
    that case-family pair as preserved.
    """
    rows = []
    tol = 1e-14

    def canonical_rows(g):
        A = g[STRUCTURE].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # deterministic lexicographic sorting
        B = np.where(np.isnan(A), np.inf, A)
        order = np.lexsort((B[:, 1], B[:, 0]))
        return A[order]

    sids = sorted(df["_sid"].unique())

    for i in range(len(FAMILIES)):
        for j in range(i + 1, len(FAMILIES)):
            f1, f2 = FAMILIES[i], FAMILIES[j]
            same = 0
            maxdiffs = []

            for sid in sids:
                a = df[(df["_sid"] == sid) & (df["model_family"] == f1)]
                b = df[(df["_sid"] == sid) & (df["model_family"] == f2)]

                if len(a) != 3 or len(b) != 3:
                    raise AssertionError(
                        f"{sid}: expected 3 rows/family, got {len(a)}, {len(b)}"
                    )

                A = canonical_rows(a)
                B = canonical_rows(b)

                nan_match = np.isnan(A) == np.isnan(B)
                if not nan_match.all():
                    d = np.inf
                    ok = False
                else:
                    finite = np.isfinite(A) & np.isfinite(B)
                    diff = np.zeros_like(A)
                    diff[finite] = np.abs(A[finite] - B[finite])
                    d = float(np.max(diff)) if diff.size else 0.0
                    ok = bool(np.all(diff[finite] <= tol))

                if ok:
                    same += 1
                maxdiffs.append(d)

            rows.append({
                "family_1": f1,
                "family_2": f2,
                "cases": len(sids),
                "identical_structure_multiset_cases": same,
                "identical_fraction": same / len(sids),
                "median_case_maxdiff": float(np.median(maxdiffs)),
                "max_case_maxdiff": float(np.max(maxdiffs)),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10I2 STRUCTURE CROSS-MODEL OPERATING-POINT AUDIT FIX1 =====")
    print("STATUS: ROUTE VALIDATION")
    print("FEATURES:", STRUCTURE)
    print("TARGET LABELS USED FOR FOLD CONSTRUCTION: NO")
    print("TARGET LABELS USED FOR FITTING: NO")
    print("TARGET LABELS USED FOR THRESHOLD SELECTION: NO")
    print("TARGET LABELS USED FOR HYPERPARAMETER TUNING: NO")
    print("TARGET LABELS USED FOR EVALUATION ONLY: YES")
    print("TARGET RECALL FOR SOURCE THRESHOLD:", TARGET_RECALL)
    print("GROUP KEY: sample_id\n")

    panel_path = Path(args.panel)
    print("panel exists:", panel_path.exists(), "path:", panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    df = pd.read_csv(panel_path, low_memory=False)

    required = ["sample_id", "model_family", "outcome"] + STRUCTURE
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

    df["_sid"] = norm_sid(df["sample_id"])
    df["_harm"] = (
        df["outcome"].astype(str).str.upper().eq("HARM").astype(int)
    )

    if set(df["model_family"].astype(str).unique()) != set(FAMILIES):
        raise AssertionError("Unexpected model-family set.")

    per_case_family = df.groupby(["_sid", "model_family"]).size()
    if not (per_case_family == 3).all():
        raise AssertionError("Expected exactly 3 rows/case/family.")

    print("Rows:", len(df))
    print("Cases:", df["_sid"].nunique())

    print("\n===== CROSS-FAMILY STRUCTURE IDENTITY AUDIT =====")
    identity = feature_identity_audit(df)
    print(identity.to_string(index=False))

    all_identity = bool((identity["identical_fraction"] == 1.0).all())
    print("ALL FAMILY PAIRS IDENTICAL:", all_identity)

    if all_identity:
        print("DECISION: STOP_STRUCTURE_ROUTE_IDENTITY_ARTIFACT")
        identity.to_csv(out / "R10I2_structure_identity_audit.csv", index=False)
        return

    print("\n===== FAMILY-WISE UNIVARIATE SAFETY RELATION =====")
    relation_rows = []

    for fam in FAMILIES:
        g = df[df["model_family"] == fam]
        y = g["_harm"].to_numpy(int)

        row = {"model_family": fam}
        for c in STRUCTURE:
            auc = safe_univariate_auc(
                y,
                pd.to_numeric(g[c], errors="coerce").to_numpy(float),
            )
            row[c + "_AUROC"] = auc
        relation_rows.append(row)

    relation = pd.DataFrame(relation_rows)
    print(relation.to_string(index=False))

    result_rows = []
    fold_rows = []

    for target_family in FAMILIES:
        source_families = [f for f in FAMILIES if f != target_family]

        source = df[df["model_family"].isin(source_families)].copy()
        target = df[df["model_family"] == target_family].copy()

        y_source = source["_harm"].to_numpy(int)
        y_target = target["_harm"].to_numpy(int)
        g_source = source["_sid"].to_numpy()
        g_target = target["_sid"].to_numpy()

        print("\n\n============================================================")
        print("TARGET FAMILY:", target_family)
        print("SOURCE FAMILIES:", source_families)
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            outer = StratifiedGroupKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=seed,
            )
            splits = list(outer.split(
                np.zeros((len(source), 1)),
                y_source,
                g_source,
            ))

            p_source = np.full(len(source), np.nan)
            p_target = np.full(len(target), np.nan)
            thresholds_target = np.full(len(target), np.nan)

            for fold, (tr_s, te_s) in enumerate(splits):
                train_cases = set(g_source[tr_s])
                test_cases = set(g_source[te_s])

                if train_cases & test_cases:
                    raise AssertionError("Outer group leakage.")

                te_t = np.flatnonzero(np.isin(g_target, list(test_cases)))

                if set(g_target[te_t]) != test_cases:
                    raise AssertionError("Target test-case set mismatch.")

                X_train = matrix(source.iloc[tr_s])
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

                m = build_model(seed + 10000 + fold)
                m.fit(X_train, y_train)

                p_source[te_s] = m.predict_proba(
                    matrix(source.iloc[te_s])
                )[:, 1]

                p_target[te_t] = m.predict_proba(
                    matrix(target.iloc[te_t])
                )[:, 1]

                thresholds_target[te_t] = threshold

                fold_rows.append({
                    "target_family": target_family,
                    "seed": seed,
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
                or np.isnan(thresholds_target).any()
            ):
                raise AssertionError("Incomplete predictions.")

            source_auc = float(roc_auc_score(y_source, p_source))
            source_pr = float(average_precision_score(y_source, p_source))
            target_auc = float(roc_auc_score(y_target, p_target))
            target_pr = float(average_precision_score(y_target, p_target))

            # Fold-specific source-only thresholds transferred to target.
            pred_target = (p_target >= thresholds_target).astype(int)

            tp = int(np.sum((pred_target == 1) & (y_target == 1)))
            fp = int(np.sum((pred_target == 1) & (y_target == 0)))
            tn = int(np.sum((pred_target == 0) & (y_target == 0)))
            fn = int(np.sum((pred_target == 0) & (y_target == 1)))

            recall = tp / (tp + fn)
            fpr = fp / (fp + tn)
            specificity = tn / (tn + fp)
            emp_ppv = tp / (tp + fp) if (tp + fp) else np.nan

            pi = PPV_PREVALENCE
            denom = recall * pi + fpr * (1 - pi)
            ppv1 = recall * pi / denom if denom > 0 else np.nan

            # Oracle target operating point: diagnostic only.
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
                f"seed={seed} "
                f"TARGET_AUROC={target_auc:.6f} "
                f"TARGET_AUPRC={target_pr:.6f} "
                f"TRANSFER_RECALL={recall:.6f} "
                f"TRANSFER_FPR={fpr:.6f} "
                f"TRANSFER_PPV1%={ppv1:.6f} "
                f"ORACLE_R90_FPR={oracle['FPR']:.6f}"
            )

            result_rows.append({
                "target_family": target_family,
                "seed": seed,
                "source_AUROC": source_auc,
                "source_AUPRC": source_pr,
                "target_AUROC": target_auc,
                "target_AUPRC": target_pr,
                "transfer_Recall": recall,
                "transfer_FPR": fpr,
                "transfer_Specificity": specificity,
                "transfer_EmpiricalPPV": emp_ppv,
                "transfer_PPV1pct": ppv1,
                "oracle_target_R90_Recall": oracle["Recall"],
                "oracle_target_R90_FPR": oracle["FPR"],
                "oracle_target_R90_PPV1pct": oracle["PPV1pct"],
            })

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)

    by_family = (
        results.groupby("target_family", sort=False)
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
        results.groupby("seed", sort=False)
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

    macro = pd.DataFrame([{
        "seeds": len(macro_seed),
        "macro_target_AUROC_mean": macro_seed["macro_target_AUROC"].mean(),
        "macro_target_AUROC_std": macro_seed["macro_target_AUROC"].std(),
        "macro_target_AUPRC_mean": macro_seed["macro_target_AUPRC"].mean(),
        "macro_target_AUPRC_std": macro_seed["macro_target_AUPRC"].std(),
        "macro_transfer_Recall_mean": macro_seed["macro_transfer_Recall"].mean(),
        "macro_transfer_Recall_std": macro_seed["macro_transfer_Recall"].std(),
        "macro_transfer_FPR_mean": macro_seed["macro_transfer_FPR"].mean(),
        "macro_transfer_FPR_std": macro_seed["macro_transfer_FPR"].std(),
        "macro_transfer_PPV1pct_mean": macro_seed["macro_transfer_PPV1pct"].mean(),
        "macro_transfer_PPV1pct_std": macro_seed["macro_transfer_PPV1pct"].std(),
        "macro_oracle_R90_FPR_mean": macro_seed["macro_oracle_R90_FPR"].mean(),
        "macro_oracle_R90_FPR_std": macro_seed["macro_oracle_R90_FPR"].std(),
    }])

    print("\n===== R10I2 BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10I2 MACRO SUMMARY =====")
    print(macro.to_string(index=False))

    worst_auc = float(by_family["target_AUROC_mean"].min())
    macro_auc = float(macro.iloc[0]["macro_target_AUROC_mean"])
    macro_fpr = float(macro.iloc[0]["macro_transfer_FPR_mean"])
    macro_recall = float(macro.iloc[0]["macro_transfer_Recall_mean"])

    print("\n===== ROUTE DECISION =====")
    print("Macro target AUROC:", macro_auc)
    print("Worst-family target AUROC:", worst_auc)
    print("Macro transferred recall:", macro_recall)
    print("Macro transferred FPR:", macro_fpr)

    if (
        macro_auc >= 0.80
        and worst_auc >= 0.75
        and macro_recall >= 0.85
        and macro_fpr <= 0.50
    ):
        decision = "STRUCTURE_BACKBONE_STRONGLY_SUPPORTED"
    elif (
        macro_auc >= 0.78
        and worst_auc >= 0.72
    ):
        decision = "STRUCTURE_BACKBONE_RANKING_SUPPORTED_OPERATING_POINT_WEAK"
    else:
        decision = "STRUCTURE_BACKBONE_NOT_SUPPORTED"

    print("DECISION:", decision)

    identity.to_csv(
        out / "R10I2_structure_identity_audit.csv",
        index=False,
    )
    relation.to_csv(
        out / "R10I2_family_univariate_relation.csv",
        index=False,
    )
    results.to_csv(
        out / "R10I2_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10I2_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10I2_macro_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10I2_fold_audit.csv",
        index=False,
    )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
