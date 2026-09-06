
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H4_structure_transfer_mechanism_audit_fix1.py

Purpose
-------
After R10H3 showed catastrophic source->external transfer failure for
uncertainty / concat / SGUC, this audit answers the next mechanistic question:

    Does the domain-stable STRUCTURE branch transfer better?

It also measures an evaluation-only target-supervised ceiling to distinguish:
A) transfer failure despite target information being present, versus
B) representation information loss on the external representation.

IMPORTANT CLAIM BOUNDARY
------------------------
Primary transfer metrics:
    source labels only for fitting;
    source-only thresholds;
    external labels evaluation only.

Target-supervised ceiling:
    diagnostic only;
    external labels ARE used in grouped OOF fitting;
    therefore it is NOT a deployable method result and must never be reported
    as external source-free performance.

Protocol
--------
- Same 1000 cases / 9 rows per case in source and external representations.
- Group key: sample_id.
- 5-fold StratifiedGroupKFold, seeds 20260816..20260820.
- Primary methods:
    uncertainty_only
    structure_only
    concat
    SGUC_fixed
- Source->external fitting:
    fit on source-training cases only;
    evaluate on paired held-out external cases.
- Target-supervised diagnostic ceiling:
    grouped OOF on external labels, same held-out case discipline.
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
N_SPLITS = 5

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
    "Q1_R10H4_structure_transfer_mechanism_audit_fix1_v1"
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


def model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def feature_panel(df):
    u_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "entropy", "prob", "confidence", "logit", "rur"
        ])
        and "outcome" not in c.lower()
        and "dice" not in c.lower()
    ]
    s_cols = [
        c for c in df.columns
        if (
            c.lower() == "source_fg_fraction"
            or c.lower() == "source_boundary_density"
            or c.lower().startswith("struct_source_")
        )
        and "rur_" not in c.lower()
    ]

    u_cols = list(dict.fromkeys(u_cols))
    s_cols = list(dict.fromkeys(s_cols))
    return u_cols, s_cols


def matrix(df, u_cols, s_cols, method):
    U = df[u_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    S = df[s_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    if method == "uncertainty_only":
        return U
    if method == "structure_only":
        return S
    if method == "concat":
        return np.concatenate([U, S], axis=1)
    if method == "SGUC_fixed":
        gate = np.tanh(np.nanmean(S, axis=1, keepdims=True))
        return np.concatenate([U, S, U * gate], axis=1)

    raise ValueError(method)


def grouped_oof(X, y, groups, seed):
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    p = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Group leakage.")

        m = model(seed + fold)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]

    if np.isnan(p).any():
        raise AssertionError("Incomplete grouped OOF.")
    return p


def source_to_external_transfer(Xs, ys, gs, Xe, ye, ge, seed):
    """
    Same group split built on source cases. Fit source only, predict held-out
    source and paired external cases.
    """
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    ps = np.full(len(ys), np.nan)
    pe = np.full(len(ye), np.nan)

    for fold, (tr, te_s) in enumerate(cv.split(Xs, ys, gs)):
        train_cases = set(gs[tr])
        test_cases = set(gs[te_s])

        if train_cases & test_cases:
            raise AssertionError("Source group leakage.")

        te_e = np.flatnonzero(np.isin(ge, list(test_cases)))

        if set(ge[te_e]) != test_cases:
            raise AssertionError("External paired case set mismatch.")
        if set(ge[te_e]) & train_cases:
            raise AssertionError("External held-out cases leaked into source training.")

        m = model(seed + 100 + fold)
        m.fit(Xs[tr], ys[tr])

        ps[te_s] = m.predict_proba(Xs[te_s])[:, 1]
        pe[te_e] = m.predict_proba(Xe[te_e])[:, 1]

    if np.isnan(ps).any() or np.isnan(pe).any():
        raise AssertionError("Incomplete transfer predictions.")

    return ps, pe


def aucs(y, p):
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
    )


def univariate_direction_audit(src, ext, y_src, y_ext, cols):
    rows = []

    for c in cols:
        xs = pd.to_numeric(src[c], errors="coerce").to_numpy(float)
        xe = pd.to_numeric(ext[c], errors="coerce").to_numpy(float)

        # Median-impute only for univariate AUROC diagnostics.
        ms = np.nanmedian(xs)
        me = np.nanmedian(xe)
        xs = np.where(np.isfinite(xs), xs, ms)
        xe = np.where(np.isfinite(xe), xe, me)

        a_s = roc_auc_score(y_src, xs)
        a_e = roc_auc_score(y_ext, xe)

        dir_s = np.sign(a_s - 0.5)
        dir_e = np.sign(a_e - 0.5)

        rows.append({
            "feature": c,
            "source_univariate_AUROC": float(a_s),
            "external_univariate_AUROC": float(a_e),
            "source_distance_from_random": float(abs(a_s - 0.5)),
            "external_distance_from_random": float(abs(a_e - 0.5)),
            "direction_flip": bool(
                dir_s != 0 and dir_e != 0 and dir_s != dir_e
            ),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H4 STRUCTURE TRANSFER MECHANISM AUDIT FIX1 =====")
    print("PRIMARY QUESTION: does structure transfer better than uncertainty?")
    print("INDEPENDENT EXTERNAL DATASET CLAIM: NO")
    print("PRIMARY SOURCE->EXTERNAL FITTING USES EXTERNAL LABELS: NO")
    print("TARGET-SUPERVISED CEILING USES EXTERNAL LABELS: YES, DIAGNOSTIC ONLY")
    print("GROUP KEY: sample_id\n")

    src_path = Path(args.source)
    ext_path = Path(args.external)

    print("source exists:", src_path.exists(), "path:", src_path)
    print("external exists:", ext_path.exists(), "path:", ext_path)

    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if not ext_path.exists():
        raise FileNotFoundError(ext_path)

    src = pd.read_csv(src_path, low_memory=False)
    ext = pd.read_csv(ext_path, low_memory=False)

    src["_sid_norm"] = norm_sid(src["sample_id"])
    ext["_sid_norm"] = norm_sid(ext["sample_id"])

    if set(src["_sid_norm"]) != set(ext["_sid_norm"]):
        raise AssertionError("Case pools differ.")

    if not (src["_sid_norm"].value_counts() == 9).all():
        raise AssertionError("Source multiplicity != 9.")
    if not (ext["_sid_norm"].value_counts() == 9).all():
        raise AssertionError("External multiplicity != 9.")

    y_src = (
        src["outcome"].astype(str).str.upper().eq("HARM").astype(int).to_numpy()
    )
    y_ext = ext["harm_label"].astype(int).to_numpy()
    g_src = src["_sid_norm"].to_numpy()
    g_ext = ext["_sid_norm"].to_numpy()

    u_cols, s_cols = feature_panel(src)

    print("Source rows:", len(src))
    print("External rows:", len(ext))
    print("Cases:", src["_sid_norm"].nunique())
    print("Uncertainty features:", len(u_cols))
    print("Structure features:", len(s_cols))
    print("U:", u_cols)
    print("S:", s_cols)

    if len(u_cols) != 27:
        raise AssertionError(f"Expected 27 uncertainty features, got {len(u_cols)}.")
    if len(s_cols) != 4:
        raise AssertionError(f"Expected 4 structure features, got {len(s_cols)}.")

    missing = [c for c in u_cols + s_cols if c not in ext.columns]
    print("Missing external features:", missing)
    if missing:
        raise AssertionError("Frozen feature mismatch.")

    methods = [
        "uncertainty_only",
        "structure_only",
        "concat",
        "SGUC_fixed",
    ]

    Xs = {m: matrix(src, u_cols, s_cols, m) for m in methods}
    Xe = {m: matrix(ext, u_cols, s_cols, m) for m in methods}

    rows = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n######## SEED {seed} ########")

        for method in methods:
            ps, pe = source_to_external_transfer(
                Xs[method], y_src, g_src,
                Xe[method], y_ext, g_ext,
                seed,
            )

            src_auc, src_pr = aucs(y_src, ps)
            ext_transfer_auc, ext_transfer_pr = aucs(y_ext, pe)

            # Target-supervised grouped OOF ceiling: DIAGNOSTIC ONLY.
            p_target_oracle = grouped_oof(
                Xe[method], y_ext, g_ext, seed + 50000
            )
            ext_oracle_auc, ext_oracle_pr = aucs(y_ext, p_target_oracle)

            reversed_transfer_auc = 1.0 - ext_transfer_auc

            print(
                f"{method:18s} "
                f"SRC_AUROC={src_auc:.6f} "
                f"EXT_TRANSFER_AUROC={ext_transfer_auc:.6f} "
                f"EXT_TRANSFER_AUPRC={ext_transfer_pr:.6f} "
                f"EXT_REVERSED_AUROC={reversed_transfer_auc:.6f} "
                f"TARGET_SUPERVISED_CEILING_AUROC={ext_oracle_auc:.6f} "
                f"TARGET_SUPERVISED_CEILING_AUPRC={ext_oracle_pr:.6f}"
            )

            rows.append({
                "seed": seed,
                "method": method,
                "source_group_oof_AUROC": src_auc,
                "source_group_oof_AUPRC": src_pr,
                "external_sourcefit_AUROC": ext_transfer_auc,
                "external_sourcefit_AUPRC": ext_transfer_pr,
                "external_sourcefit_reversed_AUROC": reversed_transfer_auc,
                "external_targetsupervised_group_oof_AUROC": ext_oracle_auc,
                "external_targetsupervised_group_oof_AUPRC": ext_oracle_pr,
                "external_labels_used_for_primary_transfer_fit": "NO",
                "external_labels_used_for_ceiling_diagnostic": "YES",
            })

    metrics = pd.DataFrame(rows)

    summary = (
        metrics.groupby("method", sort=False)
        .agg(
            seeds=("seed", "count"),
            source_AUROC_mean=("source_group_oof_AUROC", "mean"),
            source_AUROC_std=("source_group_oof_AUROC", "std"),
            external_transfer_AUROC_mean=("external_sourcefit_AUROC", "mean"),
            external_transfer_AUROC_std=("external_sourcefit_AUROC", "std"),
            external_transfer_AUPRC_mean=("external_sourcefit_AUPRC", "mean"),
            external_transfer_AUPRC_std=("external_sourcefit_AUPRC", "std"),
            external_reversed_AUROC_mean=(
                "external_sourcefit_reversed_AUROC", "mean"
            ),
            targetsupervised_ceiling_AUROC_mean=(
                "external_targetsupervised_group_oof_AUROC", "mean"
            ),
            targetsupervised_ceiling_AUROC_std=(
                "external_targetsupervised_group_oof_AUROC", "std"
            ),
            targetsupervised_ceiling_AUPRC_mean=(
                "external_targetsupervised_group_oof_AUPRC", "mean"
            ),
            targetsupervised_ceiling_AUPRC_std=(
                "external_targetsupervised_group_oof_AUPRC", "std"
            ),
        )
        .reset_index()
    )

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    # Mechanistic feature-direction audit.
    uni = univariate_direction_audit(
        src, ext, y_src, y_ext, u_cols + s_cols
    )

    print("\n===== UNIVARIATE DIRECTION AUDIT =====")
    print(
        uni.sort_values(
            ["direction_flip", "source_distance_from_random"],
            ascending=[False, False],
        ).to_string(index=False)
    )
    print(
        "\nDirection flips:",
        int(uni["direction_flip"].sum()),
        "/",
        len(uni),
    )

    # Decision logic is descriptive, not a paper claim.
    st = summary[summary["method"] == "structure_only"].iloc[0]
    un = summary[summary["method"] == "uncertainty_only"].iloc[0]

    print("\n===== ROUTE DECISION AUDIT =====")
    print(
        "Structure transfer AUROC:",
        float(st["external_transfer_AUROC_mean"])
    )
    print(
        "Uncertainty transfer AUROC:",
        float(un["external_transfer_AUROC_mean"])
    )
    print(
        "Structure target-supervised ceiling:",
        float(st["targetsupervised_ceiling_AUROC_mean"])
    )
    print(
        "Uncertainty target-supervised ceiling:",
        float(un["targetsupervised_ceiling_AUROC_mean"])
    )

    if st["external_transfer_AUROC_mean"] >= 0.65:
        decision = "STRUCTURE_TRANSFER_BACKBONE_SUPPORTED"
    elif (
        st["targetsupervised_ceiling_AUROC_mean"] >= 0.70
        or un["targetsupervised_ceiling_AUROC_mean"] >= 0.70
    ):
        decision = "CONDITIONAL_RELATIONSHIP_SHIFT_TARGET_INFORMATION_REMAINS"
    else:
        decision = "EXTERNAL_REPRESENTATION_SAFETY_INFORMATION_WEAK"

    print("DECISION:", decision)

    metrics_path = out / "R10H4_per_seed_metrics.csv"
    summary_path = out / "R10H4_summary.csv"
    uni_path = out / "R10H4_univariate_direction_audit.csv"

    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    uni.to_csv(uni_path, index=False)

    print("\nOutputs:")
    print(metrics_path)
    print(summary_path)
    print(uni_path)
    print("PASS")


if __name__ == "__main__":
    main()
