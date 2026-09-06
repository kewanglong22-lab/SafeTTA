
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H5_structure_identity_forensic_audit_fix1.py

Forensic audit after R10H4.

Question
--------
R10H4 reported structure-only source->external AUROC numerically identical to
source AUROC across all seeds, while several individual structure features also
had exactly identical source/external univariate AUROC.

Before interpreting this as "cross-domain structure transfer", verify whether
the structure values themselves are literally preserved row-by-row.

This script is AUDIT ONLY:
- no model fitting;
- no threshold tuning;
- no row-order matching;
- no nearest-neighbour matching.

Strict row identity is attempted only through a GT-derived forensic key:
    normalized sample_id
    + model_family
    + canonical source_dice
    + canonical a1_dice

This key is used only to audit lineage and feature identity, never as a model
input and never in any deployable experiment.

Outputs
-------
1) row-key uniqueness / key-set equality
2) source-vs-external HARM-label agreement
3) exact/tolerant row-wise equality for every common numeric feature
4) redundancy audit for the 4 "structure" columns
5) a scientific decision about whether R10H4 structure transfer is genuine
   representation transfer or identity preservation.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


ATOL = 1e-14

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
    "Q1_R10H5_structure_identity_forensic_audit_fix1_v1"
)

STRUCTURE_COLS = [
    "source_fg_fraction",
    "source_boundary_density",
    "struct_source_fg_fraction",
    "struct_source_boundary_density",
]

RAW_STRUCTURE_COLS = [
    "source_fg_fraction",
    "source_boundary_density",
]

ALIAS_PAIRS = [
    ("source_fg_fraction", "struct_source_fg_fraction"),
    ("source_boundary_density", "struct_source_boundary_density"),
]


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def canon_float_series(s):
    x = pd.to_numeric(s, errors="coerce")
    out = []
    for v in x:
        if pd.isna(v):
            out.append("nan")
        else:
            out.append(format(float(v), ".15g"))
    return pd.Series(out, index=s.index, dtype="object")


def safe_auc(y, x):
    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.nan
    fill = np.nanmedian(x[finite])
    x = np.where(np.isfinite(x), x, fill)
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


def numeric_compare(a, b, atol):
    a = pd.to_numeric(a, errors="coerce").to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(float)

    both_nan = np.isnan(a) & np.isnan(b)
    both_finite = np.isfinite(a) & np.isfinite(b)
    one_nan = np.isnan(a) ^ np.isnan(b)

    exact = both_nan | (both_finite & (a == b))
    close = both_nan | (both_finite & (np.abs(a - b) <= atol))

    diff = np.full(len(a), np.nan)
    diff[both_finite] = np.abs(a[both_finite] - b[both_finite])
    diff[one_nan] = np.inf

    finite_diff = diff[np.isfinite(diff)]

    if np.sum(both_finite) >= 2:
        aa = a[both_finite]
        bb = b[both_finite]
        if np.std(aa) > 0 and np.std(bb) > 0:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        else:
            corr = np.nan
    else:
        corr = np.nan

    return {
        "exact_fraction": float(np.mean(exact)),
        "within_atol_fraction": float(np.mean(close)),
        "mean_abs_diff": (
            float(np.mean(finite_diff)) if len(finite_diff) else np.nan
        ),
        "median_abs_diff": (
            float(np.median(finite_diff)) if len(finite_diff) else np.nan
        ),
        "max_abs_diff": (
            float(np.max(finite_diff)) if len(finite_diff) else np.nan
        ),
        "pearson_rowwise": corr,
        "one_sided_nan_rows": int(np.sum(one_nan)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H5 STRUCTURE IDENTITY FORENSIC AUDIT FIX1 =====")
    print("MODEL FITTING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("FORENSIC MATCH ATOL:", ATOL)
    print(
        "FORENSIC KEY: sample_id + model_family + source_dice + a1_dice"
    )
    print(
        "GT-derived key is for lineage audit only; it is NOT a model feature.\n"
    )

    sp = Path(args.source)
    ep = Path(args.external)

    print("source exists:", sp.exists(), "path:", sp)
    print("external exists:", ep.exists(), "path:", ep)

    if not sp.exists():
        raise FileNotFoundError(sp)
    if not ep.exists():
        raise FileNotFoundError(ep)

    src = pd.read_csv(sp, low_memory=False)
    ext = pd.read_csv(ep, low_memory=False)

    required_common = [
        "sample_id",
        "model_family",
        "source_dice",
        "a1_dice",
    ]
    for c in required_common:
        if c not in src.columns:
            raise AssertionError(f"Source missing forensic key column: {c}")
        if c not in ext.columns:
            raise AssertionError(f"External missing forensic key column: {c}")

    if "outcome" not in src.columns:
        raise AssertionError("Source outcome missing.")
    if "harm_label" not in ext.columns:
        raise AssertionError("External harm_label missing.")

    src["_sid_norm"] = norm_sid(src["sample_id"])
    ext["_sid_norm"] = norm_sid(ext["sample_id"])

    src["_source_dice_key"] = canon_float_series(src["source_dice"])
    ext["_source_dice_key"] = canon_float_series(ext["source_dice"])
    src["_a1_dice_key"] = canon_float_series(src["a1_dice"])
    ext["_a1_dice_key"] = canon_float_series(ext["a1_dice"])

    key = [
        "_sid_norm",
        "model_family",
        "_source_dice_key",
        "_a1_dice_key",
    ]

    print("\n===== FORENSIC KEY AUDIT =====")
    print("Source rows:", len(src))
    print("External rows:", len(ext))
    print("Source unique forensic keys:", src[key].drop_duplicates().shape[0])
    print("External unique forensic keys:", ext[key].drop_duplicates().shape[0])
    print("Source duplicated-key rows:", int(src.duplicated(key, keep=False).sum()))
    print("External duplicated-key rows:", int(ext.duplicated(key, keep=False).sum()))

    src_keyset = set(map(tuple, src[key].astype(str).to_numpy()))
    ext_keyset = set(map(tuple, ext[key].astype(str).to_numpy()))

    print("Key intersection:", len(src_keyset & ext_keyset))
    print("Source-only keys:", len(src_keyset - ext_keyset))
    print("External-only keys:", len(ext_keyset - src_keyset))
    print("KEY_SET_EQUAL:", src_keyset == ext_keyset)

    if (
        src[key].drop_duplicates().shape[0] != len(src)
        or ext[key].drop_duplicates().shape[0] != len(ext)
        or src_keyset != ext_keyset
    ):
        print("\nDECISION: FORENSIC_ROW_ALIGNMENT_NOT_PROVEN")
        print(
            "Do not interpret the R10H4 equality until a unique explicit row "
            "identity is available."
        )
        print("PASS: AUDIT STOPPED SAFELY")
        return

    src_ren = src.copy()
    ext_ren = ext.copy()

    common_numeric = []
    for c in src.columns:
        if c in ext.columns and c not in key and c not in {
            "sample_id", "model_family", "outcome", "harm_label",
            "adaptation_outcome", "benefit_label",
            "checkpoint_sha256", "model_state_id", "training_seed",
            "_external_row_id", "max_abs_feature_diff",
        }:
            # Keep only columns that are numeric in at least one side.
            if (
                pd.api.types.is_numeric_dtype(src[c])
                or pd.api.types.is_numeric_dtype(ext[c])
            ):
                common_numeric.append(c)

    keep_s = key + ["outcome"] + common_numeric
    keep_e = key + ["harm_label"] + common_numeric

    aligned = src_ren[keep_s].merge(
        ext_ren[keep_e],
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("_src", "_ext"),
    )

    if len(aligned) != len(src):
        raise AssertionError(
            f"Aligned rows={len(aligned)}, expected {len(src)}."
        )

    y_src = (
        aligned["outcome"].astype(str).str.upper().eq("HARM").astype(int)
    ).to_numpy()
    y_ext = aligned["harm_label"].astype(int).to_numpy()

    label_agreement = float(np.mean(y_src == y_ext))
    print("\n===== LABEL AGREEMENT =====")
    print("HARM label agreement:", f"{label_agreement:.12f}")
    print("Disagreement rows:", int(np.sum(y_src != y_ext)))

    rows = []

    print("\n===== ROW-WISE FEATURE IDENTITY =====")
    for c in common_numeric:
        cs = c + "_src"
        ce = c + "_ext"

        stat = numeric_compare(aligned[cs], aligned[ce], ATOL)
        stat["feature"] = c
        stat["source_univariate_AUROC"] = safe_auc(y_src, aligned[cs])
        stat["external_univariate_AUROC"] = safe_auc(y_ext, aligned[ce])
        stat["univariate_AUROC_delta"] = (
            stat["external_univariate_AUROC"]
            - stat["source_univariate_AUROC"]
        )
        rows.append(stat)

    comp = pd.DataFrame(rows)

    show_cols = [
        "feature",
        "exact_fraction",
        "within_atol_fraction",
        "max_abs_diff",
        "pearson_rowwise",
        "source_univariate_AUROC",
        "external_univariate_AUROC",
        "univariate_AUROC_delta",
    ]

    print(
        comp.sort_values(
            ["within_atol_fraction", "max_abs_diff"],
            ascending=[False, True],
        )[show_cols].to_string(index=False)
    )

    # --------------------------------------------------------------
    # Structure alias redundancy within each table.
    # --------------------------------------------------------------
    print("\n===== STRUCTURE ALIAS REDUNDANCY =====")
    alias_rows = []
    for raw, alias in ALIAS_PAIRS:
        for domain_name, df in [("source", src), ("external", ext)]:
            if raw not in df.columns or alias not in df.columns:
                stat = {
                    "domain": domain_name,
                    "raw": raw,
                    "alias": alias,
                    "available": False,
                }
            else:
                z = numeric_compare(df[raw], df[alias], ATOL)
                stat = {
                    "domain": domain_name,
                    "raw": raw,
                    "alias": alias,
                    "available": True,
                    **z,
                }
            alias_rows.append(stat)

    alias_df = pd.DataFrame(alias_rows)
    print(alias_df.to_string(index=False))

    structure_comp = comp[comp["feature"].isin(STRUCTURE_COLS)].copy()
    raw_structure_comp = comp[
        comp["feature"].isin(RAW_STRUCTURE_COLS)
    ].copy()

    print("\n===== STRUCTURE SUMMARY =====")
    print(
        structure_comp[show_cols].to_string(index=False)
        if len(structure_comp) else "No structure columns found."
    )

    structure_all_preserved = (
        len(raw_structure_comp) == len(RAW_STRUCTURE_COLS)
        and (raw_structure_comp["within_atol_fraction"] == 1.0).all()
        and (raw_structure_comp["max_abs_diff"] <= ATOL).all()
    )

    aliases_redundant = True
    for raw, alias in ALIAS_PAIRS:
        for domain_name, df in [("source", src), ("external", ext)]:
            if raw not in df.columns or alias not in df.columns:
                aliases_redundant = False
                continue
            st = numeric_compare(df[raw], df[alias], ATOL)
            if not (
                st["within_atol_fraction"] == 1.0
                and st["max_abs_diff"] <= ATOL
            ):
                aliases_redundant = False

    exact_common_count = int(
        np.sum(
            (comp["within_atol_fraction"] == 1.0)
            & (comp["max_abs_diff"] <= ATOL)
        )
    )

    print("\nExact/tolerance-preserved common numeric features:",
          exact_common_count, "/", len(comp))
    print("Raw structure features preserved row-by-row:", structure_all_preserved)
    print("struct_* columns are aliases/redundant:", aliases_redundant)

    comp.to_csv(out / "R10H5_rowwise_feature_identity.csv", index=False)
    alias_df.to_csv(out / "R10H5_structure_alias_redundancy.csv", index=False)

    summary = pd.DataFrame([{
        "rows": len(aligned),
        "harm_label_agreement": label_agreement,
        "common_numeric_features": len(comp),
        "exact_preserved_common_numeric_features": exact_common_count,
        "raw_structure_features_preserved": structure_all_preserved,
        "structure_aliases_redundant": aliases_redundant,
        "forensic_atol": ATOL,
    }])
    summary.to_csv(out / "R10H5_summary.csv", index=False)

    print("\n===== FINAL SCIENTIFIC DECISION =====")
    if (
        label_agreement == 1.0
        and structure_all_preserved
        and aliases_redundant
    ):
        print(
            "DECISION: STRUCTURE_RESULT_IS_IDENTITY_PRESERVATION_NOT_"
            "INDEPENDENT_TRANSFER"
        )
        print(
            "The raw structure variables are numerically preserved row-by-row "
            "between the two representations, and the struct_* variables are "
            "duplicates of those raw variables."
        )
        print(
            "Therefore the identical structure-only AUROC in R10H4 is expected "
            "and cannot by itself support a claim of cross-domain structural "
            "generalization."
        )
        print(
            "VALID USE: structure features remain a useful stable baseline / "
            "mechanistic control within this paired-representation experiment."
        )
    elif structure_all_preserved:
        print(
            "DECISION: STRUCTURE_VALUES_PRESERVED_BUT_LABEL_OR_ALIAS_AUDIT_"
            "REQUIRES_REVIEW"
        )
    else:
        print("DECISION: STRUCTURE_TRANSFER_REMAINS_EMPIRICAL")
        print(
            "The structure values materially changed across representations; "
            "R10H4 therefore contains genuine transfer evidence."
        )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
