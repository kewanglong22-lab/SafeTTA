
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_R05D2_state_bridge_audit_fix1.py

Purpose
-------
Test whether the current NeoPolyp external feature lineage can be linked
EXACTLY to R05D2, which already preserves:
    sample_id + model_state_id + model_family + training_seed + checkpoint_sha256

Key hypothesis
--------------
R05D2 contains BOTH:
1) raw source_* features
2) mrz__source_* features

The later R10A2_7 / R10A2_8 / R10E1 external tables use source_* column names,
but they may actually correspond to the R05D2 MRZ representation with the
"mrz__" prefix removed. If true, the previous a23 -> a27 fingerprint mismatch
does NOT mean row identity was lost; it means the representation changed.

This audit tests exact 1:1 row identity against BOTH R05D2 representations:
- RAW: source_*
- MRZ: mrz__source_* mapped to source_* names

No row-order matching.
No nearest-neighbour matching.
No label attachment.
"""

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


RAW_FEATURES = [
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

DEFAULTS = {
    "r05d2": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1/"
        "target_paot_probabilities.csv"
    ),
    "a23": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_3_external_feature_lock_v1/"
        "external_feature_lock.csv"
    ),
    "a27": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_7_external_representation_lock_v1/"
        "neopolyp_external_R10A_feature_lock.csv"
    ),
    "a28": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_8_external_rur_transform_v1/"
        "neopolyp_external_invariant_feature_table.csv"
    ),
    "e1v3": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10E1_external_complete_feature_v3/"
        "external_complete_invariant_feature_table_v2.csv"
    ),
}


def norm_sid(x):
    return str(x).strip().replace("\\", "/").lower()


def canon(v):
    if pd.isna(v):
        return "nan"
    try:
        f = float(v)
        if np.isfinite(f):
            return format(f, ".15g")
    except Exception:
        pass
    return str(v).strip().lower()


def fp_from_named_values(df, colmap):
    """
    colmap: list[(logical_name, actual_column)]
    Fingerprint uses logical names implicitly by fixed order and actual values.
    """
    out = []
    for _, row in df.iterrows():
        parts = [norm_sid(row["sample_id"])]
        parts.extend(canon(row[actual]) for _, actual in colmap)
        out.append(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest())
    return pd.Series(out, index=df.index)


def compare_exact(target_name, target_df, r05, mode):
    print(f"\n===== {target_name} vs R05D2 [{mode}] =====")

    if mode == "RAW":
        r05_cols = [(c, c) for c in RAW_FEATURES]
    elif mode == "MRZ":
        r05_cols = [(c, "mrz__" + c) for c in RAW_FEATURES]
    else:
        raise ValueError(mode)

    target_missing = [c for c in RAW_FEATURES if c not in target_df.columns]
    r05_missing = [actual for _, actual in r05_cols if actual not in r05.columns]

    print("Target missing:", target_missing)
    print("R05D2 missing:", r05_missing)
    if target_missing or r05_missing:
        print("EXACT_1_TO_1:", "NOT_TESTABLE")
        return {"target": target_name, "mode": mode, "status": "NOT_TESTABLE"}

    target_cols = [(c, c) for c in RAW_FEATURES]

    ft = fp_from_named_values(target_df, target_cols)
    fr = fp_from_named_values(r05, r05_cols)

    st = set(ft)
    sr = set(fr)

    print("Rows:", len(target_df), len(r05))
    print("Unique fingerprints:", ft.nunique(), fr.nunique())
    print("Intersection:", len(st & sr))
    print("Target-only:", len(st - sr))
    print("R05D2-only:", len(sr - st))
    print("Target duplicated fingerprint rows:", int(ft.duplicated(keep=False).sum()))
    print("R05D2 duplicated fingerprint rows:", int(fr.duplicated(keep=False).sum()))

    exact = (
        len(target_df) == len(r05)
        and ft.nunique() == len(target_df)
        and fr.nunique() == len(r05)
        and st == sr
    )

    print("EXACT_1_TO_1:", "YES" if exact else "NO")

    result = {
        "target": target_name,
        "mode": mode,
        "status": "YES" if exact else "NO",
        "rows_target": len(target_df),
        "rows_r05d2": len(r05),
        "unique_target": int(ft.nunique()),
        "unique_r05d2": int(fr.nunique()),
        "intersection": len(st & sr),
        "target_only": len(st - sr),
        "r05d2_only": len(sr - st),
    }

    if exact:
        # Strict fingerprint-based identity recovery; never row order.
        map_df = pd.DataFrame({
            "_fp": fr,
            "sample_id_r05d2": r05["sample_id"].astype(str),
            "model_family": r05["model_family"].astype(str),
            "model_state_id": r05["model_state_id"].astype(str),
            "training_seed": r05["training_seed"],
            "checkpoint_sha256": r05["checkpoint_sha256"].astype(str),
        })

        if map_df["_fp"].duplicated().any():
            raise AssertionError("R05D2 fingerprint unexpectedly non-unique.")

        target_map = pd.DataFrame({
            "_fp": ft,
            "sample_id_target": target_df["sample_id"].astype(str),
        }).merge(
            map_df,
            on="_fp",
            how="left",
            validate="one_to_one",
        )

        if target_map["model_state_id"].isna().any():
            raise AssertionError("Exact fingerprint set equality but missing mapped state.")

        sid_ok = (
            target_map["sample_id_target"].map(norm_sid)
            == target_map["sample_id_r05d2"].map(norm_sid)
        ).all()

        print("Recovered state rows:", len(target_map))
        print("Recovered model_state_id missing:", int(target_map["model_state_id"].isna().sum()))
        print("sample_id agreement:", bool(sid_ok))
        print("Unique sample_id+model_state_id:",
              target_map[["sample_id_target", "model_state_id"]].drop_duplicates().shape[0])

        if not sid_ok:
            raise AssertionError("Fingerprint bridge produced sample_id mismatch.")

        result["recovered_state_rows"] = len(target_map)
        result["sample_id_agreement"] = bool(sid_ok)

        return result, target_map

    return result, None


def feature_difference_summary(target_name, target, r05):
    """
    Pure diagnostic: compare feature distributions, not row recovery.
    Helps distinguish RAW vs MRZ scale if exact mapping is not found.
    """
    print(f"\n===== DISTRIBUTION DIAGNOSTIC: {target_name} =====")
    rows = []
    for c in RAW_FEATURES:
        if c not in target.columns:
            continue
        for mode, rc in [("RAW", c), ("MRZ", "mrz__" + c)]:
            if rc not in r05.columns:
                continue
            a = pd.to_numeric(target[c], errors="coerce")
            b = pd.to_numeric(r05[rc], errors="coerce")
            rows.append({
                "feature": c,
                "mode": mode,
                "target_mean": float(a.mean()),
                "r05_mean": float(b.mean()),
                "abs_mean_diff": float(abs(a.mean() - b.mean())),
                "target_median": float(a.median()),
                "r05_median": float(b.median()),
                "abs_median_diff": float(abs(a.median() - b.median())),
            })

    d = pd.DataFrame(rows)
    if len(d):
        agg = d.groupby("mode")[["abs_mean_diff", "abs_median_diff"]].mean()
        print(agg.to_string())
    return d


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    ap.add_argument(
        "--output_dir",
        default=(
            "F:/MEDSEG_SAFETTA/outputs/"
            "Q1_R10H1_R05D2_state_bridge_audit_fix1_v1"
        ),
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 R05D2 STATE-BRIDGE AUDIT FIX1 =====")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("LABEL ATTACHMENT: NONE")
    print("TESTED REPRESENTATIONS: R05D2 RAW and R05D2 MRZ\n")

    dfs = {}
    for k in DEFAULTS:
        p = Path(getattr(args, k))
        print(f"{k}: exists={p.exists()}  path={p}")
        if not p.exists():
            raise FileNotFoundError(p)
        dfs[k] = pd.read_csv(p, low_memory=False)

    r05 = dfs["r05d2"]

    # Sanity: R05D2 has explicit unique state key.
    state_key_n = r05[["sample_id", "model_state_id"]].drop_duplicates().shape[0]
    print("\nR05D2 rows:", len(r05))
    print("Unique sample_id+model_state_id:", state_key_n)
    if state_key_n != len(r05):
        raise AssertionError("R05D2 explicit state key is not unique.")

    summaries = []
    mappings = {}

    for target_name in ["a23", "a27", "a28", "e1v3"]:
        target = dfs[target_name]

        for mode in ["RAW", "MRZ"]:
            result = compare_exact(target_name, target, r05, mode)
            if isinstance(result, tuple):
                summary, mapping = result
            else:
                summary, mapping = result, None

            summaries.append(summary)

            if mapping is not None:
                mappings[(target_name, mode)] = mapping
                mapping.to_csv(
                    out / f"{target_name}_to_R05D2_{mode}_exact_state_mapping.csv",
                    index=False,
                )

        diag = feature_difference_summary(target_name, target, r05)
        diag.to_csv(
            out / f"{target_name}_RAW_vs_MRZ_distribution_diagnostic.csv",
            index=False,
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "R05D2_state_bridge_summary.csv", index=False)

    print("\n===== SUMMARY =====")
    print(summary_df.to_string(index=False))

    print("\n===== FINAL DECISION =====")
    current_exact = summary_df[
        (summary_df["target"] == "e1v3")
        & (summary_df["status"] == "YES")
    ]

    if len(current_exact):
        mode = current_exact.iloc[0]["mode"]
        print("DECISION: CURRENT_EXTERNAL_STATE_IDENTITY_RECOVERABLE")
        print("Current external exact R05D2 representation:", mode)
        print(
            "The current 9000-row external table can be assigned model_state_id "
            "by exact value fingerprint against R05D2, without row-order assumptions."
        )
        print(
            "NEXT: cross-check the recovered sample_id+model_state_id keys against "
            "R05D4 and build a frozen evaluation-only labeled panel."
        )
    else:
        upstream_exact = summary_df[
            (summary_df["status"] == "YES")
            & (summary_df["target"].isin(["a23", "a27", "a28"]))
        ]

        if len(upstream_exact):
            print("DECISION: UPSTREAM_STATE_IDENTITY_RECOVERABLE_REBUILD_REQUIRED")
            print("Exact upstream bridges:")
            print(
                upstream_exact[["target", "mode", "intersection"]]
                .to_string(index=False)
            )
            print(
                "The current external table is not directly recoverable, but an "
                "upstream stage is. Rebuild downstream features from the exact "
                "state-identified upstream table while retaining model_state_id."
            )
        else:
            print("DECISION: R05D2_STATE_BRIDGE_NOT_FOUND")
            print(
                "Neither RAW nor MRZ R05D2 representations exactly identify any "
                "tested R10 external table. Do not attach R05D4 labels."
            )
            print(
                "NEXT: audit the script that generated R10A2_7 to determine its "
                "true upstream input and regenerate it with explicit state keys."
            )

    print("\nOutput:", out)
    print("PASS: STATE-BRIDGE AUDIT COMPLETED")


if __name__ == "__main__":
    main()
