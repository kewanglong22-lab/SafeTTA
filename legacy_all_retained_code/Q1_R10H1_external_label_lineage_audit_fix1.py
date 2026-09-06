
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_external_label_lineage_audit_fix1.py

Purpose
-------
Audit whether the frozen 9000-row NeoPolyp external invariant feature table
can be legitimately joined to post-GT segmentation harm labels already
generated in the R05D4/R08E1 lineage.

This script DOES NOT train a model and DOES NOT modify labels.
It only audits:
- file existence
- row/column counts
- sample_id normalization
- unique-ID overlap
- duplicate/multiplicity patterns
- label-column candidates and value counts
- whether a safe 1:1 sample_id merge is possible
- whether additional keys (model_family/model/state/etc.) are required

The script intentionally refuses to silently choose a label column or
silently collapse duplicates.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DEFAULT_EXTERNAL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10E1_external_complete_feature_v3/"
    "external_complete_invariant_feature_table_v2.csv"
)

DEFAULT_R05D4 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_R08E1 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R08E1_pre_neopolyp_feature_assembly_v1_fix1/"
    "neopolyp_source_feature_table.csv"
)

LABEL_HINTS = (
    "outcome", "harm", "benefit", "label", "target",
    "dice", "risk", "failure", "fail"
)

KEY_HINTS = (
    "sample_id", "model_family", "model", "state",
    "adaptation", "method", "policy", "seed", "fold", "case"
)


def norm_id(s):
    return (
        s.astype(str)
         .str.strip()
         .str.replace("\\\\", "/", regex=False)
         .str.lower()
    )


def audit_table(name, path):
    p = Path(path)
    print(f"\n===== {name} =====")
    print("Path:", p)
    print("Exists:", p.exists())
    if not p.exists():
        return None

    df = pd.read_csv(p, low_memory=False)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:")
    for c in df.columns:
        print("  ", c)

    if "sample_id" not in df.columns:
        print("sample_id: MISSING")
    else:
        ids = norm_id(df["sample_id"])
        print("sample_id non-null:", int(df["sample_id"].notna().sum()))
        print("sample_id unique:", int(ids.nunique(dropna=True)))
        print("sample_id duplicated rows:", int(ids.duplicated(keep=False).sum()))
        vc = ids.value_counts(dropna=False)
        print("sample_id max multiplicity:", int(vc.max()) if len(vc) else 0)
        print("sample_id multiplicity distribution:")
        print(vc.value_counts().sort_index().to_string())

    label_cols = [
        c for c in df.columns
        if any(h in c.lower() for h in LABEL_HINTS)
    ]
    key_cols = [
        c for c in df.columns
        if any(h in c.lower() for h in KEY_HINTS)
    ]

    print("Candidate label columns:", label_cols)
    print("Candidate key columns:", key_cols)

    for c in label_cols:
        print(f"\n[{name}] VALUE COUNTS: {c}")
        vc = df[c].value_counts(dropna=False).head(30)
        print(vc.to_string())
        if pd.api.types.is_numeric_dtype(df[c]):
            x = pd.to_numeric(df[c], errors="coerce")
            print(
                "numeric summary:",
                {
                    "non_null": int(x.notna().sum()),
                    "min": float(x.min()) if x.notna().any() else None,
                    "median": float(x.median()) if x.notna().any() else None,
                    "mean": float(x.mean()) if x.notna().any() else None,
                    "max": float(x.max()) if x.notna().any() else None,
                }
            )

    return df


def compare_ids(ext_name, ext, lab_name, lab):
    print(f"\n===== ID OVERLAP: {ext_name} vs {lab_name} =====")
    if ext is None or lab is None:
        print("SKIP: missing table")
        return
    if "sample_id" not in ext.columns or "sample_id" not in lab.columns:
        print("SKIP: sample_id missing")
        return

    e = set(norm_id(ext["sample_id"]).dropna())
    l = set(norm_id(lab["sample_id"]).dropna())
    inter = e & l

    print("External unique IDs:", len(e))
    print("Label unique IDs:", len(l))
    print("Intersection:", len(inter))
    print("External coverage by label IDs:", f"{len(inter)/len(e):.6f}" if e else "NA")
    print("Label coverage by external IDs:", f"{len(inter)/len(l):.6f}" if l else "NA")
    print("External-only IDs:", len(e - l))
    print("Label-only IDs:", len(l - e))

    if e - l:
        print("Example external-only IDs:", sorted(list(e - l))[:10])
    if l - e:
        print("Example label-only IDs:", sorted(list(l - e))[:10])

    # Multiplicity audit restricted to intersection.
    ee = ext.copy()
    ll = lab.copy()
    ee["_sid_norm"] = norm_id(ee["sample_id"])
    ll["_sid_norm"] = norm_id(ll["sample_id"])

    ee = ee[ee["_sid_norm"].isin(inter)]
    ll = ll[ll["_sid_norm"].isin(inter)]

    e_mult = ee["_sid_norm"].value_counts()
    l_mult = ll["_sid_norm"].value_counts()

    mult = pd.DataFrame({
        "external_rows_per_id": e_mult,
        "label_rows_per_id": l_mult
    }).fillna(0).astype(int)

    print("\nMultiplicity pair counts:")
    print(
        mult.value_counts(
            ["external_rows_per_id", "label_rows_per_id"]
        ).sort_index().to_string()
    )

    safe_1to1 = (
        len(inter) == len(e)
        and (mult["external_rows_per_id"] == 1).all()
        and (mult["label_rows_per_id"] == 1).all()
    )

    print("\nSAFE_1_TO_1_SAMPLE_ID_MERGE:", "YES" if safe_1to1 else "NO")

    if not safe_1to1:
        common_extra = [
            c for c in ext.columns
            if c in lab.columns and c != "sample_id"
        ]
        likely_keys = [
            c for c in common_extra
            if any(h in c.lower() for h in KEY_HINTS)
        ]
        print("Common non-sample_id columns:", common_extra)
        print("Likely additional merge keys:", likely_keys)
        print(
            "Decision: DO NOT MERGE LABELS YET. "
            "Audit additional keys / row lineage first."
        )
    else:
        print(
            "Decision: sample_id-level join is structurally eligible, "
            "but label semantics must still be verified before external evaluation."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--r05d4", default=DEFAULT_R05D4)
    ap.add_argument("--r08e1", default=DEFAULT_R08E1)
    ap.add_argument(
        "--output_dir",
        default=(
            "F:/MEDSEG_SAFETTA/outputs/"
            "Q1_R10H1_external_label_lineage_audit_fix1_v1"
        )
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 EXTERNAL LABEL LINEAGE AUDIT FIX1 =====")
    print("TRAINING: NONE")
    print("LABEL MODIFICATION: NONE")
    print("SILENT DUPLICATE COLLAPSE: NONE")
    print("SILENT LABEL SELECTION: NONE")

    ext = audit_table("EXTERNAL_FEATURES", args.external)
    r05 = audit_table("R05D4_POST_GT_OUTCOMES", args.r05d4)
    r08 = audit_table("R08E1_FEATURE_ASSEMBLY", args.r08e1)

    compare_ids("EXTERNAL_FEATURES", ext, "R05D4_POST_GT_OUTCOMES", r05)
    compare_ids("EXTERNAL_FEATURES", ext, "R08E1_FEATURE_ASSEMBLY", r08)

    # Save compact schema summaries only; raw data are never rewritten.
    rows = []
    for name, path, df in [
        ("external", args.external, ext),
        ("r05d4", args.r05d4, r05),
        ("r08e1", args.r08e1, r08),
    ]:
        rows.append({
            "table": name,
            "path": path,
            "exists": df is not None,
            "rows": len(df) if df is not None else np.nan,
            "columns": len(df.columns) if df is not None else np.nan,
            "unique_sample_id": (
                norm_id(df["sample_id"]).nunique()
                if df is not None and "sample_id" in df.columns
                else np.nan
            ),
            "duplicated_sample_rows": (
                norm_id(df["sample_id"]).duplicated(keep=False).sum()
                if df is not None and "sample_id" in df.columns
                else np.nan
            ),
        })

    pd.DataFrame(rows).to_csv(
        out / "external_label_lineage_table_summary.csv",
        index=False
    )

    print("\nOutput:", out)
    print("PASS: AUDIT COMPLETED")


if __name__ == "__main__":
    main()
