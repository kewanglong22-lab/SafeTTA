
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H5_structure_identity_multiset_audit_fix2.py

Fix2 for R10H5.

Why fix2 is needed
------------------
The original forensic key
    sample_id + model_family + source_dice + a1_dice
is not globally unique:
    8946 unique keys / 9000 rows, 93 duplicated-key rows.

Therefore row-wise pairing inside ambiguous groups is not justified.

This fix does NOT invent a row identity. Instead, for each forensic-key group,
it compares the MULTISET of feature values between source and external tables.
For a feature to be declared preserved, every key group must have:
    - identical group size
    - identical NaN pattern/count
    - sorted finite values equal within fixed atol=1e-14

This is sufficient to prove value preservation without arbitrary row matching.

AUDIT ONLY:
- no model fitting
- no threshold tuning
- no row-order matching
- no nearest-neighbour matching
- no label attachment
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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
    "Q1_R10H5_structure_identity_multiset_audit_fix2_v1"
)

STRUCTURE_RAW = [
    "source_fg_fraction",
    "source_boundary_density",
]

STRUCTURE_ALIAS = [
    "struct_source_fg_fraction",
    "struct_source_boundary_density",
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


def numeric_multiset_equal(a, b, atol):
    """
    Compare two 1-D numeric multisets without assigning row identities.
    """
    a = pd.to_numeric(a, errors="coerce").to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(float)

    if len(a) != len(b):
        return False, np.inf

    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if int(nan_a.sum()) != int(nan_b.sum()):
        return False, np.inf

    af = np.sort(a[~nan_a])
    bf = np.sort(b[~nan_b])

    if len(af) != len(bf):
        return False, np.inf

    if len(af) == 0:
        return True, 0.0

    diffs = np.abs(af - bf)
    max_diff = float(np.max(diffs))
    return bool(np.all(diffs <= atol)), max_diff


def vector_multiset_equal(A, B, atol):
    """
    Compare multiset of 2-D row vectors without assuming row identity.

    The groups in this project are very small. We canonical-sort rows
    lexicographically and then verify coordinate-wise equality within atol.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != B.shape:
        return False, np.inf

    # Replace NaN with +inf only for deterministic sorting; equality checks
    # below still explicitly require matching NaN masks.
    As = np.where(np.isnan(A), np.inf, A)
    Bs = np.where(np.isnan(B), np.inf, B)

    if A.shape[1] == 0:
        return True, 0.0

    # np.lexsort uses last key as primary.
    oa = np.lexsort(tuple(As[:, j] for j in range(A.shape[1] - 1, -1, -1)))
    ob = np.lexsort(tuple(Bs[:, j] for j in range(B.shape[1] - 1, -1, -1)))

    A2 = A[oa]
    B2 = B[ob]

    nan_match = np.isnan(A2) == np.isnan(B2)
    if not nan_match.all():
        return False, np.inf

    finite = np.isfinite(A2) & np.isfinite(B2)
    diff = np.zeros_like(A2, dtype=float)
    diff[finite] = np.abs(A2[finite] - B2[finite])

    max_diff = float(np.max(diff)) if diff.size else 0.0
    ok = bool(np.all(diff[finite] <= atol))
    return ok, max_diff


def alias_audit(df, domain):
    rows = []
    for raw, alias in ALIAS_PAIRS:
        if raw not in df.columns or alias not in df.columns:
            rows.append({
                "domain": domain,
                "raw": raw,
                "alias": alias,
                "available": False,
                "exact_fraction": np.nan,
                "within_atol_fraction": np.nan,
                "max_abs_diff": np.nan,
            })
            continue

        a = pd.to_numeric(df[raw], errors="coerce").to_numpy(float)
        b = pd.to_numeric(df[alias], errors="coerce").to_numpy(float)

        both_nan = np.isnan(a) & np.isnan(b)
        finite = np.isfinite(a) & np.isfinite(b)

        exact = both_nan | (finite & (a == b))
        close = both_nan | (finite & (np.abs(a - b) <= ATOL))

        diffs = np.abs(a[finite] - b[finite]) if finite.any() else np.array([])

        rows.append({
            "domain": domain,
            "raw": raw,
            "alias": alias,
            "available": True,
            "exact_fraction": float(np.mean(exact)),
            "within_atol_fraction": float(np.mean(close)),
            "max_abs_diff": float(np.max(diffs)) if len(diffs) else 0.0,
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

    print("===== R10H5 STRUCTURE IDENTITY MULTISET AUDIT FIX2 =====")
    print("MODEL FITTING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("ROW IDENTITY ASSIGNMENT INSIDE DUPLICATE KEYS: NONE")
    print("FORENSIC MULTISET ATOL:", ATOL)
    print("KEY: sample_id + model_family + source_dice + a1_dice\n")

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

    required = ["sample_id", "model_family", "source_dice", "a1_dice"]
    for c in required:
        if c not in src.columns:
            raise AssertionError(f"Source missing {c}")
        if c not in ext.columns:
            raise AssertionError(f"External missing {c}")

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

    sgroups = {k: g for k, g in src.groupby(key, dropna=False, sort=False)}
    egroups = {k: g for k, g in ext.groupby(key, dropna=False, sort=False)}

    skeys = set(sgroups)
    ekeys = set(egroups)

    print("Source rows:", len(src))
    print("External rows:", len(ext))
    print("Source groups:", len(sgroups))
    print("External groups:", len(egroups))
    print("Key-set equal:", skeys == ekeys)

    if skeys != ekeys:
        print("DECISION: MULTISET_AUDIT_NOT_POSSIBLE_KEY_MISMATCH")
        return

    group_size_mismatch = 0
    duplicate_groups = 0
    max_group_size = 0

    for k in skeys:
        ns = len(sgroups[k])
        ne = len(egroups[k])
        max_group_size = max(max_group_size, ns, ne)
        if ns != ne:
            group_size_mismatch += 1
        if ns > 1 or ne > 1:
            duplicate_groups += 1

    print("Group-size mismatches:", group_size_mismatch)
    print("Duplicate-key groups:", duplicate_groups)
    print("Maximum group size:", max_group_size)

    if group_size_mismatch:
        print("DECISION: MULTISET_AUDIT_NOT_POSSIBLE_GROUP_SIZE_MISMATCH")
        return

    # --------------------------------------------------------------
    # Label multiset agreement within each forensic group.
    # --------------------------------------------------------------
    if "outcome" not in src.columns or "harm_label" not in ext.columns:
        raise AssertionError("Required harm labels missing.")

    label_groups_equal = 0
    label_groups_total = len(skeys)

    for k in skeys:
        ys = (
            sgroups[k]["outcome"]
            .astype(str)
            .str.upper()
            .eq("HARM")
            .astype(int)
            .sort_values()
            .to_numpy()
        )
        ye = (
            egroups[k]["harm_label"]
            .astype(int)
            .sort_values()
            .to_numpy()
        )
        if np.array_equal(ys, ye):
            label_groups_equal += 1

    print("\n===== LABEL MULTISET AUDIT =====")
    print(
        "Groups with identical HARM-label multiset:",
        label_groups_equal,
        "/",
        label_groups_total,
    )

    # --------------------------------------------------------------
    # Per-feature multiset preservation across all groups.
    # --------------------------------------------------------------
    excluded = {
        "sample_id", "model_family", "outcome", "harm_label",
        "adaptation_outcome", "benefit_label",
        "checkpoint_sha256", "model_state_id", "training_seed",
        "_external_row_id", "max_abs_feature_diff",
        "_sid_norm", "_source_dice_key", "_a1_dice_key",
    }

    common_numeric = []
    for c in src.columns:
        if c in ext.columns and c not in excluded and c not in required:
            if (
                pd.api.types.is_numeric_dtype(src[c])
                or pd.api.types.is_numeric_dtype(ext[c])
            ):
                common_numeric.append(c)

    rows = []

    print("\n===== PER-FEATURE GROUP-MULTISET PRESERVATION =====")
    for c in common_numeric:
        equal_groups = 0
        max_diff = 0.0

        for k in skeys:
            ok, d = numeric_multiset_equal(
                sgroups[k][c],
                egroups[k][c],
                ATOL,
            )
            if ok:
                equal_groups += 1
            if np.isfinite(d):
                max_diff = max(max_diff, d)
            else:
                max_diff = np.inf

        frac = equal_groups / len(skeys)
        preserved = equal_groups == len(skeys)

        rows.append({
            "feature": c,
            "groups_total": len(skeys),
            "groups_equal_multiset": equal_groups,
            "group_preservation_fraction": frac,
            "max_group_sorted_abs_diff": max_diff,
            "preserved_all_groups": preserved,
        })

    feat = pd.DataFrame(rows)
    if len(feat):
        print(
            feat.sort_values(
                ["preserved_all_groups", "group_preservation_fraction", "feature"],
                ascending=[False, False, True],
            ).to_string(index=False)
        )

    # --------------------------------------------------------------
    # Joint RAW-structure vector multiset audit.
    # This is stronger than checking the two features separately.
    # --------------------------------------------------------------
    print("\n===== JOINT RAW-STRUCTURE VECTOR MULTISET AUDIT =====")
    missing_structure = [
        c for c in STRUCTURE_RAW
        if c not in src.columns or c not in ext.columns
    ]
    print("Missing raw structure columns:", missing_structure)

    joint_equal = 0
    joint_max_diff = 0.0

    if not missing_structure:
        for k in skeys:
            A = sgroups[k][STRUCTURE_RAW].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            B = egroups[k][STRUCTURE_RAW].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)

            ok, d = vector_multiset_equal(A, B, ATOL)
            if ok:
                joint_equal += 1
            if np.isfinite(d):
                joint_max_diff = max(joint_max_diff, d)
            else:
                joint_max_diff = np.inf

        print(
            "Groups with identical joint structure-vector multiset:",
            joint_equal,
            "/",
            len(skeys),
        )
        print("Maximum matched structure-vector diff:", joint_max_diff)

    structure_joint_preserved = (
        not missing_structure
        and joint_equal == len(skeys)
        and joint_max_diff <= ATOL
    )

    # --------------------------------------------------------------
    # Alias redundancy.
    # --------------------------------------------------------------
    print("\n===== STRUCTURE ALIAS REDUNDANCY =====")
    alias_df = pd.concat([
        alias_audit(src, "source"),
        alias_audit(ext, "external"),
    ], ignore_index=True)
    print(alias_df.to_string(index=False))

    aliases_redundant = bool(
        len(alias_df) == 4
        and alias_df["available"].all()
        and (alias_df["within_atol_fraction"] == 1.0).all()
        and (alias_df["max_abs_diff"] <= ATOL).all()
    )

    # Save.
    feat.to_csv(out / "R10H5_fix2_feature_multiset_preservation.csv", index=False)
    alias_df.to_csv(out / "R10H5_fix2_structure_alias_redundancy.csv", index=False)

    preserved_features = (
        feat.loc[feat["preserved_all_groups"], "feature"].tolist()
        if len(feat) else []
    )

    summary = pd.DataFrame([{
        "rows_source": len(src),
        "rows_external": len(ext),
        "forensic_groups": len(skeys),
        "duplicate_key_groups": duplicate_groups,
        "max_group_size": max_group_size,
        "label_multiset_groups_equal": label_groups_equal,
        "label_multiset_groups_total": label_groups_total,
        "common_numeric_features": len(feat),
        "features_preserved_all_groups": len(preserved_features),
        "joint_raw_structure_preserved": structure_joint_preserved,
        "structure_aliases_redundant": aliases_redundant,
        "atol": ATOL,
    }])
    summary.to_csv(out / "R10H5_fix2_summary.csv", index=False)

    print("\nPreserved common numeric features:", len(preserved_features), "/", len(feat))
    print("Preserved feature names:", preserved_features)
    print("Joint raw structure preserved:", structure_joint_preserved)
    print("struct_* aliases redundant:", aliases_redundant)

    print("\n===== FINAL SCIENTIFIC DECISION =====")
    if (
        label_groups_equal == label_groups_total
        and structure_joint_preserved
        and aliases_redundant
    ):
        print(
            "DECISION: STRUCTURE_IDENTITY_PRESERVATION_PROVEN_BY_MULTISET"
        )
        print(
            "The two raw structure descriptors are preserved as an identical "
            "within-key multiset across all forensic groups, including the "
            "ambiguous duplicate-key groups."
        )
        print(
            "The struct_* columns are redundant aliases of those same two "
            "raw descriptors."
        )
        print(
            "Therefore R10H4 structure-only equality is an identity-preservation "
            "control, NOT evidence of independent cross-domain structural "
            "generalization."
        )
    elif structure_joint_preserved:
        print(
            "DECISION: STRUCTURE_VALUES_PRESERVED_BUT_LABEL_OR_ALIAS_AUDIT_FAILED"
        )
    else:
        print("DECISION: STRUCTURE_IDENTITY_PRESERVATION_NOT_PROVEN")
        print(
            "R10H4 structure-only performance remains an empirical transfer "
            "result and requires further interpretation."
        )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
