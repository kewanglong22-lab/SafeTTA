
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_external_row_lineage_recovery_audit_fix1.py

Goal
----
Recover and VERIFY the missing row-level identity of the 9000-row NeoPolyp
external invariant-feature table without relying on row order.

Key idea
--------
Both:
  (A) external_complete_invariant_feature_table_v2.csv
and
  (B) neopolyp_source_feature_table.csv
contain the same raw pre-adaptation feature family.

For each row we build a deterministic fingerprint from:
  sample_id + all shared raw feature values.

Only if every external row maps uniquely 1:1 to exactly one R08E1 row do we
declare the lineage recoverable. No nearest-neighbour matching, row-order
matching, duplicate collapsing, or silent fallback is allowed.

Then the recovered R08E1 label metadata are cross-checked against R05D4
post-GT outcomes where possible.

This is an AUDIT script. It does NOT train/evaluate SGUC.
"""

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


DEFAULT_EXTERNAL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10E1_external_complete_feature_v3/"
    "external_complete_invariant_feature_table_v2.csv"
)

DEFAULT_R08E1 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R08E1_pre_neopolyp_feature_assembly_v1_fix1/"
    "neopolyp_source_feature_table.csv"
)

DEFAULT_R05D4 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H1_external_row_lineage_recovery_audit_fix1_v1"
)

RAW_FEATURES_EXPECTED = [
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


def norm_sid(x):
    return str(x).strip().replace("\\", "/").lower()


def canonical_num(x):
    """Stable representation for CSV floats; no approximate matching."""
    if pd.isna(x):
        return "nan"
    return format(float(x), ".15g")


def make_fingerprint(row, feature_cols):
    parts = [norm_sid(row["sample_id"])]
    parts.extend(canonical_num(row[c]) for c in feature_cols)
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def print_uniqueness(df, name, cols):
    if not all(c in df.columns for c in cols):
        print(f"{name} key {cols}: MISSING COLUMN")
        return None
    n_unique = df[cols].astype(str).drop_duplicates().shape[0]
    dup_rows = int(df.duplicated(cols, keep=False).sum())
    print(
        f"{name} key={cols}: unique={n_unique}/{len(df)} "
        f"duplicated_rows={dup_rows}"
    )
    return n_unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--r08e1", default=DEFAULT_R08E1)
    ap.add_argument("--r05d4", default=DEFAULT_R05D4)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 EXTERNAL ROW-LINEAGE RECOVERY AUDIT FIX1 =====")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("SILENT DUPLICATE COLLAPSE: FORBIDDEN")
    print("TRAINING: NONE")
    print("EVALUATION: NONE\n")

    ext_path = Path(args.external)
    r08_path = Path(args.r08e1)
    r05_path = Path(args.r05d4)

    for name, p in [
        ("external", ext_path),
        ("R08E1", r08_path),
        ("R05D4", r05_path),
    ]:
        print(f"{name} exists: {p.exists()}  path={p}")
        if not p.exists():
            raise FileNotFoundError(p)

    ext = pd.read_csv(ext_path, low_memory=False)
    r08 = pd.read_csv(r08_path, low_memory=False)
    r05 = pd.read_csv(r05_path, low_memory=False)

    print("\nRows:")
    print("  external:", len(ext))
    print("  R08E1:", len(r08))
    print("  R05D4:", len(r05))

    # ------------------------------------------------------------------
    # 1. Exact feature-lineage audit
    # ------------------------------------------------------------------
    shared = [
        c for c in RAW_FEATURES_EXPECTED
        if c in ext.columns and c in r08.columns
    ]
    missing_ext = [c for c in RAW_FEATURES_EXPECTED if c not in ext.columns]
    missing_r08 = [c for c in RAW_FEATURES_EXPECTED if c not in r08.columns]

    print("\n===== SHARED RAW FEATURE PANEL =====")
    print("Expected:", len(RAW_FEATURES_EXPECTED))
    print("Shared:", len(shared))
    print("Missing external:", missing_ext)
    print("Missing R08E1:", missing_r08)
    print("Shared columns:")
    for c in shared:
        print("  ", c)

    if len(shared) != len(RAW_FEATURES_EXPECTED):
        raise AssertionError(
            "Exact raw feature panel is incomplete; lineage recovery aborted."
        )

    # Ensure numeric equality can be meaningfully checked.
    max_abs_diff_report = {}
    for c in shared:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
        r08[c] = pd.to_numeric(r08[c], errors="coerce")

    print("\nBuilding deterministic SHA256 fingerprints...")
    ext["_feature_fingerprint"] = ext.apply(
        lambda r: make_fingerprint(r, shared), axis=1
    )
    r08["_feature_fingerprint"] = r08.apply(
        lambda r: make_fingerprint(r, shared), axis=1
    )

    ext_fp_counts = ext["_feature_fingerprint"].value_counts()
    r08_fp_counts = r08["_feature_fingerprint"].value_counts()

    print("External unique fingerprints:", ext_fp_counts.size)
    print("R08E1 unique fingerprints:", r08_fp_counts.size)
    print(
        "External duplicated fingerprint rows:",
        int(ext["_feature_fingerprint"].duplicated(keep=False).sum())
    )
    print(
        "R08E1 duplicated fingerprint rows:",
        int(r08["_feature_fingerprint"].duplicated(keep=False).sum())
    )

    ext_set = set(ext["_feature_fingerprint"])
    r08_set = set(r08["_feature_fingerprint"])
    inter = ext_set & r08_set

    print("Fingerprint intersection:", len(inter))
    print("External-only fingerprints:", len(ext_set - r08_set))
    print("R08E1-only fingerprints:", len(r08_set - ext_set))

    exact_unique_1to1 = (
        len(ext) == len(r08)
        and len(ext_set) == len(ext)
        and len(r08_set) == len(r08)
        and ext_set == r08_set
    )

    print(
        "\nEXACT_UNIQUE_FEATURE_FINGERPRINT_1_TO_1:",
        "YES" if exact_unique_1to1 else "NO"
    )

    # ------------------------------------------------------------------
    # 2. If exact fingerprints fail, audit whether the failure is merely
    #    numeric serialization by comparing within sample_id, but DO NOT
    #    use approximate matching to recover labels.
    # ------------------------------------------------------------------
    if not exact_unique_1to1:
        print("\n===== FAILURE DIAGNOSTIC =====")
        print(
            "Exact row recovery is NOT authorized because at least one "
            "fingerprint is missing or duplicated."
        )

        common_sids = sorted(
            set(ext["sample_id"].map(norm_sid))
            & set(r08["sample_id"].map(norm_sid))
        )

        # Compact nearest-distance diagnostics only.
        diagnostics = []
        for sid in common_sids[:1000]:
            e = ext[ext["sample_id"].map(norm_sid) == sid][shared].to_numpy(float)
            b = r08[r08["sample_id"].map(norm_sid) == sid][shared].to_numpy(float)
            if len(e) == 0 or len(b) == 0:
                continue
            # Pairwise max-absolute feature difference.
            D = np.max(np.abs(e[:, None, :] - b[None, :, :]), axis=2)
            diagnostics.append(float(np.nanmin(D)))

        if diagnostics:
            print(
                "Minimum within-sample row-distance diagnostic "
                "(max-abs over features):"
            )
            print("  median:", float(np.nanmedian(diagnostics)))
            print("  max:", float(np.nanmax(diagnostics)))

        print("\nDECISION: STOP_ROW_LINEAGE_RECOVERY")
        print(
            "Do not attach outcomes to the external table until an explicit "
            "row key or exact fingerprint lineage is available."
        )

        pd.DataFrame({
            "external_only_fingerprint": list(ext_set - r08_set)
        }).to_csv(out / "external_only_fingerprints.csv", index=False)
        pd.DataFrame({
            "r08e1_only_fingerprint": list(r08_set - ext_set)
        }).to_csv(out / "r08e1_only_fingerprints.csv", index=False)

        print("Output:", out)
        return

    # ------------------------------------------------------------------
    # 3. Exact deterministic recovery from R08E1
    # ------------------------------------------------------------------
    print("\n===== EXACT ROW RECOVERY =====")

    r08_meta_cols = [
        c for c in [
            "_feature_fingerprint",
            "sample_id",
            "model_family",
            "source_dice",
            "a1_dice",
            "outcome",
        ]
        if c in r08.columns
    ]

    recovered = ext.merge(
        r08[r08_meta_cols],
        on="_feature_fingerprint",
        how="left",
        validate="one_to_one",
        suffixes=("", "_r08"),
    )

    print("Recovered rows:", len(recovered))
    print("Missing outcome:", int(recovered["outcome"].isna().sum()))
    print("Missing model_family:", int(recovered["model_family"].isna().sum()))
    print("Outcome counts:")
    print(recovered["outcome"].value_counts(dropna=False).to_string())
    print("Model-family counts:")
    print(recovered["model_family"].value_counts(dropna=False).to_string())

    sid_agree = (
        recovered["sample_id"].map(norm_sid)
        == recovered["sample_id_r08"].map(norm_sid)
    )
    print("sample_id agreement:", int(sid_agree.sum()), "/", len(recovered))
    if not sid_agree.all():
        raise AssertionError("Fingerprint recovered mismatched sample_id.")

    # ------------------------------------------------------------------
    # 4. Independent R05D4 label cross-check.
    #    Prefer exact GT-value key if sample_id+model_family is non-unique.
    # ------------------------------------------------------------------
    print("\n===== R05D4 CROSS-CHECK =====")
    print_uniqueness(recovered, "Recovered", ["sample_id", "model_family"])
    print_uniqueness(r05, "R05D4", ["sample_id", "model_family"])

    # Use sample_id + model_family + source_dice + a1_dice as cross-check key.
    cross_cols = ["sample_id", "model_family", "source_dice", "a1_dice"]
    can_cross = all(c in recovered.columns for c in cross_cols) and all(
        c in r05.columns for c in cross_cols
    )

    if not can_cross:
        print("R05D4 exact GT-value cross-check: UNAVAILABLE")
        cross_status = "UNAVAILABLE"
    else:
        rec_check = recovered.copy()
        r05_check = r05.copy()

        rec_check["_sid"] = rec_check["sample_id"].map(norm_sid)
        r05_check["_sid"] = r05_check["sample_id"].map(norm_sid)

        for c in ["source_dice", "a1_dice"]:
            rec_check["_k_" + c] = rec_check[c].map(canonical_num)
            r05_check["_k_" + c] = r05_check[c].map(canonical_num)

        kcols = ["_sid", "model_family", "_k_source_dice", "_k_a1_dice"]

        print_uniqueness(rec_check, "Recovered cross-key", kcols)
        print_uniqueness(r05_check, "R05D4 cross-key", kcols)

        r05_small = r05_check[
            kcols + [
                "adaptation_outcome",
                "harm_label",
                "benefit_label",
            ]
        ].copy()

        # Only execute strict merge when R05 key is unique.
        if r05_small.duplicated(kcols).any():
            print("R05D4 exact GT-value cross-check: NON_UNIQUE_KEY")
            cross_status = "NON_UNIQUE_KEY"
        else:
            checked = rec_check.merge(
                r05_small,
                on=kcols,
                how="left",
                validate="many_to_one",
            )

            matched = checked["adaptation_outcome"].notna()
            print("R05D4 matched rows:", int(matched.sum()), "/", len(checked))

            outcome_agree = (
                checked.loc[matched, "outcome"].astype(str).str.upper()
                == checked.loc[matched, "adaptation_outcome"]
                    .astype(str).str.upper()
            )
            print(
                "Outcome agreement on matched rows:",
                int(outcome_agree.sum()), "/", int(matched.sum())
            )

            harm_from_outcome = (
                checked.loc[matched, "outcome"]
                .astype(str).str.upper().eq("HARM").astype(int)
            )
            harm_agree = (
                harm_from_outcome.to_numpy()
                == checked.loc[matched, "harm_label"].astype(int).to_numpy()
            )
            print(
                "Harm-label agreement on matched rows:",
                int(harm_agree.sum()), "/", int(matched.sum())
            )

            full_match = int(matched.sum()) == len(checked)
            full_outcome_agree = bool(outcome_agree.all()) if len(outcome_agree) else False
            full_harm_agree = bool(np.all(harm_agree)) if len(harm_agree) else False

            if full_match and full_outcome_agree and full_harm_agree:
                cross_status = "PASS"
            else:
                cross_status = "FAIL"

            checked[
                [
                    "sample_id",
                    "model_family",
                    "source_dice",
                    "a1_dice",
                    "outcome",
                    "adaptation_outcome",
                    "harm_label",
                    "benefit_label",
                ]
            ].to_csv(
                out / "r05d4_crosscheck_rows.csv",
                index=False
            )

    print("\nR05D4_CROSSCHECK_STATUS:", cross_status)

    # ------------------------------------------------------------------
    # 5. Final decision.
    # ------------------------------------------------------------------
    final_pass = (
        exact_unique_1to1
        and len(recovered) == len(ext)
        and recovered["outcome"].notna().all()
        and sid_agree.all()
        and cross_status in {"PASS", "UNAVAILABLE"}
    )

    # Save mapping audit, not the final labeled external feature table.
    mapping_cols = [
        "_feature_fingerprint",
        "sample_id",
        "sample_id_r08",
        "model_family",
        "source_dice",
        "a1_dice",
        "outcome",
    ]
    recovered[mapping_cols].to_csv(
        out / "external_to_r08e1_exact_row_mapping.csv",
        index=False
    )

    print("\n===== FINAL LINEAGE DECISION =====")
    if final_pass:
        print("DECISION: ROW_LINEAGE_RECOVERABLE")
        print("External rows can be linked to R08E1 outcomes by exact feature fingerprint.")
        if cross_status == "PASS":
            print("R05D4 independent post-GT label cross-check also PASSED.")
        print(
            "NEXT: build a frozen external labeled evaluation panel in a "
            "separate script; do not train on these external outcomes."
        )
    else:
        print("DECISION: STOP_ROW_LINEAGE_RECOVERY")
        print("Do not build an external labeled panel yet.")

    print("\nOutput:", out)
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
