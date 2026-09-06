
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H2_NeoPolyp_external_labeled_panel_builder_fix1.py

Build the frozen NeoPolyp evaluation-only labeled panel after the lineage audit
proved that the current 9000 external rows are uniquely traceable to R05D2
under atol=1e-14 and that the recovered explicit keys agree 9000/9000 with R05D4.

IMPORTANT
---------
- This script DOES NOT train or tune any model.
- R05D4 labels are attached for evaluation only.
- No row-order matching.
- No nearest-neighbour matching.
- Matching is restricted within the same sample_id.
- All 18 RAW features must match within the FROZEN atol=1e-14.
- The mapping must be a complete one-to-one bijection.
- Labels are joined only by explicit key:
      sample_id + model_state_id
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


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

FROZEN_ATOL = 1e-14

DEFAULT_EXTERNAL = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10E1_external_complete_feature_v3/"
    "external_complete_invariant_feature_table_v2.csv"
)

DEFAULT_R05D2 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1/"
    "target_paot_probabilities.csv"
)

DEFAULT_R05D4 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H2_NeoPolyp_external_labeled_panel_fix1_v1"
)


def norm_sid(s):
    return (
        s.astype(str)
         .str.strip()
         .str.replace("\\", "/", regex=False)
         .str.lower()
    )


def numeric_matrix(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        for c in cols
    ])


def strict_match_matrix(A, B, atol):
    a = A[:, None, :]
    b = B[None, :, :]

    both_nan = np.isnan(a) & np.isnan(b)
    finite_close = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(a - b) <= atol)
    )
    return np.all(both_nan | finite_close, axis=2)


def max_abs_diff(a, b):
    diff = np.abs(a - b)
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    diff[both_nan] = 0.0
    diff[one_nan] = np.inf
    return float(np.max(diff))


def recover_explicit_state_identity(ext, r05):
    """
    Recover model_state_id for every external row under the already-audited,
    frozen numerical equivalence rule.
    """
    e = ext.copy()
    r = r05.copy()

    e["_external_row_id"] = np.arange(len(e), dtype=np.int64)
    e["_sid_norm"] = norm_sid(e["sample_id"])
    r["_sid_norm"] = norm_sid(r["sample_id"])

    if set(e["_sid_norm"]) != set(r["_sid_norm"]):
        raise AssertionError("External/R05D2 sample_id sets are not identical.")

    mapping = []

    for sid in tqdm(sorted(set(e["_sid_norm"])), desc="Recovering state identity"):
        eg = e[e["_sid_norm"] == sid]
        rg = r[r["_sid_norm"] == sid]

        if len(eg) != 9 or len(rg) != 9:
            raise AssertionError(
                f"{sid}: expected 9 external and 9 R05D2 rows, "
                f"got {len(eg)} and {len(rg)}"
            )

        A = numeric_matrix(eg, RAW_FEATURES)
        B = numeric_matrix(rg, RAW_FEATURES)

        M = strict_match_matrix(A, B, FROZEN_ATOL)

        row_counts = M.sum(axis=1)
        col_counts = M.sum(axis=0)

        if not np.all(row_counts == 1):
            raise AssertionError(
                f"{sid}: target row match counts are not all exactly one: "
                f"{row_counts.tolist()}"
            )
        if not np.all(col_counts == 1):
            raise AssertionError(
                f"{sid}: R05D2 row reuse/ambiguity detected: "
                f"{col_counts.tolist()}"
            )

        for i in range(len(eg)):
            j = int(np.flatnonzero(M[i])[0])
            er = eg.iloc[i]
            rr = rg.iloc[j]

            d = max_abs_diff(A[i], B[j])
            if d > FROZEN_ATOL:
                raise AssertionError(
                    f"Unexpected diff {d} > frozen atol {FROZEN_ATOL}"
                )

            mapping.append({
                "_external_row_id": int(er["_external_row_id"]),
                "sample_id": str(er["sample_id"]),
                "sample_id_r05d2": str(rr["sample_id"]),
                "model_family": str(rr["model_family"]),
                "model_state_id": str(rr["model_state_id"]),
                "training_seed": int(rr["training_seed"]),
                "checkpoint_sha256": str(rr["checkpoint_sha256"]),
                "max_abs_feature_diff": d,
            })

    mapping = pd.DataFrame(mapping)

    if len(mapping) != len(e):
        raise AssertionError(
            f"Recovered {len(mapping)} rows, expected {len(e)}."
        )

    if mapping["_external_row_id"].nunique() != len(e):
        raise AssertionError("External row ID is not one-to-one.")

    if (
        mapping[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
        != len(e)
    ):
        raise AssertionError("sample_id + model_state_id is not unique.")

    if mapping["max_abs_feature_diff"].max() > FROZEN_ATOL:
        raise AssertionError("Recovered mapping exceeds frozen tolerance.")

    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=DEFAULT_EXTERNAL)
    ap.add_argument("--r05d2", default=DEFAULT_R05D2)
    ap.add_argument("--r05d4", default=DEFAULT_R05D4)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H2 NEOPOLYP EXTERNAL LABELED PANEL BUILDER FIX1 =====")
    print("STATUS: EVALUATION-ONLY PANEL CONSTRUCTION")
    print("TRAINING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("FROZEN NUMERIC MATCH ATOL:", FROZEN_ATOL)
    print("LABEL JOIN KEY: sample_id + model_state_id\n")

    paths = {
        "external": Path(args.external),
        "r05d2": Path(args.r05d2),
        "r05d4": Path(args.r05d4),
    }

    for name, p in paths.items():
        print(name, "exists:", p.exists(), "path:", p)
        if not p.exists():
            raise FileNotFoundError(p)

    ext = pd.read_csv(paths["external"], low_memory=False)
    r05 = pd.read_csv(paths["r05d2"], low_memory=False)
    d4 = pd.read_csv(paths["r05d4"], low_memory=False)

    print("\nInput rows:")
    print(" external:", len(ext))
    print(" R05D2:", len(r05))
    print(" R05D4:", len(d4))

    if len(ext) != 9000 or len(r05) != 9000 or len(d4) != 9000:
        raise AssertionError("Expected exactly 9000 rows in all three tables.")

    for c in ["sample_id"] + RAW_FEATURES:
        if c not in ext.columns:
            raise AssertionError(f"External missing required feature: {c}")
        if c not in r05.columns:
            raise AssertionError(f"R05D2 missing required feature: {c}")

    required_r05_meta = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    for c in required_r05_meta:
        if c not in r05.columns:
            raise AssertionError(f"R05D2 missing identity column: {c}")

    required_labels = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "source_dice",
        "a1_dice",
        "delta_dice",
        "adaptation_outcome",
        "harm_label",
        "benefit_label",
    ]
    for c in required_labels:
        if c not in d4.columns:
            raise AssertionError(f"R05D4 missing required column: {c}")

    # --------------------------------------------------------------
    # Recover explicit identity under the already-audited fixed rule.
    # --------------------------------------------------------------
    mapping = recover_explicit_state_identity(ext, r05)

    print("\n===== RECOVERED IDENTITY =====")
    print("Rows:", len(mapping))
    print(
        "Unique sample_id+model_state_id:",
        mapping[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
    )
    print(
        "Max numeric bridge difference:",
        mapping["max_abs_feature_diff"].max()
    )
    print("Model families:")
    print(mapping["model_family"].value_counts().to_string())
    print("Model states:", mapping["model_state_id"].nunique())

    # Attach recovered identity to external rows by our internally created
    # immutable external row ID, not by file row order between separate files.
    ext2 = ext.copy()
    ext2["_external_row_id"] = np.arange(len(ext2), dtype=np.int64)

    identified = ext2.merge(
        mapping,
        on="_external_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_bridge"),
    )

    # Verify the mapping's sample_id is identical to the external row.
    sid_ok = (
        norm_sid(identified["sample_id"])
        == norm_sid(identified["sample_id_bridge"])
    )
    print("sample_id recovery agreement:", int(sid_ok.sum()), "/", len(identified))
    if not sid_ok.all():
        raise AssertionError("Recovered identity sample_id mismatch.")

    # Keep canonical external sample_id.
    identified = identified.drop(
        columns=["sample_id_bridge", "sample_id_r05d2"]
    )

    # --------------------------------------------------------------
    # Join post-GT labels using explicit unique key only.
    # --------------------------------------------------------------
    identified["_sid_norm"] = norm_sid(identified["sample_id"])

    d4j = d4.copy()
    d4j["_sid_norm"] = norm_sid(d4j["sample_id"])

    key = ["_sid_norm", "model_state_id"]

    if identified[key].drop_duplicates().shape[0] != len(identified):
        raise AssertionError("Identified external key is not unique.")
    if d4j[key].drop_duplicates().shape[0] != len(d4j):
        raise AssertionError("R05D4 label key is not unique.")

    label_cols = [
        "_sid_norm",
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "source_dice",
        "a1_dice",
        "delta_dice",
        "adaptation_outcome",
        "harm_label",
        "benefit_label",
    ]

    labeled = identified.merge(
        d4j[label_cols],
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_bridge", "_r05d4"),
    )

    # --------------------------------------------------------------
    # Cross-check identity metadata before accepting labels.
    # --------------------------------------------------------------
    fam_ok = (
        labeled["model_family_bridge"].astype(str)
        == labeled["model_family_r05d4"].astype(str)
    )
    seed_ok = (
        labeled["training_seed_bridge"].astype(str)
        == labeled["training_seed_r05d4"].astype(str)
    )
    sha_ok = (
        labeled["checkpoint_sha256_bridge"].astype(str)
        == labeled["checkpoint_sha256_r05d4"].astype(str)
    )

    print("\n===== R05D4 LABEL JOIN AUDIT =====")
    print("Rows after join:", len(labeled))
    print("harm_label non-null:", int(labeled["harm_label"].notna().sum()))
    print("model_family agreement:", int(fam_ok.sum()), "/", len(labeled))
    print("training_seed agreement:", int(seed_ok.sum()), "/", len(labeled))
    print("checkpoint_sha256 agreement:", int(sha_ok.sum()), "/", len(labeled))

    if not labeled["harm_label"].notna().all():
        raise AssertionError("Some external rows failed to receive R05D4 labels.")
    if not fam_ok.all():
        raise AssertionError("model_family mismatch against R05D4.")
    if not seed_ok.all():
        raise AssertionError("training_seed mismatch against R05D4.")
    if not sha_ok.all():
        raise AssertionError("checkpoint_sha256 mismatch against R05D4.")

    # Canonicalize identity columns.
    labeled["model_family"] = labeled["model_family_bridge"]
    labeled["training_seed"] = labeled["training_seed_bridge"]
    labeled["checkpoint_sha256"] = labeled["checkpoint_sha256_bridge"]

    drop_cols = [
        "_sid_norm",
        "model_family_bridge",
        "model_family_r05d4",
        "training_seed_bridge",
        "training_seed_r05d4",
        "checkpoint_sha256_bridge",
        "checkpoint_sha256_r05d4",
    ]
    labeled = labeled.drop(columns=drop_cols)

    # Put identity/labels at the front while retaining every external feature.
    front = [
        "_external_row_id",
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "max_abs_feature_diff",
        "source_dice",
        "a1_dice",
        "delta_dice",
        "adaptation_outcome",
        "harm_label",
        "benefit_label",
    ]
    rest = [c for c in labeled.columns if c not in front]
    labeled = labeled[front + rest]

    print("\n===== FINAL LABEL DISTRIBUTION =====")
    print("adaptation_outcome:")
    print(labeled["adaptation_outcome"].value_counts(dropna=False).to_string())
    print("\nharm_label:")
    print(labeled["harm_label"].value_counts(dropna=False).to_string())
    print("\nbenefit_label:")
    print(labeled["benefit_label"].value_counts(dropna=False).to_string())

    expected_harm = {"0": 5546, "1": 3454}
    observed_harm = (
        labeled["harm_label"]
        .astype(int)
        .astype(str)
        .value_counts()
        .to_dict()
    )

    if observed_harm != expected_harm:
        raise AssertionError(
            f"Unexpected harm counts: {observed_harm}, expected {expected_harm}"
        )

    expected_outcomes = {"NEUTRAL": 4403, "HARM": 3454, "BENEFIT": 1143}
    observed_outcomes = (
        labeled["adaptation_outcome"]
        .astype(str)
        .str.upper()
        .value_counts()
        .to_dict()
    )
    if observed_outcomes != expected_outcomes:
        raise AssertionError(
            f"Unexpected outcome counts: {observed_outcomes}, "
            f"expected {expected_outcomes}"
        )

    panel_path = out / "neopolyp_external_evaluation_labeled_panel.csv"
    mapping_path = out / "neopolyp_external_state_identity_mapping.csv"
    audit_path = out / "neopolyp_external_panel_audit_summary.csv"

    labeled.to_csv(panel_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    audit = pd.DataFrame([{
        "rows": len(labeled),
        "unique_sample_id": norm_sid(labeled["sample_id"]).nunique(),
        "unique_sample_state": labeled[
            ["sample_id", "model_state_id"]
        ].drop_duplicates().shape[0],
        "harm_rows": int((labeled["harm_label"].astype(int) == 1).sum()),
        "nonharm_rows": int((labeled["harm_label"].astype(int) == 0).sum()),
        "benefit_rows": int((labeled["benefit_label"].astype(int) == 1).sum()),
        "neutral_rows": int(
            labeled["adaptation_outcome"]
            .astype(str).str.upper().eq("NEUTRAL").sum()
        ),
        "max_abs_numeric_bridge_diff": float(
            labeled["max_abs_feature_diff"].max()
        ),
        "frozen_atol": FROZEN_ATOL,
        "label_join_key": "sample_id+model_state_id",
        "training_used": "NO",
        "threshold_tuning_used": "NO",
    }])
    audit.to_csv(audit_path, index=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION: EXTERNAL_EVALUATION_PANEL_LOCKED")
    print("Rows:", len(labeled))
    print("Unique sample_id:", norm_sid(labeled["sample_id"]).nunique())
    print(
        "Unique sample_id+model_state_id:",
        labeled[["sample_id", "model_state_id"]]
        .drop_duplicates()
        .shape[0]
    )
    print("HARM:", int((labeled["harm_label"].astype(int) == 1).sum()))
    print("NON-HARM:", int((labeled["harm_label"].astype(int) == 0).sum()))
    print("Maximum numeric bridge diff:", labeled["max_abs_feature_diff"].max())
    print("External labels used for fitting: NO")
    print("External labels used for threshold tuning: NO")

    print("\nOutputs:")
    print(panel_path)
    print(mapping_path)
    print(audit_path)
    print("PASS")


if __name__ == "__main__":
    main()
