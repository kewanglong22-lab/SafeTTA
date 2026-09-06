
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_numeric_equivalence_state_bridge_audit_fix1.py

Purpose
-------
Determine whether the current 9000-row NeoPolyp external table is the SAME
R05D2 RAW 18-feature representation up to harmless floating-point
serialization / transformation noise.

Why this audit is justified
---------------------------
Previous exact-fingerprint audit showed:
  A27 vs R05D2 RAW: 8944 / 9000 rows exact
  A28 vs R05D2 RAW: 8933 / 9000 rows exact
  E1v3 vs R05D2 RAW: 8920 / 9000 rows exact

and the RAW feature distributions had exactly the same means/medians.
That strongly suggests row identity may be recoverable under a very small
fixed numerical tolerance.

Safety rules
------------
- NO row-order matching
- NO nearest-neighbour matching
- NO label attachment
- Matching is only allowed within the SAME sample_id.
- A row is accepted only if it has exactly ONE R05D2 candidate satisfying
  np.isclose for ALL 18 raw features at a predeclared tolerance.
- The mapping must be a full one-to-one bijection.
- We report the SMALLEST tolerance at which a complete unique bijection exists.
- If no strict tolerance yields a full unique bijection, STOP.

This script only audits/reconstructs state identity. It does not train SGUC.
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

DEFAULT_R05D2 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1/"
    "target_paot_probabilities.csv"
)

DEFAULT_E1V3 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10E1_external_complete_feature_v3/"
    "external_complete_invariant_feature_table_v2.csv"
)

DEFAULT_R05D4 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10H1_numeric_equivalence_state_bridge_audit_fix1_v1"
)

# Strictly predeclared tolerances, from near-machine precision upward.
# rtol=0 is deliberate: absolute differences are directly auditable.
TOLERANCES = [
    0.0,
    1e-15,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
]


def norm_sid_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def numeric_matrix(df, cols):
    arr = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in cols
    ])
    return arr


def allclose_matrix(A, B, atol):
    """
    Return M[i,j] == True iff all 18 dimensions of A[i] and B[j]
    are equal within fixed absolute tolerance.
    NaNs only match NaNs.
    """
    # shape: nA x nB x d
    a = A[:, None, :]
    b = B[None, :, :]

    both_nan = np.isnan(a) & np.isnan(b)
    finite_close = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= atol)
    dim_ok = both_nan | finite_close
    return np.all(dim_ok, axis=2)


def min_pairwise_max_abs(A, B):
    """
    Diagnostic only. Returns pairwise max absolute difference over dimensions.
    NaN==NaN contributes 0; one-sided NaN contributes inf.
    """
    a = A[:, None, :]
    b = B[None, :, :]

    diff = np.abs(a - b)
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)

    diff[both_nan] = 0.0
    diff[one_nan] = np.inf
    return np.max(diff, axis=2)


def audit_tolerance(target, r05, atol):
    """
    Test full within-sample one-to-one equivalence at a fixed tolerance.
    Returns summary and, only if fully unique, the exact mapping rows.
    """
    t = target.copy()
    r = r05.copy()

    t["_sid_norm"] = norm_sid_series(t["sample_id"])
    r["_sid_norm"] = norm_sid_series(r["sample_id"])

    sids_t = set(t["_sid_norm"])
    sids_r = set(r["_sid_norm"])

    if sids_t != sids_r:
        return {
            "atol": atol,
            "full_unique_bijection": False,
            "reason": "sample_id_set_mismatch",
            "matched_rows": 0,
            "ambiguous_target_rows": np.nan,
            "unmatched_target_rows": np.nan,
            "reused_r05_rows": np.nan,
        }, None

    mapping_rows = []
    matched = 0
    unmatched = 0
    ambiguous = 0
    reused = 0
    per_sample_fail = 0

    for sid in sids_t:
        tg = t[t["_sid_norm"] == sid]
        rg = r[r["_sid_norm"] == sid]

        if len(tg) != len(rg):
            per_sample_fail += 1
            continue

        A = numeric_matrix(tg, RAW_FEATURES)
        B = numeric_matrix(rg, RAW_FEATURES)
        M = allclose_matrix(A, B, atol)

        row_counts = M.sum(axis=1)
        col_counts = M.sum(axis=0)

        unmatched += int(np.sum(row_counts == 0))
        ambiguous += int(np.sum(row_counts > 1))
        reused += int(np.sum(col_counts > 1))

        # Strict bijection condition:
        # every target row has exactly one candidate,
        # every R05 row has exactly one target.
        if not (np.all(row_counts == 1) and np.all(col_counts == 1)):
            per_sample_fail += 1
            continue

        for i in range(len(tg)):
            j = int(np.flatnonzero(M[i])[0])

            tr = tg.iloc[i]
            rr = rg.iloc[j]

            # max absolute difference for audit
            av = numeric_matrix(tg.iloc[[i]], RAW_FEATURES)[0]
            bv = numeric_matrix(rg.iloc[[j]], RAW_FEATURES)[0]
            diff = np.abs(av - bv)
            both_nan = np.isnan(av) & np.isnan(bv)
            diff[both_nan] = 0.0
            max_abs = float(np.nanmax(diff)) if len(diff) else 0.0

            mapping_rows.append({
                "sample_id_target": tr["sample_id"],
                "sample_id_r05d2": rr["sample_id"],
                "model_family": rr["model_family"],
                "model_state_id": rr["model_state_id"],
                "training_seed": rr["training_seed"],
                "checkpoint_sha256": rr["checkpoint_sha256"],
                "max_abs_feature_diff": max_abs,
            })
            matched += 1

    full = (
        per_sample_fail == 0
        and matched == len(target)
        and unmatched == 0
        and ambiguous == 0
        and reused == 0
    )

    summary = {
        "atol": atol,
        "full_unique_bijection": bool(full),
        "reason": "PASS" if full else "not_full_unique_bijection",
        "matched_rows": matched,
        "unmatched_target_rows": unmatched,
        "ambiguous_target_rows": ambiguous,
        "reused_r05_rows": reused,
        "failed_samples": per_sample_fail,
    }

    mapping = pd.DataFrame(mapping_rows) if full else None
    return summary, mapping


def mismatch_distance_diagnostic(target, r05):
    """
    For each sample_id, calculate the best possible max-abs distance for each
    target row. Diagnostic only; never used to assign identity.
    """
    t = target.copy()
    r = r05.copy()
    t["_sid_norm"] = norm_sid_series(t["sample_id"])
    r["_sid_norm"] = norm_sid_series(r["sample_id"])

    vals = []

    for sid in tqdm(sorted(set(t["_sid_norm"])), desc="Distance diagnostic"):
        tg = t[t["_sid_norm"] == sid]
        rg = r[r["_sid_norm"] == sid]

        if len(tg) == 0 or len(rg) == 0:
            continue

        A = numeric_matrix(tg, RAW_FEATURES)
        B = numeric_matrix(rg, RAW_FEATURES)
        D = min_pairwise_max_abs(A, B)

        best = np.min(D, axis=1)
        vals.extend(best.tolist())

    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]

    out = {
        "rows": len(vals),
        "finite_rows": len(finite),
        "exact_zero": int(np.sum(finite == 0.0)),
        "max": float(np.max(finite)) if len(finite) else np.nan,
        "p50": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
        "p90": float(np.quantile(finite, 0.90)) if len(finite) else np.nan,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
        "p99": float(np.quantile(finite, 0.99)) if len(finite) else np.nan,
        "p999": float(np.quantile(finite, 0.999)) if len(finite) else np.nan,
    }
    return out, vals


def crosscheck_mapping_with_r05d4(mapping, r05d4):
    """
    Cross-check only; does not attach labels to the external feature table.
    """
    print("\n===== R05D4 EXPLICIT-KEY CROSS-CHECK =====")

    key = ["sample_id", "model_state_id"]

    m = mapping.copy()
    m["sample_id"] = norm_sid_series(m["sample_id_target"])
    d4 = r05d4.copy()
    d4["sample_id"] = norm_sid_series(d4["sample_id"])

    print("Mapping unique key:", m[key].drop_duplicates().shape[0], "/", len(m))
    print("R05D4 unique key:", d4[key].drop_duplicates().shape[0], "/", len(d4))

    checked = m.merge(
        d4[
            [
                "sample_id",
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
                "adaptation_outcome",
                "harm_label",
                "benefit_label",
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_bridge", "_r05d4"),
    )

    print("Matched R05D4 rows:", int(checked["harm_label"].notna().sum()), "/", len(checked))

    family_ok = (
        checked["model_family_bridge"].astype(str)
        == checked["model_family_r05d4"].astype(str)
    )
    seed_ok = (
        checked["training_seed_bridge"].astype(str)
        == checked["training_seed_r05d4"].astype(str)
    )
    sha_ok = (
        checked["checkpoint_sha256_bridge"].astype(str)
        == checked["checkpoint_sha256_r05d4"].astype(str)
    )

    print("model_family agreement:", int(family_ok.sum()), "/", len(checked))
    print("training_seed agreement:", int(seed_ok.sum()), "/", len(checked))
    print("checkpoint_sha256 agreement:", int(sha_ok.sum()), "/", len(checked))

    full = (
        checked["harm_label"].notna().all()
        and family_ok.all()
        and seed_ok.all()
        and sha_ok.all()
    )

    print("R05D4_CROSSCHECK:", "PASS" if full else "FAIL")
    return full, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r05d2", default=DEFAULT_R05D2)
    ap.add_argument("--external", default=DEFAULT_E1V3)
    ap.add_argument("--r05d4", default=DEFAULT_R05D4)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 NUMERIC-EQUIVALENCE STATE BRIDGE AUDIT FIX1 =====")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR MATCHING: FORBIDDEN")
    print("LABEL ATTACHMENT: NONE")
    print("MATCH RESTRICTION: SAME sample_id ONLY")
    print("MATCH RULE: all 18 RAW features within fixed absolute tolerance\n")

    r05_path = Path(args.r05d2)
    ext_path = Path(args.external)
    d4_path = Path(args.r05d4)

    for name, p in [("R05D2", r05_path), ("EXTERNAL", ext_path), ("R05D4", d4_path)]:
        print(name, "exists:", p.exists(), "path:", p)
        if not p.exists():
            raise FileNotFoundError(p)

    r05 = pd.read_csv(r05_path, low_memory=False)
    ext = pd.read_csv(ext_path, low_memory=False)
    d4 = pd.read_csv(d4_path, low_memory=False)

    for c in ["sample_id"] + RAW_FEATURES:
        if c not in r05.columns:
            raise AssertionError(f"R05D2 missing {c}")
        if c not in ext.columns:
            raise AssertionError(f"External missing {c}")

    print("\nRows:", len(ext), len(r05))
    print("External unique sample_id:", norm_sid_series(ext["sample_id"]).nunique())
    print("R05D2 unique sample_id:", norm_sid_series(r05["sample_id"]).nunique())

    print("\n===== DISTANCE DIAGNOSTIC =====")
    diag, vals = mismatch_distance_diagnostic(ext, r05)
    for k, v in diag.items():
        print(f"{k}: {v}")

    pd.DataFrame({"best_within_sample_max_abs_diff": vals}).to_csv(
        out / "within_sample_distance_diagnostic.csv",
        index=False,
    )

    summaries = []
    accepted_mapping = None
    accepted_tol = None

    print("\n===== FIXED-TOLERANCE BIJECTION AUDIT =====")
    for atol in TOLERANCES:
        summary, mapping = audit_tolerance(ext, r05, atol)
        summaries.append(summary)

        print(
            f"atol={atol:.1e} "
            f"full_unique_bijection={summary['full_unique_bijection']} "
            f"matched={summary['matched_rows']} "
            f"unmatched={summary['unmatched_target_rows']} "
            f"ambiguous={summary['ambiguous_target_rows']} "
            f"reused={summary['reused_r05_rows']} "
            f"failed_samples={summary['failed_samples']}"
        )

        if summary["full_unique_bijection"]:
            accepted_mapping = mapping
            accepted_tol = atol
            break

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "numeric_equivalence_tolerance_summary.csv", index=False)

    print("\n===== FINAL DECISION =====")
    if accepted_mapping is None:
        print("DECISION: NUMERIC_EQUIVALENCE_BRIDGE_NOT_PROVEN")
        print(
            "No predeclared strict tolerance produced a complete unique "
            "within-sample bijection. Do NOT recover model_state_id."
        )
        print(
            "NEXT: inspect/regenerate the upstream R10A2_7 feature table with "
            "explicit model_state_id retained."
        )
        print("\nOutput:", out)
        return

    print("Smallest full-bijection atol:", accepted_tol)
    print("Recovered rows:", len(accepted_mapping))
    print(
        "Unique sample_id+model_state_id:",
        accepted_mapping[
            ["sample_id_target", "model_state_id"]
        ].drop_duplicates().shape[0],
    )
    print(
        "Mapping max feature difference:",
        float(accepted_mapping["max_abs_feature_diff"].max()),
    )

    mapping_path = out / "external_to_R05D2_numeric_equivalence_state_mapping.csv"
    accepted_mapping.to_csv(mapping_path, index=False)

    d4_pass, checked = crosscheck_mapping_with_r05d4(accepted_mapping, d4)
    checked.to_csv(
        out / "external_state_mapping_R05D4_crosscheck.csv",
        index=False,
    )

    # Scientific guardrail: tolerate only genuinely tiny numeric discrepancies.
    # If a full mapping requires >1e-8 absolute tolerance, flag as too loose
    # for an automatic lineage claim.
    strict_enough = accepted_tol <= 1e-8

    print("\nSTRICT_NUMERIC_TOLERANCE_ACCEPTABLE:", strict_enough)

    if d4_pass and strict_enough:
        print("DECISION: EXTERNAL_STATE_IDENTITY_RECOVERED")
        print(
            "The current external rows are uniquely traceable to R05D2 "
            "sample_id+model_state_id under a fixed, very small numeric tolerance."
        )
        print("R05D4 explicit-key cross-check also passed.")
        print(
            "NEXT: build a separate frozen evaluation-only external labeled panel "
            "using this audited state mapping; external labels must never be used "
            "for fitting or threshold tuning."
        )
    else:
        print("DECISION: NUMERIC_EQUIVALENCE_BRIDGE_NOT_ACCEPTED")
        if not d4_pass:
            print("Reason: R05D4 explicit-key cross-check failed.")
        if not strict_enough:
            print("Reason: required numeric tolerance is > 1e-8.")

    print("\nOutputs:")
    print(out / "numeric_equivalence_tolerance_summary.csv")
    print(mapping_path)
    print(out / "external_state_mapping_R05D4_crosscheck.csv")
    print("PASS: AUDIT COMPLETED")


if __name__ == "__main__":
    main()
