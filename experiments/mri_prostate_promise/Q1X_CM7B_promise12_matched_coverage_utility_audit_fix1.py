#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1.py

SafeTTA Q1 enhancement — CM7B.

Purpose
-------
Post-GT fixed-protocol utility audit on PROMISE12.

Question:
    At exactly the same adaptation coverage, does the frozen SafeTTA risk gate
    outperform random slice selection?

This is NOT a new prospective endpoint. CM6 remains the prospective target
evaluation. CM7B uses the already locked CM6 outcomes and threshold only.

Primary comparisons
-------------------
Frozen SafeTTA fixed-SOURCE gate
vs
Random adaptation at exactly the same number of adapted slices PER FAMILY.

Metrics:
- prevented HARM fraction (= HARM recall of retain-SOURCE gate);
- mean patient-level 3D deployed Dice;
- mean slice deployed Dice.

Statistics:
A) 10,000 full-target matched-coverage randomization replicates.
B) 2,000 paired patient-cluster bootstrap replicates, with 10 matched random
   policies averaged inside each bootstrap replicate.

No model, threshold, HARM definition, representation, or target policy is
changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM7B-v1-fix1"
BUILD = "Q1X_CM7B_PROMISE12_MATCHED_COVERAGE_UTILITY_AUDIT_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"

CM6_LOCK = (
    OUT
    / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
    / "CM6_PROMISE12_GT_REVEAL_LOCKED_EVALUATION_LOCK.json"
)
EXPECTED_CM6_LOCK_SHA = (
    "151b067b73ab690694f40f8f72d06fc0414de56b5763f1bc31a8eb478db4f4a7"
)

CM7A_LOCK = (
    OUT
    / "Q1X_CM7A_promise12_external_ranking_baseline_audit_fix2_v1"
    / "CM7A_PROMISE12_EXTERNAL_RANKING_BASELINE_AUDIT_LOCK.json"
)
EXPECTED_CM7A_LOCK_SHA = (
    "47fc4257368542f34e3cc348229cd250c8ec2349cba1819731aff60af2190cd7"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM7B_promise12_matched_coverage_utility_audit_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_PATIENTS = 50
N_SLICES_PER_FAMILY = 1377
N_ROWS = 4131

FULL_RANDOM_REPS = 10000
FULL_RANDOM_SEED = 20260908

BOOTSTRAP_REPS = 2000
BOOTSTRAP_INNER_RANDOM_DRAWS = 10
BOOTSTRAP_SEED = 20260909

EXPECTED_CM6_SAFE_POOLED = {
    "threshold": 0.2950987857155136,
    "harm_recall": 0.9007633587786259,
    "fpr": 0.8007798652959943,
    "adaptation_coverage": 0.1675139191479061,
    "mean_patient_3d_deployed_dice": 0.7446111204829475,
}

PASS_DECISION = (
    "PROMISE12_MATCHED_COVERAGE_UTILITY_AUDIT_COMPLETE_READY_FOR_FINAL_SYNTHESIS"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_lineage():
    for label, path, expected in [
        ("CM6_LOCK", CM6_LOCK, EXPECTED_CM6_LOCK_SHA),
        ("CM7A_LOCK", CM7A_LOCK, EXPECTED_CM7A_LOCK_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm6 = load_json(CM6_LOCK)
    cm7a = load_json(CM7A_LOCK)

    if cm6.get("status") != "PASS":
        raise RuntimeError("CM6 status changed.")
    if cm7a.get("status") != "PASS":
        raise RuntimeError("CM7A status changed.")

    if cm6.get("post_gt_changes", {}).get("thresholds") is not False:
        raise RuntimeError("CM6 post-GT threshold boundary changed.")
    if cm6.get("post_gt_changes", {}).get("models") is not False:
        raise RuntimeError("CM6 post-GT model boundary changed.")
    if cm6.get("post_gt_changes", {}).get("representation") is not False:
        raise RuntimeError("CM6 post-GT representation boundary changed.")

    return cm6, cm7a


def load_outcomes(cm6):
    path = Path(cm6["artifacts"]["outcomes"])
    if sha256_file(path) != cm6["artifacts"]["outcomes_sha256"]:
        raise RuntimeError("CM6 outcome SHA mismatch.")

    df = pd.read_csv(path)

    required = {
        "family",
        "global_index",
        "case_key",
        "slice_index",
        "risk_score",
        "harmful",
        "source_dice",
        "tent1_dice",
        "source_tp",
        "source_fp",
        "source_fn",
        "tent_tp",
        "tent_fp",
        "tent_fn",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing CM6 outcome columns: {sorted(missing)}"
        )

    if len(df) != N_ROWS:
        raise RuntimeError(f"Unexpected CM6 rows: {len(df)}")
    if df["case_key"].nunique() != N_PATIENTS:
        raise RuntimeError("Unexpected PROMISE patient count.")

    for family in FAMILIES:
        sub = df[df["family"] == family].sort_values("global_index")
        if len(sub) != N_SLICES_PER_FAMILY:
            raise RuntimeError(f"{family}: row count changed.")
        if not np.array_equal(
            sub["global_index"].to_numpy(dtype=np.int64),
            np.arange(N_SLICES_PER_FAMILY, dtype=np.int64),
        ):
            raise RuntimeError(f"{family}: global_index ordering changed.")

    return df.reset_index(drop=True)


def patient_group_ids(df):
    """
    Full-dataset group IDs: one group per (family, patient), 150 total.
    """
    fam_to_i = {f: i for i, f in enumerate(FAMILIES)}
    patient_keys = sorted(df["case_key"].astype(str).unique())
    patient_to_i = {p: i for i, p in enumerate(patient_keys)}

    g = np.empty(len(df), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        g[i] = (
            fam_to_i[str(row.family)] * N_PATIENTS
            + patient_to_i[str(row.case_key)]
        )

    return g, len(FAMILIES) * N_PATIENTS


def deployed_counts(df, adapt_mask):
    adapt = np.asarray(adapt_mask, dtype=bool)

    source_tp = df["source_tp"].to_numpy(dtype=np.float64)
    source_fp = df["source_fp"].to_numpy(dtype=np.float64)
    source_fn = df["source_fn"].to_numpy(dtype=np.float64)

    tent_tp = df["tent_tp"].to_numpy(dtype=np.float64)
    tent_fp = df["tent_fp"].to_numpy(dtype=np.float64)
    tent_fn = df["tent_fn"].to_numpy(dtype=np.float64)

    tp = np.where(adapt, tent_tp, source_tp)
    fp = np.where(adapt, tent_fp, source_fp)
    fn = np.where(adapt, tent_fn, source_fn)

    return tp, fp, fn


def mean_patient_3d_dice_from_groups(
    df,
    adapt_mask,
    group_ids,
    n_groups,
):
    tp, fp, fn = deployed_counts(df, adapt_mask)

    tp_g = np.bincount(
        group_ids,
        weights=tp,
        minlength=n_groups,
    )
    fp_g = np.bincount(
        group_ids,
        weights=fp,
        minlength=n_groups,
    )
    fn_g = np.bincount(
        group_ids,
        weights=fn,
        minlength=n_groups,
    )

    den = 2.0 * tp_g + fp_g + fn_g
    dice = np.ones_like(den, dtype=np.float64)

    nonzero = den > 0
    dice[nonzero] = (
        2.0 * tp_g[nonzero] / den[nonzero]
    )

    return float(dice.mean())


def policy_metrics(
    df,
    adapt_mask,
    group_ids=None,
    n_groups=None,
):
    adapt = np.asarray(adapt_mask, dtype=bool)
    harmful = df["harmful"].to_numpy(dtype=np.int64) == 1

    harm_total = int(harmful.sum())
    if harm_total <= 0:
        raise RuntimeError("No HARM rows.")

    prevented_harm = float(
        np.sum(harmful & (~adapt)) / harm_total
    )

    source_dice = df["source_dice"].to_numpy(dtype=np.float64)
    tent_dice = df["tent1_dice"].to_numpy(dtype=np.float64)

    deployed_slice = np.where(
        adapt,
        tent_dice,
        source_dice,
    )

    if group_ids is None:
        group_ids, n_groups = patient_group_ids(df)

    patient_dice = mean_patient_3d_dice_from_groups(
        df,
        adapt,
        group_ids,
        n_groups,
    )

    return {
        "adaptation_coverage": float(adapt.mean()),
        "adapted_rows": int(adapt.sum()),
        "prevented_harm_fraction": prevented_harm,
        "mean_slice_deployed_dice": float(
            deployed_slice.mean()
        ),
        "mean_patient_3d_deployed_dice": patient_dice,
    }


def fixed_safe_policy(df, tau):
    return (
        df["risk_score"].to_numpy(dtype=np.float64)
        < float(tau)
    )


def source_only_metrics(df):
    group_ids, n_groups = patient_group_ids(df)
    return policy_metrics(
        df,
        np.zeros(len(df), dtype=bool),
        group_ids,
        n_groups,
    )


def tent_all_metrics(df):
    group_ids, n_groups = patient_group_ids(df)
    return policy_metrics(
        df,
        np.ones(len(df), dtype=bool),
        group_ids,
        n_groups,
    )


def family_exact_k(df, adapt_mask):
    out = {}
    fam = df["family"].astype(str).to_numpy()
    for family in FAMILIES:
        m = fam == family
        out[family] = {
            "n": int(m.sum()),
            "k_adapt": int(np.sum(adapt_mask[m])),
            "coverage": float(np.mean(adapt_mask[m])),
        }
    return out


def random_exact_family_coverage(
    df,
    family_k,
    rng,
):
    fam = df["family"].astype(str).to_numpy()
    adapt = np.zeros(len(df), dtype=bool)

    for family in FAMILIES:
        idx = np.flatnonzero(fam == family)
        k = int(family_k[family]["k_adapt"])

        if k < 0 or k > len(idx):
            raise RuntimeError("Invalid matched-coverage k.")

        if k > 0:
            chosen = rng.choice(
                idx,
                size=k,
                replace=False,
            )
            adapt[chosen] = True

    return adapt


def full_target_randomization_test(
    df,
    safe_metrics,
    family_k,
):
    rng = np.random.default_rng(FULL_RANDOM_SEED)
    group_ids, n_groups = patient_group_ids(df)

    rows = []

    for rep in tqdm(
        range(FULL_RANDOM_REPS),
        desc="CM7B full-target matched randomization",
        unit="rep",
        dynamic_ncols=True,
    ):
        adapt = random_exact_family_coverage(
            df,
            family_k,
            rng,
        )
        m = policy_metrics(
            df,
            adapt,
            group_ids,
            n_groups,
        )
        m["rep"] = rep
        rows.append(m)

    rnd = pd.DataFrame(rows)

    summary = []

    for metric in [
        "prevented_harm_fraction",
        "mean_slice_deployed_dice",
        "mean_patient_3d_deployed_dice",
    ]:
        vals = rnd[metric].to_numpy(dtype=np.float64)
        safe = float(safe_metrics[metric])
        delta = safe - vals

        # One-sided randomization p-value:
        # probability a random matched-coverage policy is at least as good.
        p_random_ge_safe = float(
            (1 + np.sum(vals >= safe))
            / (1 + len(vals))
        )

        summary.append({
            "metric": metric,
            "safettta": safe,
            "random_mean": float(vals.mean()),
            "random_ci95_low": float(
                np.quantile(vals, 0.025)
            ),
            "random_ci95_high": float(
                np.quantile(vals, 0.975)
            ),
            "safettta_minus_random_mean": float(
                delta.mean()
            ),
            "delta_ci95_low": float(
                np.quantile(delta, 0.025)
            ),
            "delta_ci95_high": float(
                np.quantile(delta, 0.975)
            ),
            "one_sided_p_random_ge_safettta": (
                p_random_ge_safe
            ),
        })

    return rnd, pd.DataFrame(summary)


def bootstrap_dataset(df, sampled_patients):
    """
    Build a patient-cluster bootstrap dataset.
    Duplicate sampled patients receive distinct bootstrap cluster IDs.
    """
    patient_to_rows = {
        p: np.flatnonzero(
            df["case_key"].astype(str).to_numpy() == p
        )
        for p in sorted(
            df["case_key"].astype(str).unique()
        )
    }

    parts = []
    cluster_parts = []

    for occurrence, patient in enumerate(sampled_patients):
        idx = patient_to_rows[str(patient)]
        parts.append(idx)

        # Within one patient occurrence there are 3 family clusters.
        family = df.iloc[idx]["family"].astype(str).to_numpy()
        fam_to_i = {f: i for i, f in enumerate(FAMILIES)}
        cluster = np.asarray(
            [
                fam_to_i[f] * len(sampled_patients)
                + occurrence
                for f in family
            ],
            dtype=np.int64,
        )
        cluster_parts.append(cluster)

    idx = np.concatenate(parts)
    clusters = np.concatenate(cluster_parts)

    boot = df.iloc[idx].reset_index(drop=True)

    return boot, clusters, len(FAMILIES) * len(sampled_patients)


def paired_patient_bootstrap(
    df,
    tau,
):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.asarray(
        sorted(df["case_key"].astype(str).unique())
    )

    if len(patients) != N_PATIENTS:
        raise RuntimeError("Unexpected bootstrap patient count.")

    rows = []

    for rep in tqdm(
        range(BOOTSTRAP_REPS),
        desc="CM7B paired patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )

        boot, group_ids, n_groups = bootstrap_dataset(
            df,
            sampled,
        )

        safe_adapt = fixed_safe_policy(boot, tau)
        safe = policy_metrics(
            boot,
            safe_adapt,
            group_ids,
            n_groups,
        )
        family_k = family_exact_k(
            boot,
            safe_adapt,
        )

        random_metrics = []

        for _ in range(BOOTSTRAP_INNER_RANDOM_DRAWS):
            rnd_adapt = random_exact_family_coverage(
                boot,
                family_k,
                rng,
            )
            random_metrics.append(
                policy_metrics(
                    boot,
                    rnd_adapt,
                    group_ids,
                    n_groups,
                )
            )

        row = {"rep": rep}

        for metric in [
            "prevented_harm_fraction",
            "mean_slice_deployed_dice",
            "mean_patient_3d_deployed_dice",
        ]:
            rnd_mean = float(
                np.mean(
                    [
                        r[metric]
                        for r in random_metrics
                    ]
                )
            )
            row[f"safe_{metric}"] = float(
                safe[metric]
            )
            row[f"random_{metric}"] = rnd_mean
            row[f"delta_{metric}"] = float(
                safe[metric] - rnd_mean
            )

        rows.append(row)

    boot_df = pd.DataFrame(rows)

    summary = []

    for metric in [
        "prevented_harm_fraction",
        "mean_slice_deployed_dice",
        "mean_patient_3d_deployed_dice",
    ]:
        vals = boot_df[
            f"delta_{metric}"
        ].to_numpy(dtype=np.float64)

        summary.append({
            "metric": metric,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "inner_random_draws_per_rep": (
                BOOTSTRAP_INNER_RANDOM_DRAWS
            ),
            "mean_delta_safettta_minus_random": float(
                vals.mean()
            ),
            "ci95_low": float(
                np.quantile(vals, 0.025)
            ),
            "ci95_high": float(
                np.quantile(vals, 0.975)
            ),
            "safettta_significantly_better_95ci": bool(
                np.quantile(vals, 0.025) > 0.0
            ),
        })

    return boot_df, pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM7B PROMISE12 MATCHED-COVERAGE UTILITY AUDIT ====="
    )
    print("PROSPECTIVE_ENDPOINT=NO")
    print("CM6_PROSPECTIVE_RESULT_REMAINS_PRIMARY=YES")
    print("SAFE_POLICY=FROZEN_CM6_FIXED_SOURCE_THRESHOLD")
    print("RANDOM_BASELINE=EXACT_SAME_ADAPTED_ROWS_PER_FAMILY")
    print("FULL_RANDOM_REPS=", FULL_RANDOM_REPS)
    print("PATIENT_BOOTSTRAP_REPS=", BOOTSTRAP_REPS)
    print(
        "BOOTSTRAP_INNER_RANDOM_DRAWS=",
        BOOTSTRAP_INNER_RANDOM_DRAWS,
    )
    print("THRESHOLD_CHANGE=NO")
    print("MODEL_CHANGE=NO")
    print("HARM_DEFINITION_CHANGE=NO")

    cm6, cm7a = verify_lineage()
    df = load_outcomes(cm6)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    tau = float(
        cm6["operating_point_results"]
        ["fixed_source_threshold"]
    )

    if abs(
        tau - EXPECTED_CM6_SAFE_POOLED["threshold"]
    ) > 1e-12:
        raise RuntimeError(
            "Frozen CM6 SOURCE threshold changed."
        )

    safe_adapt = fixed_safe_policy(df, tau)
    group_ids, n_groups = patient_group_ids(df)

    safe = policy_metrics(
        df,
        safe_adapt,
        group_ids,
        n_groups,
    )

    # Exact reproduction gates against CM6.
    if abs(
        safe["adaptation_coverage"]
        - EXPECTED_CM6_SAFE_POOLED["adaptation_coverage"]
    ) > 1e-12:
        raise RuntimeError(
            "CM7B does not reproduce CM6 fixed-policy coverage."
        )
    if abs(
        safe["prevented_harm_fraction"]
        - EXPECTED_CM6_SAFE_POOLED["harm_recall"]
    ) > 1e-12:
        raise RuntimeError(
            "CM7B does not reproduce CM6 fixed-policy HARM recall."
        )
    if abs(
        safe["mean_patient_3d_deployed_dice"]
        - EXPECTED_CM6_SAFE_POOLED[
            "mean_patient_3d_deployed_dice"
        ]
    ) > 1e-12:
        raise RuntimeError(
            "CM7B does not reproduce CM6 deployed 3D Dice."
        )

    source_only = source_only_metrics(df)
    tent_all = tent_all_metrics(df)

    family_k = family_exact_k(
        df,
        safe_adapt,
    )

    print("\n===== CM7B FROZEN POLICY REPRODUCTION =====")
    print("threshold:", tau)
    print("family exact matched coverage:")
    print(json.dumps(family_k, indent=2))
    print("SafeTTA:", json.dumps(safe, indent=2))
    print("SOURCE-only:", json.dumps(source_only, indent=2))
    print("TENT-all:", json.dumps(tent_all, indent=2))
    print("CM6 reproduction: PASS")

    rnd_df, rnd_summary = (
        full_target_randomization_test(
            df,
            safe,
            family_k,
        )
    )

    rnd_path = (
        args.output_dir
        / "CM7B_FULL_TARGET_MATCHED_RANDOMIZATION_REPLICATES.csv"
    )
    rnd_summary_path = (
        args.output_dir
        / "CM7B_FULL_TARGET_MATCHED_RANDOMIZATION_SUMMARY.csv"
    )
    rnd_df.to_csv(rnd_path, index=False)
    rnd_summary.to_csv(
        rnd_summary_path,
        index=False,
    )

    print("\n===== CM7B FULL-TARGET RANDOM MATCHED-COVERAGE SUMMARY =====")
    print(rnd_summary.to_string(index=False))

    boot_df, boot_summary = (
        paired_patient_bootstrap(
            df,
            tau,
        )
    )

    boot_path = (
        args.output_dir
        / "CM7B_PAIRED_PATIENT_BOOTSTRAP_REPLICATES.csv"
    )
    boot_summary_path = (
        args.output_dir
        / "CM7B_PAIRED_PATIENT_BOOTSTRAP_SUMMARY.csv"
    )
    boot_df.to_csv(boot_path, index=False)
    boot_summary.to_csv(
        boot_summary_path,
        index=False,
    )

    print("\n===== CM7B PAIRED PATIENT-BOOTSTRAP SAFETTA - RANDOM =====")
    print(boot_summary.to_string(index=False))

    point_summary = {
        "SOURCE_ONLY": source_only,
        "TENT_ALL": tent_all,
        "SAFETTA_FIXED_SOURCE_GATE": safe,
        "family_matched_coverage": family_k,
    }
    point_path = (
        args.output_dir
        / "CM7B_POINT_UTILITY_SUMMARY.json"
    )
    save_json(point_path, point_summary)

    lock = {
        "status": "PASS",
        "decision": PASS_DECISION,
        "version": VERSION,
        "build": BUILD,
        "scope": "POST_GT_FIXED_PROTOCOL_UTILITY_AUDIT",
        "cm6_lock_sha256": EXPECTED_CM6_LOCK_SHA,
        "cm7a_lock_sha256": EXPECTED_CM7A_LOCK_SHA,
        "frozen_policy": {
            "threshold": tau,
            "rule": (
                "risk_score >= threshold => retain SOURCE; "
                "risk_score < threshold => adapt"
            ),
            "cm6_exact_reproduction": True,
        },
        "matched_random_baseline": {
            "matching": (
                "exact adapted-row count per model family"
            ),
            "full_target_randomization_reps": (
                FULL_RANDOM_REPS
            ),
            "patient_bootstrap_reps": (
                BOOTSTRAP_REPS
            ),
            "inner_random_draws_per_bootstrap_rep": (
                BOOTSTRAP_INNER_RANDOM_DRAWS
            ),
        },
        "point_results": point_summary,
        "interpretation_boundary": (
            "CM7B is a post-GT fixed-protocol utility audit. "
            "It tests whether the prospectively frozen CM6 ranking gate "
            "adds utility beyond random selection at exactly matched "
            "family-wise adaptation coverage. No target policy is changed."
        ),
        "changes_after_gt": {
            "threshold": False,
            "model": False,
            "representation": False,
            "harm_definition": False,
        },
        "artifacts": {
            "point_summary": str(point_path),
            "point_summary_sha256": sha256_file(
                point_path
            ),
            "full_random_replicates": str(
                rnd_path
            ),
            "full_random_replicates_sha256": (
                sha256_file(rnd_path)
            ),
            "full_random_summary": str(
                rnd_summary_path
            ),
            "full_random_summary_sha256": (
                sha256_file(rnd_summary_path)
            ),
            "patient_bootstrap_replicates": str(
                boot_path
            ),
            "patient_bootstrap_replicates_sha256": (
                sha256_file(boot_path)
            ),
            "patient_bootstrap_summary": str(
                boot_summary_path
            ),
            "patient_bootstrap_summary_sha256": (
                sha256_file(boot_summary_path)
            ),
        },
        "next_stage": (
            "FINAL_CROSS_DATASET_SYNTHESIS_AND_MANUSCRIPT_UPDATE"
        ),
    }

    lock_path = (
        args.output_dir
        / "CM7B_PROMISE12_MATCHED_COVERAGE_UTILITY_AUDIT_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM7B FINAL =====")
    print("Prospective endpoint: NO (CM6 remains primary)")
    print(
        "Random baseline exact family-wise coverage match: YES"
    )
    print("Threshold change: NO")
    print("Model change: NO")
    print("HARM definition change: NO")
    print("Decision=", PASS_DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
