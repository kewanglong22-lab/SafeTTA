
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10J0_source_prediction_npz_schema_audit_fix1.py

Purpose
-------
R10I2 established a genuine cross-model morphology ranking signal:
- macro target AUROC ~0.806
- worst-family AUROC ~0.768
- source-only R90 threshold transfers to recall ~0.899
- but macro FPR remains ~0.562
- oracle target R90 FPR is similarly weak, so threshold calibration is NOT the
  main bottleneck; the 2-D morphology representation is too coarse.

Before constructing richer morphology descriptors, audit the original
pre-adaptation prediction assets referenced by R05D3.

This script is AUDIT ONLY:
- no model training
- no threshold tuning
- no feature engineering
- no GT-based feature construction
- no A1/adapted prediction use
- no row-order matching

It validates explicit state lineage and inspects NPZ schemas so the next
morphology builder can be written against the actual stored arrays rather than
guessing key names or shapes.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1/"
    "model_case_prediction_manifest.csv"
)

DEFAULT_OUTCOMES = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_schema_audit_fix1_v1"
)

EXPECTED_FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def resolve_path(raw, manifest_path):
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve()


def safe_numeric_summary(arr):
    x = np.asarray(arr)
    out = {
        "shape": "x".join(str(v) for v in x.shape),
        "ndim": int(x.ndim),
        "dtype": str(x.dtype),
        "size": int(x.size),
    }

    if x.size == 0:
        out.update({
            "finite_fraction": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "unique_count_capped": 0,
        })
        return out

    if np.issubdtype(x.dtype, np.number):
        xf = x.astype(np.float64, copy=False).ravel()
        finite = np.isfinite(xf)
        out["finite_fraction"] = float(finite.mean())

        if finite.any():
            vals = xf[finite]
            out["min"] = float(vals.min())
            out["max"] = float(vals.max())
            out["mean"] = float(vals.mean())
            out["std"] = float(vals.std())

            if vals.size <= 200000:
                out["unique_count_capped"] = int(
                    min(np.unique(vals).size, 10001)
                )
            else:
                sample = vals[::max(1, vals.size // 200000)]
                out["unique_count_capped"] = int(
                    min(np.unique(sample).size, 10001)
                )
        else:
            out["min"] = np.nan
            out["max"] = np.nan
            out["mean"] = np.nan
            out["std"] = np.nan
            out["unique_count_capped"] = 0
    else:
        out.update({
            "finite_fraction": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "unique_count_capped": np.nan,
        })

    return out


def classify_key_name(key):
    k = key.lower()

    forbidden_tokens = [
        "gt", "ground_truth", "label", "dice",
        "a1", "adapt", "post", "after",
        "harm", "benefit", "outcome",
    ]

    source_tokens = [
        "source", "pre", "before", "base", "baseline",
    ]

    prob_tokens = [
        "prob", "logit", "pred", "mask", "seg",
    ]

    has_forbidden = any(t in k for t in forbidden_tokens)
    has_source = any(t in k for t in source_tokens)
    has_prediction = any(t in k for t in prob_tokens)

    if has_forbidden:
        return "FORBIDDEN_FOR_PRE_ADAPTATION_FEATURES"
    if has_source and has_prediction:
        return "STRONG_SOURCE_CANDIDATE"
    if has_source:
        return "SOURCE_NAMED_CANDIDATE"
    if has_prediction:
        return "AMBIGUOUS_PREDICTION_CANDIDATE"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--inspect_per_state",
        type=int,
        default=2,
        help="NPZ files inspected per model_state_id after path audit.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    outcomes_path = Path(args.outcomes)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10J0 SOURCE-PREDICTION NPZ SCHEMA AUDIT FIX1 =====")
    print("STATUS: ASSET / LINEAGE AUDIT")
    print("MODEL TRAINING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("FEATURE ENGINEERING: NONE")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("GT-BASED FEATURE CONSTRUCTION: FORBIDDEN")
    print("A1/ADAPTED PREDICTION USE FOR NEXT FEATURES: FORBIDDEN")
    print("EXPLICIT STATE KEY: sample_id + model_state_id\n")

    print("manifest exists:", manifest_path.exists(), "path:", manifest_path)
    print("outcomes exists:", outcomes_path.exists(), "path:", outcomes_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)

    man = pd.read_csv(manifest_path, low_memory=False)
    outc = pd.read_csv(outcomes_path, low_memory=False)

    required_manifest = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz",
    ]
    required_outcomes = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "harm_label",
    ]

    for c in required_manifest:
        if c not in man.columns:
            raise AssertionError(f"Manifest missing required column: {c}")
    for c in required_outcomes:
        if c not in outc.columns:
            raise AssertionError(f"Outcomes missing required column: {c}")

    man["_sid"] = norm_sid(man["sample_id"])
    outc["_sid"] = norm_sid(outc["sample_id"])

    key = ["_sid", "model_state_id"]

    print("\n===== EXPLICIT-KEY LINEAGE =====")
    print("Manifest rows:", len(man))
    print("Outcome rows:", len(outc))
    print("Manifest unique cases:", man["_sid"].nunique())
    print("Outcome unique cases:", outc["_sid"].nunique())
    print("Manifest unique states:", man["model_state_id"].nunique())
    print("Outcome unique states:", outc["model_state_id"].nunique())
    print(
        "Manifest unique sample+state:",
        man[key].drop_duplicates().shape[0],
        "/",
        len(man),
    )
    print(
        "Outcome unique sample+state:",
        outc[key].drop_duplicates().shape[0],
        "/",
        len(outc),
    )

    if len(man) != 9000 or len(outc) != 9000:
        raise AssertionError("Expected 9000 rows in both R05D3 and R05D4.")
    if man[key].drop_duplicates().shape[0] != len(man):
        raise AssertionError("Manifest explicit key is not unique.")
    if outc[key].drop_duplicates().shape[0] != len(outc):
        raise AssertionError("Outcome explicit key is not unique.")

    mkeys = set(map(tuple, man[key].astype(str).to_numpy()))
    okeys = set(map(tuple, outc[key].astype(str).to_numpy()))
    print("Explicit key sets equal:", mkeys == okeys)
    if mkeys != okeys:
        raise AssertionError("R05D3/R05D4 explicit key sets differ.")

    check = man[
        key + ["model_family", "training_seed", "checkpoint_sha256"]
    ].merge(
        outc[
            key + ["model_family", "training_seed", "checkpoint_sha256"]
        ],
        on=key,
        validate="one_to_one",
        suffixes=("_d3", "_d4"),
    )

    fam_ok = (
        check["model_family_d3"].astype(str)
        == check["model_family_d4"].astype(str)
    )
    seed_ok = (
        check["training_seed_d3"].astype(str)
        == check["training_seed_d4"].astype(str)
    )
    sha_ok = (
        check["checkpoint_sha256_d3"].astype(str)
        == check["checkpoint_sha256_d4"].astype(str)
    )

    print("model_family agreement:", int(fam_ok.sum()), "/", len(check))
    print("training_seed agreement:", int(seed_ok.sum()), "/", len(check))
    print("checkpoint agreement:", int(sha_ok.sum()), "/", len(check))

    if not (fam_ok.all() and seed_ok.all() and sha_ok.all()):
        raise AssertionError("Explicit-state metadata cross-check failed.")

    print("\nFamilies:")
    print(man["model_family"].value_counts().to_string())

    if set(man["model_family"].astype(str).unique()) != set(EXPECTED_FAMILIES):
        raise AssertionError("Unexpected model-family set.")

    # --------------------------------------------------------------
    # Resolve and audit every NPZ path, but do not open every file.
    # --------------------------------------------------------------
    resolved = []
    exists_flags = []

    print("\n===== NPZ PATH AUDIT =====")
    for raw in tqdm(
        man["state_prediction_npz"].astype(str),
        total=len(man),
        desc="Checking NPZ paths",
    ):
        p = resolve_path(raw, manifest_path)
        resolved.append(str(p))
        exists_flags.append(p.exists())

    man["_resolved_npz"] = resolved
    man["_npz_exists"] = exists_flags

    print("Rows with existing NPZ:", int(man["_npz_exists"].sum()), "/", len(man))
    print("Unique NPZ paths:", man["_resolved_npz"].nunique())
    print(
        "Unique existing NPZ paths:",
        man.loc[man["_npz_exists"], "_resolved_npz"].nunique(),
    )

    if not man["_npz_exists"].all():
        missing = man.loc[
            ~man["_npz_exists"],
            ["sample_id", "model_state_id", "state_prediction_npz", "_resolved_npz"],
        ]
        missing.to_csv(out / "R10J0_missing_npz_paths.csv", index=False)
        print("Missing NPZ examples:")
        print(missing.head(30).to_string(index=False))
        raise AssertionError("Some state_prediction_npz paths do not exist.")

    # --------------------------------------------------------------
    # Deterministic inspection sample: first N files per state ID.
    # --------------------------------------------------------------
    inspect_rows = []

    for state_id, g in man.groupby("model_state_id", sort=True):
        gg = g.sort_values(["_sid", "_resolved_npz"])
        inspect_rows.append(gg.head(args.inspect_per_state))

    inspect_df = pd.concat(inspect_rows, ignore_index=True)

    print("\nFiles selected for schema inspection:", len(inspect_df))
    print("States covered:", inspect_df["model_state_id"].nunique())

    schema_rows = []
    file_rows = []

    for _, r in tqdm(
        inspect_df.iterrows(),
        total=len(inspect_df),
        desc="Inspecting NPZ schema",
    ):
        p = Path(r["_resolved_npz"])

        with np.load(p, allow_pickle=False) as z:
            keys = list(z.files)

            file_rows.append({
                "sample_id": r["sample_id"],
                "model_family": r["model_family"],
                "model_state_id": r["model_state_id"],
                "training_seed": r["training_seed"],
                "npz_path": str(p),
                "key_count": len(keys),
                "keys": ";".join(keys),
            })

            for k in keys:
                arr = z[k]
                stat = safe_numeric_summary(arr)
                schema_rows.append({
                    "sample_id": r["sample_id"],
                    "model_family": r["model_family"],
                    "model_state_id": r["model_state_id"],
                    "npz_path": str(p),
                    "key": k,
                    "key_class": classify_key_name(k),
                    **stat,
                })

    schema = pd.DataFrame(schema_rows)
    files = pd.DataFrame(file_rows)

    print("\n===== NPZ KEY SUMMARY =====")
    key_summary = (
        schema.groupby(["key", "key_class"], dropna=False)
        .agg(
            files_seen=("npz_path", "nunique"),
            states_seen=("model_state_id", "nunique"),
            families_seen=("model_family", "nunique"),
            ndim_values=("ndim", lambda s: ";".join(map(str, sorted(set(s))))),
            shape_values=("shape", lambda s: ";".join(sorted(set(map(str, s))))),
            dtype_values=("dtype", lambda s: ";".join(sorted(set(map(str, s))))),
            min_observed=("min", "min"),
            max_observed=("max", "max"),
            unique_count_max=("unique_count_capped", "max"),
        )
        .reset_index()
        .sort_values(["key_class", "key"])
    )

    print(key_summary.to_string(index=False))

    candidate = key_summary[
        key_summary["key_class"].isin([
            "STRONG_SOURCE_CANDIDATE",
            "SOURCE_NAMED_CANDIDATE",
            "AMBIGUOUS_PREDICTION_CANDIDATE",
        ])
    ].copy()

    forbidden = key_summary[
        key_summary["key_class"] == "FORBIDDEN_FOR_PRE_ADAPTATION_FEATURES"
    ].copy()

    print("\n===== SOURCE-ONLY ARRAY CANDIDATES =====")
    if len(candidate):
        print(candidate.to_string(index=False))
    else:
        print("NONE DETECTED BY KEY NAME")

    print("\n===== FORBIDDEN / POST-ADAPTATION / GT-LIKE KEYS =====")
    if len(forbidden):
        print(forbidden.to_string(index=False))
    else:
        print("NONE DETECTED BY KEY NAME")

    # Save.
    man[
        [
            "sample_id",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "state_prediction_npz",
            "_resolved_npz",
            "_npz_exists",
        ]
    ].to_csv(out / "R10J0_npz_path_audit.csv", index=False)

    files.to_csv(out / "R10J0_inspected_files.csv", index=False)
    schema.to_csv(out / "R10J0_npz_array_schema_rows.csv", index=False)
    key_summary.to_csv(out / "R10J0_npz_key_summary.csv", index=False)
    candidate.to_csv(out / "R10J0_source_array_candidates.csv", index=False)
    forbidden.to_csv(out / "R10J0_forbidden_array_keys.csv", index=False)

    summary = {
        "manifest_rows": int(len(man)),
        "unique_cases": int(man["_sid"].nunique()),
        "unique_model_states": int(man["model_state_id"].nunique()),
        "all_npz_exist": bool(man["_npz_exists"].all()),
        "unique_npz_paths": int(man["_resolved_npz"].nunique()),
        "inspected_files": int(len(files)),
        "unique_keys_seen": sorted(schema["key"].astype(str).unique().tolist()),
        "source_candidate_keys": candidate["key"].astype(str).tolist(),
        "forbidden_keys": forbidden["key"].astype(str).tolist(),
    }

    with open(out / "R10J0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    if len(candidate) == 0:
        print("DECISION: SOURCE_PREDICTION_ARRAY_NOT_IDENTIFIED")
        print(
            "Do not build morphology features until the actual pre-adaptation "
            "array key is identified."
        )
    else:
        print("DECISION: SOURCE_PREDICTION_ASSET_SCHEMA_RECOVERED")
        print(
            "Candidate source-only prediction arrays were identified by schema. "
            "Next step: freeze the exact source array key(s) and construct "
            "standard morphology/topology descriptors only from those arrays."
        )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
