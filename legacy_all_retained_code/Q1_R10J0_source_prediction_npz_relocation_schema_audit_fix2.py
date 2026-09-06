
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2.py

Purpose
-------
R10J0 Fix1 proved the R05D3 <-> R05D4 explicit state lineage is intact, but
all 9 manifest NPZ paths point to a historical "__building" directory that no
longer exists.

Fix2 performs a non-destructive relocation audit:

1) Read the frozen R05D3 manifest.
2) Extract the 9 expected NPZ basenames.
3) Recursively search F:/MEDSEG_SAFETTA/outputs for exact basename matches.
4) If an expected basename has:
   - one exact match: accept it;
   - multiple exact matches with identical SHA256: accept a deterministic
     canonical copy and record all equivalent copies;
   - multiple different SHA256 files: STOP as ambiguous;
   - no exact match: record family/seed fuzzy candidates and STOP.
5) Only after all 9 assets are safely recovered:
   - write a NEW recovered manifest (never overwrite R05D3);
   - inspect all 9 NPZ schemas;
   - print keys / shapes / dtypes / ranges;
   - identify source-only prediction-array candidates by key name.

AUDIT ONLY
----------
- no model training
- no threshold tuning
- no paper feature engineering
- no row-order matching
- no modification of the frozen R05D3 manifest
"""

import argparse
from pathlib import Path
import hashlib
import json
import os
import re

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

DEFAULT_SEARCH_ROOT = "F:/MEDSEG_SAFETTA/outputs"

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2_v1"
)


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    pred_tokens = [
        "prob", "logit", "pred", "mask", "seg",
    ]

    if any(t in k for t in forbidden_tokens):
        return "FORBIDDEN_FOR_PRE_ADAPTATION_FEATURES"
    if any(t in k for t in source_tokens) and any(t in k for t in pred_tokens):
        return "STRONG_SOURCE_CANDIDATE"
    if any(t in k for t in source_tokens):
        return "SOURCE_NAMED_CANDIDATE"
    if any(t in k for t in pred_tokens):
        return "AMBIGUOUS_PREDICTION_CANDIDATE"
    return "OTHER"


def safe_array_summary(arr):
    x = np.asarray(arr)
    row = {
        "shape": "x".join(map(str, x.shape)),
        "ndim": int(x.ndim),
        "dtype": str(x.dtype),
        "size": int(x.size),
    }

    if x.size == 0 or not np.issubdtype(x.dtype, np.number):
        row.update({
            "finite_fraction": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        })
        return row

    xf = x.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(xf)
    row["finite_fraction"] = float(finite.mean())

    if finite.any():
        v = xf[finite]
        row["min"] = float(v.min())
        row["max"] = float(v.max())
        row["mean"] = float(v.mean())
        row["std"] = float(v.std())
    else:
        row["min"] = np.nan
        row["max"] = np.nan
        row["mean"] = np.nan
        row["std"] = np.nan

    return row


def expected_tokens(basename):
    """
    Example:
      deeplabv3_r50_seed20260817_predictions.npz
    -> family token 'deeplabv3r50', seed '20260817'
    """
    stem = Path(basename).stem.lower()
    m = re.search(r"seed(\d+)", stem)
    seed = m.group(1) if m else ""

    fam = stem.split("_seed")[0]
    fam_norm = re.sub(r"[^a-z0-9]+", "", fam)
    return fam_norm, seed


def normalized_name(path):
    return re.sub(r"[^a-z0-9]+", "", Path(path).stem.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    ap.add_argument("--search_root", default=DEFAULT_SEARCH_ROOT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    outcomes_path = Path(args.outcomes)
    search_root = Path(args.search_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10J0 SOURCE-PREDICTION NPZ RELOCATION + SCHEMA AUDIT FIX2 =====")
    print("STATUS: NON-DESTRUCTIVE ASSET RECOVERY AUDIT")
    print("MODEL TRAINING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("FEATURE ENGINEERING: NONE")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("FROZEN R05D3 MANIFEST MODIFICATION: FORBIDDEN")
    print("SEARCH ROOT:", search_root)
    print()

    print("manifest exists:", manifest_path.exists(), "path:", manifest_path)
    print("outcomes exists:", outcomes_path.exists(), "path:", outcomes_path)
    print("search root exists:", search_root.exists(), "path:", search_root)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)
    if not search_root.exists():
        raise FileNotFoundError(search_root)

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

    print("\n===== EXPLICIT-KEY LINEAGE RECHECK =====")
    print("Manifest rows:", len(man))
    print("Outcome rows:", len(outc))
    print("Manifest unique sample+state:", man[key].drop_duplicates().shape[0])
    print("Outcome unique sample+state:", outc[key].drop_duplicates().shape[0])

    if len(man) != 9000 or len(outc) != 9000:
        raise AssertionError("Expected 9000 rows in both files.")
    if man[key].drop_duplicates().shape[0] != 9000:
        raise AssertionError("Manifest explicit key not unique.")
    if outc[key].drop_duplicates().shape[0] != 9000:
        raise AssertionError("Outcome explicit key not unique.")

    mkeys = set(map(tuple, man[key].astype(str).to_numpy()))
    okeys = set(map(tuple, outc[key].astype(str).to_numpy()))
    print("Explicit key sets equal:", mkeys == okeys)
    if mkeys != okeys:
        raise AssertionError("Explicit key-set mismatch.")

    x = man[
        key + ["model_family", "training_seed", "checkpoint_sha256"]
    ].merge(
        outc[
            key + ["model_family", "training_seed", "checkpoint_sha256"]
        ],
        on=key,
        validate="one_to_one",
        suffixes=("_d3", "_d4"),
    )

    fam_ok = x["model_family_d3"].astype(str).eq(x["model_family_d4"].astype(str))
    seed_ok = x["training_seed_d3"].astype(str).eq(x["training_seed_d4"].astype(str))
    sha_ok = x["checkpoint_sha256_d3"].astype(str).eq(x["checkpoint_sha256_d4"].astype(str))

    print("model_family agreement:", int(fam_ok.sum()), "/", len(x))
    print("training_seed agreement:", int(seed_ok.sum()), "/", len(x))
    print("checkpoint agreement:", int(sha_ok.sum()), "/", len(x))

    if not (fam_ok.all() and seed_ok.all() and sha_ok.all()):
        raise AssertionError("R05D3/R05D4 state metadata mismatch.")

    # Expected 9 state-level NPZ basenames.
    state_assets = (
        man[
            ["model_state_id", "model_family", "training_seed", "state_prediction_npz"]
        ]
        .drop_duplicates()
        .copy()
    )

    if len(state_assets) != 9:
        raise AssertionError(f"Expected 9 unique state assets, got {len(state_assets)}.")

    state_assets["expected_basename"] = state_assets["state_prediction_npz"].map(
        lambda s: Path(str(s)).name
    )

    print("\n===== EXPECTED STATE ASSETS =====")
    print(
        state_assets[
            ["model_state_id", "model_family", "training_seed", "expected_basename"]
        ].to_string(index=False)
    )

    target_names = set(state_assets["expected_basename"].tolist())

    # --------------------------------------------------------------
    # Single filesystem traversal:
    # - exact filename hits for the 9 expected assets
    # - all NPZ files for fuzzy family/seed fallback diagnostics
    # --------------------------------------------------------------
    exact_hits = {name: [] for name in target_names}
    all_npz = []

    print("\n===== FILESYSTEM SEARCH =====")
    for root, dirs, files in tqdm(
        os.walk(search_root),
        desc="Scanning output directories",
        unit="dir",
    ):
        # Skip obvious irrelevant caches if they occur.
        dirs[:] = [
            d for d in dirs
            if d not in {"__pycache__", ".git", ".ipynb_checkpoints"}
        ]

        for fn in files:
            if not fn.lower().endswith(".npz"):
                continue

            p = Path(root) / fn
            all_npz.append(str(p))

            if fn in exact_hits:
                exact_hits[fn].append(str(p))

    print("Total NPZ files found under search root:", len(all_npz))

    # --------------------------------------------------------------
    # Evaluate exact hits + hashes.
    # --------------------------------------------------------------
    relocation_rows = []
    chosen_by_basename = {}
    hard_fail = False

    for _, r in state_assets.iterrows():
        basename = r["expected_basename"]
        hits = exact_hits.get(basename, [])

        row_base = {
            "model_state_id": r["model_state_id"],
            "model_family": r["model_family"],
            "training_seed": r["training_seed"],
            "expected_basename": basename,
            "exact_match_count": len(hits),
        }

        if len(hits) == 0:
            fam_norm, seed = expected_tokens(basename)
            fuzzy = []

            for p in all_npz:
                n = normalized_name(p)
                if seed and seed not in n:
                    continue
                if fam_norm and fam_norm not in n:
                    continue
                fuzzy.append(p)

            relocation_rows.append({
                **row_base,
                "status": "MISSING_EXACT",
                "chosen_path": "",
                "chosen_sha256": "",
                "all_exact_paths": "",
                "fuzzy_candidates": ";".join(fuzzy[:100]),
            })

            print("\nMISSING EXACT:", basename)
            print(" fuzzy candidates:", len(fuzzy))
            for q in fuzzy[:20]:
                print("   ", q)

            hard_fail = True
            continue

        hashes = []
        for p in hits:
            h = sha256_file(p)
            hashes.append((p, h))

        unique_hashes = sorted(set(h for _, h in hashes))

        if len(hits) == 1:
            chosen = hits[0]
            chosen_hash = hashes[0][1]
            status = "UNIQUE_EXACT"
        elif len(unique_hashes) == 1:
            # Byte-identical copies: deterministic canonical path.
            chosen = sorted(hits, key=lambda s: (len(s), s.lower()))[0]
            chosen_hash = unique_hashes[0]
            status = "MULTIPLE_BYTE_IDENTICAL"
        else:
            chosen = ""
            chosen_hash = ""
            status = "AMBIGUOUS_DIFFERENT_CONTENT"
            hard_fail = True

        relocation_rows.append({
            **row_base,
            "status": status,
            "chosen_path": chosen,
            "chosen_sha256": chosen_hash,
            "all_exact_paths": ";".join(hits),
            "fuzzy_candidates": "",
        })

        print(
            f"{basename}: exact={len(hits)} "
            f"unique_sha256={len(unique_hashes)} status={status}"
        )
        if chosen:
            print(" chosen:", chosen)

        if chosen:
            chosen_by_basename[basename] = chosen

    relocation = pd.DataFrame(relocation_rows)
    relocation_path = out / "R10J0_fix2_relocation_audit.csv"
    relocation.to_csv(relocation_path, index=False)

    print("\n===== RELOCATION SUMMARY =====")
    print(
        relocation[
            [
                "model_state_id",
                "expected_basename",
                "exact_match_count",
                "status",
                "chosen_path",
            ]
        ].to_string(index=False)
    )

    if hard_fail or len(chosen_by_basename) != 9:
        print("\n===== FINAL DECISION =====")
        print("DECISION: STATE_PREDICTION_ASSET_RELOCATION_NOT_RESOLVED")
        print("Recovered safe assets:", len(chosen_by_basename), "/ 9")
        print("Do NOT regenerate or infer NPZ contents yet.")
        print("Audit output:", relocation_path)
        print("PASS: SAFE STOP")
        return

    # --------------------------------------------------------------
    # Build NEW recovered manifest only.
    # --------------------------------------------------------------
    recovered = man.drop(columns=["_sid"]).copy()
    recovered["state_prediction_npz_original"] = recovered["state_prediction_npz"]
    recovered["state_prediction_npz"] = recovered["state_prediction_npz"].map(
        lambda s: chosen_by_basename[Path(str(s)).name]
    )

    recovered_manifest_path = (
        out / "model_case_prediction_manifest_relocated_fix2.csv"
    )
    recovered.to_csv(recovered_manifest_path, index=False)

    print("\n===== RECOVERED MANIFEST =====")
    print("Rows:", len(recovered))
    print("Unique state assets:", recovered["state_prediction_npz"].nunique())
    print("Original R05D3 overwritten: NO")
    print("Recovered manifest:", recovered_manifest_path)

    # --------------------------------------------------------------
    # Inspect ALL 9 state-level NPZ files.
    # --------------------------------------------------------------
    schema_rows = []
    file_rows = []

    unique_state_rows = (
        recovered[
            [
                "model_state_id",
                "model_family",
                "training_seed",
                "state_prediction_npz",
            ]
        ]
        .drop_duplicates()
        .sort_values(["model_family", "training_seed"])
    )

    print("\n===== NPZ SCHEMA INSPECTION =====")

    for _, r in tqdm(
        unique_state_rows.iterrows(),
        total=len(unique_state_rows),
        desc="Inspecting 9 NPZ files",
    ):
        p = Path(r["state_prediction_npz"])

        with np.load(p, allow_pickle=False) as z:
            keys = list(z.files)

            file_rows.append({
                "model_state_id": r["model_state_id"],
                "model_family": r["model_family"],
                "training_seed": r["training_seed"],
                "npz_path": str(p),
                "sha256": sha256_file(p),
                "key_count": len(keys),
                "keys": ";".join(keys),
            })

            print("\nSTATE:", r["model_state_id"])
            print("PATH:", p)
            print("KEYS:", keys)

            for k in keys:
                arr = z[k]
                stat = safe_array_summary(arr)
                row = {
                    "model_state_id": r["model_state_id"],
                    "model_family": r["model_family"],
                    "training_seed": r["training_seed"],
                    "npz_path": str(p),
                    "key": k,
                    "key_class": classify_key_name(k),
                    **stat,
                }
                schema_rows.append(row)

                print(
                    f"  key={k} class={row['key_class']} "
                    f"shape={row['shape']} dtype={row['dtype']} "
                    f"min={row['min']} max={row['max']}"
                )

    schema = pd.DataFrame(schema_rows)
    files_df = pd.DataFrame(file_rows)

    key_summary = (
        schema.groupby(["key", "key_class"], dropna=False)
        .agg(
            state_files_seen=("model_state_id", "nunique"),
            families_seen=("model_family", "nunique"),
            shapes=("shape", lambda s: ";".join(sorted(set(map(str, s))))),
            dtypes=("dtype", lambda s: ";".join(sorted(set(map(str, s))))),
            min_observed=("min", "min"),
            max_observed=("max", "max"),
        )
        .reset_index()
        .sort_values(["key_class", "key"])
    )

    candidate = key_summary[
        key_summary["key_class"].isin([
            "STRONG_SOURCE_CANDIDATE",
            "SOURCE_NAMED_CANDIDATE",
            "AMBIGUOUS_PREDICTION_CANDIDATE",
        ])
    ].copy()

    forbidden = key_summary[
        key_summary["key_class"]
        == "FORBIDDEN_FOR_PRE_ADAPTATION_FEATURES"
    ].copy()

    print("\n===== NPZ KEY SUMMARY =====")
    print(key_summary.to_string(index=False))

    print("\n===== SOURCE-ONLY ARRAY CANDIDATES =====")
    if len(candidate):
        print(candidate.to_string(index=False))
    else:
        print("NONE")

    print("\n===== FORBIDDEN / POST-ADAPTATION / GT-LIKE KEYS =====")
    if len(forbidden):
        print(forbidden.to_string(index=False))
    else:
        print("NONE")

    files_df.to_csv(out / "R10J0_fix2_recovered_npz_files.csv", index=False)
    schema.to_csv(out / "R10J0_fix2_npz_schema_rows.csv", index=False)
    key_summary.to_csv(out / "R10J0_fix2_npz_key_summary.csv", index=False)
    candidate.to_csv(out / "R10J0_fix2_source_array_candidates.csv", index=False)
    forbidden.to_csv(out / "R10J0_fix2_forbidden_array_keys.csv", index=False)

    summary = {
        "status": "PASS",
        "relocated_assets": 9,
        "original_manifest_overwritten": False,
        "recovered_manifest": str(recovered_manifest_path),
        "source_candidate_keys": candidate["key"].astype(str).tolist(),
        "forbidden_keys": forbidden["key"].astype(str).tolist(),
    }

    with open(out / "R10J0_fix2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    if len(candidate) == 0:
        print("DECISION: NPZ_ASSETS_RECOVERED_BUT_SOURCE_ARRAY_KEY_UNRESOLVED")
    else:
        print("DECISION: SOURCE_PREDICTION_ASSET_SCHEMA_RECOVERED")
        print(
            "Next step may freeze exact pre-adaptation source array key(s) "
            "and build richer standard morphology features."
        )

    print("\nOutput:", out)
    print("PASS")


if __name__ == "__main__":
    main()
