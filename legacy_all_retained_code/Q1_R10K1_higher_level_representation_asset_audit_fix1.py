
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K1_higher_level_representation_asset_audit_fix1.py

Purpose
-------
R10K0 falsified the output-mask-only spatial representation route:
SpatialPCA32 and M2+SpatialPCA32 both degraded cross-model safety ranking.

Before moving to higher-level image-conditioned / internal representations,
audit whether the exact frozen model checkpoints and the original NeoPolyp
input-image lineage are still recoverable.

AUDIT ONLY:
- no model loading
- no inference
- no representation extraction
- no fitting
- no threshold tuning
- no target-label use

Outputs:
1) exact checkpoint SHA256 recovery table for the 9 model states;
2) candidate image-manifest table ranked by exact sample_id coverage and
   existing image-path coverage;
3) explicit route-readiness decision.

No original manifest is modified.
"""

import argparse
from pathlib import Path
import hashlib
import os
import re
import json

import pandas as pd
from tqdm import tqdm


DEFAULT_STATE_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2_v1/"
    "model_case_prediction_manifest_relocated_fix2.csv"
)

DEFAULT_PROJECT_ROOT = "F:/MEDSEG_SAFETTA"

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K1_higher_level_representation_asset_audit_fix1_v1"
)

MODEL_EXTS = {
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors"
}

TABLE_EXTS = {
    ".csv", ".tsv"
}

PATH_HINTS = (
    "image", "img", "path", "file", "filename", "filepath",
    "rgb", "input"
)

EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv"
}


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


def resolve_candidate_path(raw, table_path, project_root):
    if pd.isna(raw):
        return None

    s = str(raw).strip().strip('"').strip("'")
    if not s:
        return None

    p = Path(s)

    if p.is_absolute():
        return p

    # First: table-relative path.
    q = (table_path.parent / p)
    if q.exists():
        return q

    # Second: project-root relative path.
    q2 = project_root / p
    return q2


def looks_path_like(series):
    """
    Lightweight content heuristic for a candidate path-like column.
    """
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
    )
    if len(vals) == 0:
        return False

    vals = vals.head(100)
    pattern = re.compile(
        r"(\.(png|jpg|jpeg|bmp|tif|tiff|webp|npy|npz)$)|([\\/])",
        flags=re.IGNORECASE,
    )
    hits = sum(bool(pattern.search(v)) for v in vals)
    return hits / len(vals) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_manifest", default=DEFAULT_STATE_MANIFEST)
    ap.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    state_manifest = Path(args.state_manifest)
    project_root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10K1 HIGHER-LEVEL REPRESENTATION ASSET AUDIT FIX1 =====")
    print("STATUS: ASSET / LINEAGE AUDIT")
    print("MODEL LOADING: NONE")
    print("INFERENCE: NONE")
    print("REPRESENTATION EXTRACTION: NONE")
    print("MODEL FITTING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("TARGET LABEL USE: NONE")
    print("ORIGINAL MANIFEST MODIFICATION: NONE")
    print()

    print("state manifest exists:", state_manifest.exists(), state_manifest)
    print("project root exists:", project_root.exists(), project_root)

    if not state_manifest.exists():
        raise FileNotFoundError(state_manifest)
    if not project_root.exists():
        raise FileNotFoundError(project_root)

    man = pd.read_csv(state_manifest, low_memory=False)

    required = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    ]
    missing = [c for c in required if c not in man.columns]
    if missing:
        raise AssertionError(f"State manifest missing columns: {missing}")

    man["_sid"] = norm_sid(man["sample_id"])

    if len(man) != 9000:
        raise AssertionError(f"Expected 9000 state rows, got {len(man)}.")
    if man["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 unique cases.")
    if man["model_state_id"].nunique() != 9:
        raise AssertionError("Expected 9 model states.")

    states = (
        man[
            [
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
            ]
        ]
        .drop_duplicates()
        .sort_values(["model_family", "training_seed"])
        .reset_index(drop=True)
    )

    if len(states) != 9:
        raise AssertionError("Expected 9 unique state metadata rows.")

    target_shas = set(
        states["checkpoint_sha256"]
        .astype(str)
        .str.lower()
        .tolist()
    )

    print("\n===== FROZEN MODEL STATES =====")
    print(states.to_string(index=False))

    # --------------------------------------------------------------
    # One filesystem traversal: collect model files and tabular manifests.
    # --------------------------------------------------------------
    model_candidates = []
    table_candidates = []

    print("\n===== PROJECT FILESYSTEM INVENTORY =====")
    for root, dirs, files in tqdm(
        os.walk(project_root),
        desc="Scanning project",
        unit="dir",
        dynamic_ncols=True,
    ):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDE_DIR_NAMES
        ]

        for fn in files:
            p = Path(root) / fn
            ext = p.suffix.lower()

            if ext in MODEL_EXTS:
                model_candidates.append(p)
            elif ext in TABLE_EXTS:
                table_candidates.append(p)

    print("Model-like files found:", len(model_candidates))
    print("CSV/TSV files found:", len(table_candidates))

    # --------------------------------------------------------------
    # Exact checkpoint recovery by frozen SHA256.
    # --------------------------------------------------------------
    print("\n===== CHECKPOINT SHA256 RECOVERY =====")

    sha_to_paths = {s: [] for s in target_shas}
    hashed_rows = []

    for p in tqdm(
        model_candidates,
        desc="Hashing model files",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            h = sha256_file(p).lower()
        except Exception as e:
            hashed_rows.append({
                "path": str(p),
                "sha256": "",
                "error": repr(e),
            })
            continue

        hashed_rows.append({
            "path": str(p),
            "sha256": h,
            "error": "",
        })

        if h in sha_to_paths:
            sha_to_paths[h].append(str(p))

    checkpoint_rows = []

    for _, r in states.iterrows():
        sha = str(r["checkpoint_sha256"]).lower()
        hits = sha_to_paths.get(sha, [])

        checkpoint_rows.append({
            "model_state_id": r["model_state_id"],
            "model_family": r["model_family"],
            "training_seed": r["training_seed"],
            "checkpoint_sha256": sha,
            "exact_sha256_match_count": len(hits),
            "exact_sha256_paths": ";".join(hits),
            "recovered": len(hits) >= 1,
        })

        print(
            f"{r['model_state_id']}: "
            f"exact_sha256_matches={len(hits)}"
        )
        for q in hits[:10]:
            print("  ", q)

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    checkpoint_pass = bool(checkpoint_df["recovered"].all())

    print(
        "Recovered frozen checkpoints:",
        int(checkpoint_df["recovered"].sum()),
        "/ 9",
    )

    # --------------------------------------------------------------
    # Candidate input-image manifest discovery.
    # Read headers first; only inspect files containing sample_id.
    # --------------------------------------------------------------
    print("\n===== IMAGE-LINEAGE TABLE DISCOVERY =====")

    source_case_set = set(man["_sid"].unique().tolist())
    image_manifest_rows = []

    for table_path in tqdm(
        table_candidates,
        desc="Auditing tables",
        unit="table",
        dynamic_ncols=True,
    ):
        sep = "\t" if table_path.suffix.lower() == ".tsv" else ","

        try:
            head = pd.read_csv(
                table_path,
                sep=sep,
                nrows=5,
                low_memory=False,
            )
        except Exception:
            continue

        cols = list(head.columns)
        sid_cols = [
            c for c in cols
            if c.lower() in {
                "sample_id", "case_id", "image_id", "id"
            }
        ]

        if not sid_cols:
            continue

        sid_col = (
            "sample_id"
            if "sample_id" in sid_cols
            else sid_cols[0]
        )

        named_path_cols = [
            c for c in cols
            if any(h in c.lower() for h in PATH_HINTS)
            and c != sid_col
        ]

        if not named_path_cols:
            continue

        usecols = [sid_col] + named_path_cols

        try:
            d = pd.read_csv(
                table_path,
                sep=sep,
                usecols=usecols,
                low_memory=False,
            )
        except Exception:
            continue

        if len(d) == 0:
            continue

        sid_norm = norm_sid(d[sid_col])
        overlap_mask = sid_norm.isin(source_case_set)
        overlap_rows = int(overlap_mask.sum())
        overlap_cases = int(sid_norm[overlap_mask].nunique())

        if overlap_cases == 0:
            continue

        for c in named_path_cols:
            if c not in d.columns:
                continue

            if not looks_path_like(d[c]):
                continue

            sub = d.loc[overlap_mask, [sid_col, c]].copy()
            sub["_sid"] = norm_sid(sub[sid_col])

            existing_rows = 0
            existing_cases = set()
            example_paths = []

            for _, rr in sub.iterrows():
                rp = resolve_candidate_path(
                    rr[c],
                    table_path,
                    project_root,
                )
                if rp is None:
                    continue

                if len(example_paths) < 5:
                    example_paths.append(str(rp))

                if rp.exists() and rp.is_file():
                    existing_rows += 1
                    existing_cases.add(rr["_sid"])

            image_manifest_rows.append({
                "table_path": str(table_path),
                "sample_id_column": sid_col,
                "image_path_column": c,
                "table_rows": len(d),
                "overlap_rows": overlap_rows,
                "overlap_unique_cases": overlap_cases,
                "coverage_fraction_1000": overlap_cases / 1000.0,
                "existing_path_rows": existing_rows,
                "existing_path_unique_cases": len(existing_cases),
                "existing_path_fraction_1000": len(existing_cases) / 1000.0,
                "example_resolved_paths": ";".join(example_paths),
            })

    if image_manifest_rows:
        image_df = pd.DataFrame(image_manifest_rows)
        image_df = image_df.sort_values(
            [
                "existing_path_unique_cases",
                "overlap_unique_cases",
            ],
            ascending=[False, False],
        ).reset_index(drop=True)
    else:
        image_df = pd.DataFrame(columns=[
            "table_path",
            "sample_id_column",
            "image_path_column",
            "table_rows",
            "overlap_rows",
            "overlap_unique_cases",
            "coverage_fraction_1000",
            "existing_path_rows",
            "existing_path_unique_cases",
            "existing_path_fraction_1000",
            "example_resolved_paths",
        ])

    print("\nTop image-lineage candidates:")
    if len(image_df):
        print(
            image_df.head(20).to_string(index=False)
        )
    else:
        print("NONE FOUND")

    image_pass = bool(
        len(image_df)
        and image_df.iloc[0]["existing_path_unique_cases"] == 1000
    )

    # --------------------------------------------------------------
    # Decisions.
    # --------------------------------------------------------------
    print("\n===== ROUTE READINESS =====")
    print("Exact frozen checkpoints recovered:", checkpoint_pass)
    print("1000-case input-image lineage recovered:", image_pass)

    if checkpoint_pass and image_pass:
        decision = "INTERNAL_AND_IMAGE_CONDITIONED_ROUTES_READY"
    elif image_pass:
        decision = "IMAGE_CONDITIONED_ROUTE_READY_CHECKPOINT_ROUTE_UNRESOLVED"
    elif checkpoint_pass:
        decision = "CHECKPOINT_ROUTE_READY_INPUT_IMAGE_LINEAGE_UNRESOLVED"
    else:
        decision = "HIGHER_LEVEL_ASSETS_NOT_YET_READY"

    print("DECISION:", decision)

    checkpoint_df.to_csv(
        out / "R10K1_checkpoint_recovery.csv",
        index=False,
    )
    pd.DataFrame(hashed_rows).to_csv(
        out / "R10K1_hashed_model_files.csv",
        index=False,
    )
    image_df.to_csv(
        out / "R10K1_image_lineage_candidates.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "checkpoint_pass": checkpoint_pass,
        "image_lineage_pass": image_pass,
        "states": 9,
        "cases": 1000,
        "model_like_files_scanned": len(model_candidates),
        "tables_scanned": len(table_candidates),
    }

    with open(
        out / "R10K1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nOutputs:")
    print(out / "R10K1_checkpoint_recovery.csv")
    print(out / "R10K1_image_lineage_candidates.csv")
    print(out / "R10K1_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
