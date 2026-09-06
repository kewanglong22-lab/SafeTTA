#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10L2A_source_training_lineage_discovery_audit_fix1.py

Purpose
-------
Before any PolypGen inference, recover the ACTUAL source-training lineage of
the nine frozen segmentation model states.

This is a provenance / leakage audit only.

Inputs
------
1) R10K1 exact checkpoint-recovery table (9 frozen states)
2) R10L1B frozen 1532-case unique PolypGen static manifest
3) MEDSEG_SAFETTA project root

Audit strategy
--------------
A. Recover each frozen checkpoint path and derive its training-run directory.
B. Inspect only training-run artifacts and matching code artifacts for:
   - explicit dataset/data-root references;
   - manifest-like CSV/TSV tables containing image paths;
   - train/val split indicators;
   - mentions of PolypGen / NeoPolyp.
C. For any EXPLICIT train-image manifest rows whose files still exist:
   - compute exact SHA256;
   - compare against the frozen PolypGen RGB hashes.
D. Never infer training membership from directory proximity alone.

No model inference.
No external performance.
No TTA.
No safety scoring.
No threshold tuning.
No method tuning.

Important decision boundary
---------------------------
Zero overlap is only considered confirmed for a model state when an explicit
source-training image manifest/path list is recovered for that state.

If historical artifacts do not expose the exact train-image list, the script
returns LINEAGE_INCOMPLETE rather than guessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import os
import re

import pandas as pd
from tqdm import tqdm


DEFAULT_CHECKPOINT_RECOVERY = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K1_higher_level_representation_asset_audit_fix1_v1/"
    "R10K1_checkpoint_recovery.csv"
)

DEFAULT_POLYPGEN_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1/"
    "R10L1B_POLYPGEN_UNIQUE_STATIC_MANIFEST.csv"
)

DEFAULT_POLYPGEN_LOCK = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L1B_polypgen_unique_static_manifest_lock_fix3_v1/"
    "R10L1B_MANIFEST_LOCK.json"
)

DEFAULT_PROJECT_ROOT = "F:/MEDSEG_SAFETTA"

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10L2A_source_training_lineage_discovery_audit_fix1_v1"
)

RASTER_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

TEXT_EXTS = {
    ".txt", ".log", ".md", ".json", ".yaml", ".yml",
    ".py", ".ini", ".cfg", ".toml", ".ps1"
}

TABLE_EXTS = {".csv", ".tsv"}

PATH_COLUMNS = {
    "image_path", "img_path", "image", "img",
    "filepath", "file_path", "path", "filename", "file"
}

SPLIT_COLUMNS = {
    "split", "subset", "phase", "partition",
    "set", "mode", "stage"
}

TRAIN_TOKENS = {
    "train", "training", "source_train", "source-training",
    "source_training"
}

PROVENANCE_KEYWORDS = [
    "dataset",
    "data_root",
    "dataset_root",
    "image_root",
    "images_root",
    "mask_root",
    "train_root",
    "train_dir",
    "train_path",
    "source_root",
    "source_data",
    "source_dataset",
    "manifest",
    "train",
    "val",
    "validation",
    "polypgen",
    "neopolyp",
    "kvasir",
    "clinicdb",
    "cvc",
    "etis",
    "colon",
    "polyp",
]

EXCLUDE_SCAN_DIRS = {
    "__pycache__", ".git", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv"
}

MAX_TEXT_BYTES = 10 * 1024 * 1024


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def derive_run_root(checkpoint_path):
    p = Path(checkpoint_path)
    parent = p.parent

    if parent.name.lower() == "checkpoints":
        return parent.parent

    if parent.name.lower().startswith("seed_"):
        return parent.parent

    return parent


def read_text_safely(path):
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
    except Exception:
        return None

    for enc in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return None


def absolute_f_paths(text):
    """
    Extract explicit F:/... or F:\\... path-like strings.
    Conservative: stop at quotes, CR/LF, angle brackets, semicolon.
    """
    pat = re.compile(
        r"""(?i)(F:[\\/][^"'<>;\r\n]+)"""
    )
    vals = []
    for m in pat.finditer(text):
        s = m.group(1).strip()
        # Trim common punctuation from text/config endings.
        s = s.rstrip("),]} ")
        vals.append(s)
    return vals


def line_snippets(text, max_hits=100):
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        ll = line.lower()
        if any(k in ll for k in PROVENANCE_KEYWORDS):
            hits.append((i, line.strip()))
            if len(hits) >= max_hits:
                break
    return hits


def table_path_columns(columns):
    out = []
    for c in columns:
        cl = str(c).strip().lower()
        if cl in PATH_COLUMNS:
            out.append(c)
        elif "image" in cl and "path" in cl:
            out.append(c)
        elif "img" in cl and "path" in cl:
            out.append(c)
    return out


def table_split_columns(columns):
    out = []
    for c in columns:
        cl = str(c).strip().lower()
        if cl in SPLIT_COLUMNS or "split" in cl:
            out.append(c)
    return out


def classify_training_rows(df, table_path):
    """
    Return:
      mask, basis, confidence

    Training membership is accepted only when there is an explicit split
    column OR the table filename itself clearly says train/training.
    """
    split_cols = table_split_columns(df.columns)

    for sc in split_cols:
        vals = (
            df[sc]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mask = vals.isin(TRAIN_TOKENS) | vals.str.startswith("train")
        if mask.any():
            return mask, f"explicit_split_column:{sc}", "EXPLICIT"

    name = table_path.name.lower()
    if (
        "train" in name
        and "val" not in name
        and "test" not in name
    ):
        return (
            pd.Series(True, index=df.index),
            "training_table_filename",
            "EXPLICIT_FILE_LEVEL",
        )

    return (
        pd.Series(False, index=df.index),
        "no_explicit_training_membership",
        "NONE",
    )


def resolve_path(raw, table_path, project_root):
    if pd.isna(raw):
        return None

    s = str(raw).strip().strip('"').strip("'")
    if not s:
        return None

    p = Path(s)
    if p.is_absolute():
        return p

    q = table_path.parent / p
    if q.exists():
        return q

    q = project_root / p
    return q


def matching_code_files(code_root, run_roots):
    """
    Match code artifacts by stable run tokens only.
    This is discovery evidence; it never by itself establishes train membership.
    """
    tokens = set()
    for rr in run_roots:
        name = rr.name.lower()
        for t in re.split(r"[^a-z0-9]+", name):
            if len(t) >= 4 and (
                t.startswith("s0")
                or t.startswith("r0")
                or "pranet" in t
                or "deeplab" in t
                or "segformer" in t
            ):
                tokens.add(t)

    matches = []
    if not code_root.exists():
        return matches

    for p in code_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS | TABLE_EXTS:
            continue

        pl = p.name.lower()
        if any(t in pl for t in tokens):
            matches.append(p)

    return sorted(set(matches))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint_recovery",
        default=DEFAULT_CHECKPOINT_RECOVERY,
    )
    ap.add_argument(
        "--polypgen_manifest",
        default=DEFAULT_POLYPGEN_MANIFEST,
    )
    ap.add_argument(
        "--polypgen_lock",
        default=DEFAULT_POLYPGEN_LOCK,
    )
    ap.add_argument(
        "--project_root",
        default=DEFAULT_PROJECT_ROOT,
    )
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint_recovery)
    polypgen_manifest_path = Path(args.polypgen_manifest)
    polypgen_lock_path = Path(args.polypgen_lock)
    project_root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10L2A SOURCE-TRAINING LINEAGE DISCOVERY AUDIT FIX1 =====")
    print("STATUS: PROVENANCE / TRAINING-OVERLAP AUDIT")
    print("MODEL INFERENCE: NONE")
    print("EXTERNAL PERFORMANCE: NONE")
    print("TTA: NONE")
    print("SAFETY SCORING: NONE")
    print("METHOD TUNING: NONE")
    print("TRAIN MEMBERSHIP GUESSING FROM DIRECTORY PROXIMITY: NO")
    print()

    for name, p in {
        "checkpoint recovery": checkpoint_path,
        "PolypGen unique manifest": polypgen_manifest_path,
        "PolypGen manifest lock": polypgen_lock_path,
        "project root": project_root,
    }.items():
        print(name, "exists:", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(p)

    with open(polypgen_lock_path, "r", encoding="utf-8") as f:
        pg_lock = json.load(f)

    expected_pg = (
        "POLYPGEN_STATIC_UNIQUE_MANIFEST_LOCKED_"
        "PENDING_SOURCE_TRAINING_AUDIT"
    )
    if pg_lock.get("decision") != expected_pg:
        raise AssertionError(
            f"Unexpected PolypGen manifest-lock state: "
            f"{pg_lock.get('decision')}"
        )

    pg = pd.read_csv(polypgen_manifest_path, low_memory=False)
    if len(pg) != 1532:
        raise AssertionError(
            f"Expected frozen 1532 PolypGen rows, got {len(pg)}."
        )
    if "image_sha256" not in pg.columns:
        raise AssertionError("PolypGen locked manifest missing image_sha256.")

    pg_hashes = set(
        pg["image_sha256"].astype(str).str.lower().tolist()
    )
    if len(pg_hashes) != 1532:
        raise AssertionError(
            "PolypGen locked RGB hashes are not one-to-one."
        )

    ck = pd.read_csv(checkpoint_path, low_memory=False)
    required_ck = [
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "exact_sha256_paths",
        "recovered",
    ]
    missing = [c for c in required_ck if c not in ck.columns]
    if missing:
        raise AssertionError(f"Checkpoint table missing: {missing}")

    if len(ck) != 9:
        raise AssertionError(f"Expected 9 checkpoint states, got {len(ck)}.")
    if not ck["recovered"].astype(bool).all():
        raise AssertionError("Not all frozen checkpoints are recovered.")

    # Resolve one exact checkpoint path per state.
    state_rows = []
    run_roots = []

    for r in ck.itertuples(index=False):
        paths = [
            Path(x)
            for x in str(r.exact_sha256_paths).split(";")
            if str(x).strip()
        ]
        existing = [p for p in paths if p.exists()]
        if len(existing) != 1:
            raise AssertionError(
                f"{r.model_state_id}: expected exactly one existing checkpoint "
                f"path, got {len(existing)}"
            )

        cp = existing[0]
        rr = derive_run_root(cp)
        run_roots.append(rr)

        state_rows.append({
            "model_state_id": r.model_state_id,
            "model_family": r.model_family,
            "training_seed": int(r.training_seed),
            "checkpoint_sha256": str(r.checkpoint_sha256).lower(),
            "checkpoint_path": str(cp),
            "run_root": str(rr),
            "run_root_exists": rr.exists(),
        })

    state_df = pd.DataFrame(state_rows)

    print("\n===== FROZEN STATE RUN ROOTS =====")
    print(state_df.to_string(index=False))

    if not state_df["run_root_exists"].all():
        raise AssertionError("One or more frozen training run roots are missing.")

    unique_run_roots = sorted(
        {Path(x) for x in state_df["run_root"].tolist()},
        key=lambda x: str(x).lower(),
    )

    # ------------------------------------------------------------------
    # Collect evidence files from run roots + matching code artifacts.
    # ------------------------------------------------------------------
    evidence_files = set()

    for rr in unique_run_roots:
        for root_dir, dirs, files in os.walk(rr):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_SCAN_DIRS]
            for fn in files:
                p = Path(root_dir) / fn
                if p.suffix.lower() in TEXT_EXTS | TABLE_EXTS:
                    evidence_files.add(p)

    code_root = project_root / "code"
    evidence_files.update(
        matching_code_files(code_root, unique_run_roots)
    )

    evidence_files = sorted(
        evidence_files,
        key=lambda x: str(x).lower(),
    )

    print("\nEvidence files selected:", len(evidence_files))

    # ------------------------------------------------------------------
    # Text provenance discovery.
    # ------------------------------------------------------------------
    snippet_rows = []
    explicit_path_rows = []
    polypgen_mentions = []

    for p in tqdm(
        evidence_files,
        desc="Scanning provenance text",
        unit="file",
        dynamic_ncols=True,
    ):
        if p.suffix.lower() not in TEXT_EXTS:
            continue

        text = read_text_safely(p)
        if text is None:
            continue

        for line_no, line in line_snippets(text):
            snippet_rows.append({
                "evidence_file": str(p),
                "line": line_no,
                "snippet": line[:2000],
            })

            if "polypgen" in line.lower():
                polypgen_mentions.append({
                    "evidence_file": str(p),
                    "line": line_no,
                    "snippet": line[:2000],
                })

        for raw_path in absolute_f_paths(text):
            q = Path(raw_path)
            explicit_path_rows.append({
                "evidence_file": str(p),
                "explicit_path": raw_path,
                "exists": q.exists(),
                "is_file": q.is_file() if q.exists() else False,
                "is_dir": q.is_dir() if q.exists() else False,
            })

    snippets = pd.DataFrame(snippet_rows)
    explicit_paths = pd.DataFrame(explicit_path_rows)
    polypgen_mentions_df = pd.DataFrame(polypgen_mentions)

    print("\nExplicit PolypGen mentions in historical training evidence:",
          len(polypgen_mentions_df))

    if len(polypgen_mentions_df):
        print(
            polypgen_mentions_df.head(50).to_string(index=False)
        )

    print("\nExisting explicit F: paths discovered:",
          int(explicit_paths["exists"].sum()) if len(explicit_paths) else 0)

    if len(explicit_paths):
        print(
            explicit_paths[
                explicit_paths["exists"]
            ].drop_duplicates(
                subset=["explicit_path"]
            ).head(100).to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Manifest/table audit.
    # ------------------------------------------------------------------
    manifest_rows = []
    train_image_rows = []

    for table_path in tqdm(
        [p for p in evidence_files if p.suffix.lower() in TABLE_EXTS],
        desc="Auditing training tables",
        unit="table",
        dynamic_ncols=True,
    ):
        sep = "\t" if table_path.suffix.lower() == ".tsv" else ","

        try:
            df = pd.read_csv(
                table_path,
                sep=sep,
                low_memory=False,
            )
        except Exception as e:
            manifest_rows.append({
                "table_path": str(table_path),
                "rows": -1,
                "path_columns": "",
                "training_basis": "READ_ERROR",
                "training_confidence": "NONE",
                "explicit_train_rows": 0,
                "existing_train_images": 0,
                "error": repr(e),
            })
            continue

        pcols = table_path_columns(df.columns)

        if not pcols:
            continue

        train_mask, basis, confidence = classify_training_rows(
            df,
            table_path,
        )

        explicit_train_rows = int(train_mask.sum())
        existing_train_images = 0

        if explicit_train_rows > 0:
            train_df = df.loc[train_mask]

            # Prefer the most image-specific path column.
            scored_cols = []
            for c in pcols:
                cl = str(c).lower()
                score = 0
                if "image" in cl:
                    score += 4
                if "img" in cl:
                    score += 3
                if "path" in cl:
                    score += 2
                if "mask" in cl or "gt" in cl or "label" in cl:
                    score -= 5
                scored_cols.append((score, c))

            scored_cols.sort(reverse=True, key=lambda x: x[0])
            chosen_col = scored_cols[0][1]

            for idx, raw in train_df[chosen_col].items():
                rp = resolve_path(raw, table_path, project_root)
                if (
                    rp is not None
                    and rp.exists()
                    and rp.is_file()
                    and rp.suffix.lower() in RASTER_EXTS
                ):
                    existing_train_images += 1
                    train_image_rows.append({
                        "table_path": str(table_path),
                        "training_basis": basis,
                        "training_confidence": confidence,
                        "path_column": str(chosen_col),
                        "row_index": int(idx),
                        "image_path": str(rp),
                    })

        manifest_rows.append({
            "table_path": str(table_path),
            "rows": len(df),
            "path_columns": ";".join(map(str, pcols)),
            "training_basis": basis,
            "training_confidence": confidence,
            "explicit_train_rows": explicit_train_rows,
            "existing_train_images": existing_train_images,
            "error": "",
        })

    manifests = pd.DataFrame(manifest_rows)
    train_images = pd.DataFrame(train_image_rows)

    print("\n===== EXPLICIT TRAIN-IMAGE MANIFEST EVIDENCE =====")
    if len(manifests):
        useful = manifests[manifests["existing_train_images"] > 0]
        print("Manifest tables with existing explicit train images:", len(useful))
        if len(useful):
            print(useful.to_string(index=False))
    else:
        print("NONE")

    # ------------------------------------------------------------------
    # Exact SHA256 overlap for recovered explicit train images.
    # ------------------------------------------------------------------
    if len(train_images):
        train_images = train_images.drop_duplicates(
            subset=["image_path"]
        ).reset_index(drop=True)

        hashes = []
        overlaps = []

        for p in tqdm(
            train_images["image_path"].map(Path).tolist(),
            desc="Hashing explicit source-train images",
            unit="image",
            dynamic_ncols=True,
        ):
            h = sha256_file(p).lower()
            hashes.append(h)
            overlaps.append(h in pg_hashes)

        train_images["image_sha256"] = hashes
        train_images["exact_polypgen_overlap"] = overlaps

        exact_overlap_n = int(sum(overlaps))
        unique_train_images = int(train_images["image_sha256"].nunique())
    else:
        exact_overlap_n = 0
        unique_train_images = 0
        train_images = pd.DataFrame(columns=[
            "table_path",
            "training_basis",
            "training_confidence",
            "path_column",
            "row_index",
            "image_path",
            "image_sha256",
            "exact_polypgen_overlap",
        ])

    print("\nExplicit unique source-train image hashes:", unique_train_images)
    print("Exact PolypGen RGB overlaps:", exact_overlap_n)

    if exact_overlap_n > 0:
        print(
            train_images[
                train_images["exact_polypgen_overlap"]
            ].head(100).to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Link recovered manifest evidence to frozen state run roots.
    #
    # Conservative rule:
    # - direct evidence under that state's run root counts;
    # - code-only evidence is discovery-only and does not satisfy completeness.
    # ------------------------------------------------------------------
    state_audit_rows = []

    useful_tables = set()
    if len(manifests):
        useful_tables = set(
            manifests.loc[
                manifests["existing_train_images"] > 0,
                "table_path",
            ].tolist()
        )

    for r in state_df.itertuples(index=False):
        rr = Path(r.run_root)

        direct_tables = []
        for t in useful_tables:
            tp = Path(t)
            try:
                tp.relative_to(rr)
                direct_tables.append(t)
            except ValueError:
                pass

        direct_train_images = 0
        direct_overlap = 0

        if len(train_images) and direct_tables:
            sub = train_images[
                train_images["table_path"].isin(direct_tables)
            ]
            direct_train_images = int(sub["image_sha256"].nunique())
            direct_overlap = int(
                sub["exact_polypgen_overlap"].astype(bool).sum()
            )

        if direct_train_images > 0 and direct_overlap == 0:
            lineage_status = "EXPLICIT_TRAIN_IMAGES_RECOVERED_ZERO_EXACT_OVERLAP"
        elif direct_overlap > 0:
            lineage_status = "POLYPGEN_EXACT_TRAINING_OVERLAP_DETECTED"
        else:
            lineage_status = "EXACT_TRAIN_IMAGE_LINEAGE_INCOMPLETE"

        state_audit_rows.append({
            "model_state_id": r.model_state_id,
            "model_family": r.model_family,
            "training_seed": r.training_seed,
            "run_root": r.run_root,
            "direct_training_manifest_count": len(direct_tables),
            "direct_unique_train_images": direct_train_images,
            "direct_exact_polypgen_overlaps": direct_overlap,
            "lineage_status": lineage_status,
            "direct_training_tables": ";".join(direct_tables),
        })

    state_audit = pd.DataFrame(state_audit_rows)

    print("\n===== PER-STATE SOURCE-TRAINING LINEAGE =====")
    print(state_audit.to_string(index=False))

    all_explicit = bool(
        (
            state_audit["lineage_status"]
            == "EXPLICIT_TRAIN_IMAGES_RECOVERED_ZERO_EXACT_OVERLAP"
        ).all()
    )

    any_overlap = bool(
        (
            state_audit["lineage_status"]
            == "POLYPGEN_EXACT_TRAINING_OVERLAP_DETECTED"
        ).any()
    )

    any_historical_polypgen_mention = len(polypgen_mentions_df) > 0

    if any_overlap:
        decision = "POLYPGEN_SOURCE_TRAINING_EXACT_OVERLAP_DETECTED_STOP"
    elif all_explicit and not any_historical_polypgen_mention:
        decision = (
            "ALL_FROZEN_STATES_EXPLICIT_SOURCE_TRAINING_"
            "ZERO_EXACT_POLYPGEN_OVERLAP"
        )
    else:
        decision = (
            "SOURCE_TRAINING_LINEAGE_INCOMPLETE_"
            "EXTERNAL_VALIDATION_NOT_YET_AUTHORIZED"
        )

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------
    state_df.to_csv(
        out / "R10L2A_frozen_state_run_roots.csv",
        index=False,
    )
    snippets.to_csv(
        out / "R10L2A_provenance_text_snippets.csv",
        index=False,
    )
    explicit_paths.to_csv(
        out / "R10L2A_explicit_paths.csv",
        index=False,
    )
    polypgen_mentions_df.to_csv(
        out / "R10L2A_historical_polypgen_mentions.csv",
        index=False,
    )
    manifests.to_csv(
        out / "R10L2A_manifest_table_audit.csv",
        index=False,
    )
    train_images.to_csv(
        out / "R10L2A_explicit_train_image_hashes.csv",
        index=False,
    )
    state_audit.to_csv(
        out / "R10L2A_per_state_lineage_audit.csv",
        index=False,
    )

    summary = {
        "decision": decision,
        "frozen_states": 9,
        "polypgen_locked_cases": 1532,
        "evidence_files_scanned": len(evidence_files),
        "historical_polypgen_mentions": len(polypgen_mentions_df),
        "explicit_unique_source_train_images_hashed": unique_train_images,
        "exact_polypgen_training_overlaps": exact_overlap_n,
        "states_with_explicit_zero_overlap_lineage": int(
            (
                state_audit["lineage_status"]
                == "EXPLICIT_TRAIN_IMAGES_RECOVERED_ZERO_EXACT_OVERLAP"
            ).sum()
        ),
        "states_with_incomplete_exact_train_lineage": int(
            (
                state_audit["lineage_status"]
                == "EXACT_TRAIN_IMAGE_LINEAGE_INCOMPLETE"
            ).sum()
        ),
        "model_inference_run": False,
        "external_performance_evaluated": False,
    }

    with open(
        out / "R10L2A_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)
    print("MODEL INFERENCE RUN: NO")
    print("EXTERNAL PERFORMANCE EVALUATED: NO")
    print("EXACT POLYPGEN TRAINING OVERLAPS:", exact_overlap_n)
    print(
        "STATES WITH EXPLICIT ZERO-OVERLAP TRAIN LINEAGE:",
        summary["states_with_explicit_zero_overlap_lineage"],
        "/ 9",
    )
    print("\nOutputs:")
    print(out / "R10L2A_per_state_lineage_audit.csv")
    print(out / "R10L2A_explicit_train_image_hashes.csv")
    print(out / "R10L2A_provenance_text_snippets.csv")
    print(out / "R10L2A_summary.json")
    print("PASS")


if __name__ == "__main__":
    main()
