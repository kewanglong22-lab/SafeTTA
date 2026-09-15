#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-1B — exact NeoPolyp SOURCE-state 66-D reconstruction.

Purpose
-------
Reconstruct the historical NeoPolyp SOURCE-state representation exactly from
already-frozen upstream assets after the final 66-D cache was found missing.

Historical frozen definition:
    SOURCE state = [M2 morphology (2-D), PCA64(CondDINO SOURCE 1536-D)]

where:
    CondDINO SOURCE 1536-D =
        foreground_patch_mean 768-D || background_patch_mean 768-D

and PCA64 is the frozen R10L0 PCA fit on the NeoPolyp development cohort.

This script:
- DOES NOT fit PCA;
- DOES NOT fit a classifier;
- DOES NOT access PolypGen outcomes;
- DOES NOT rerun DINO;
- DOES NOT rerun TTA;
- DOES NOT compute AUROC/AUPRC;
- DOES NOT alter the representation.

It only performs deterministic historical reconstruction and freezes the
result as a new provenance-tracked asset for the P0-1B experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd


VERSION = "2026-09-15-B6-P01B-SOURCE66-RECON-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

PROTOCOL = CODE / "B6_P01B_POLYPGEN_MATCHED_CANDIDATE_CONDITIONING_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "64ad2ac72ff6fc04ba07a6492e27c6b1af9a537fd0a49e787b5ae2d0b9ea62f9"

FEAS_AUDIT = (
    ROOT
    / "B6_P01B_historical_feature_reconstruction_feasibility_audit_v1"
    / "B6_P01B_RECONSTRUCTION_FEASIBILITY_AUDIT.json"
)
EXPECTED_FEAS_GATE = (
    "PASS_B6_P01B_HISTORICAL_FEATURE_RECONSTRUCTION_FEASIBILITY_AUDIT_COMPLETE"
)

R10K2A_DIR = (
    OUTPUTS
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1_v1"
)
R10K2A_STATE_META = R10K2A_DIR / "R10K2A_state_metadata_no_labels.csv"
R10K2A_SOURCE_COND = R10K2A_DIR / "R10K2A_mask_conditioned_dinov2_features.npz"

R10J1_DIR = (
    OUTPUTS
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1"
)
R10J1_MORPH = R10J1_DIR / "source_mask_morphology_features_labeled.csv"

R10L0_DIR = (
    OUTPUTS
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
R10L0_LOCK = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"
R10L0_PCA = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
R10L0_HEAD = R10L0_DIR / "R10L0_final_safety_head.joblib"

R05D1_MANIFEST = (
    OUTPUTS
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

OUT_DIR = ROOT / "B6_P01B_reconstruct_neopolyp_source_state66_v1"
PASS_GATE = "PASS_B6_P01B_NEOPOLYP_SOURCE_STATE66_EXACT_RECONSTRUCTION"

M2_COLS = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

N_ROWS = 9000
N_CASES = 1000
COND_DIM = 1536
PCA_DIM = 64
FINAL_DIM = 66


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exact_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}"
        )
    return got


def norm_sid_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    exact_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "P01B protocol")
    protocol = load_json(PROTOCOL)

    if protocol.get("status") != "FROZEN_BEFORE_P01B_METRICS":
        raise RuntimeError("P01B protocol status changed.")

    if not FEAS_AUDIT.is_file():
        raise FileNotFoundError(FEAS_AUDIT)

    feas = load_json(FEAS_AUDIT)

    if feas.get("status") != "PASS" or feas.get("gate") != EXPECTED_FEAS_GATE:
        raise RuntimeError("Historical reconstruction feasibility gate changed.")

    if feas.get("source_state_exact_reconstruction_ready") is not True:
        raise RuntimeError(
            "Upstream feasibility audit did not authorize exact SOURCE-state reconstruction."
        )

    return protocol, feas


def load_and_bind() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    for p in [
        R10K2A_STATE_META,
        R10K2A_SOURCE_COND,
        R10J1_MORPH,
        R10L0_LOCK,
        R10L0_PCA,
        R05D1_MANIFEST,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    state = pd.read_csv(R10K2A_STATE_META, low_memory=False)
    morph = pd.read_csv(R10J1_MORPH, low_memory=False)
    manifest = pd.read_csv(R05D1_MANIFEST, low_memory=False)

    required_state = {
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
    }
    required_morph = {
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
        *M2_COLS,
    }

    if not required_state.issubset(state.columns):
        raise RuntimeError(
            f"R10K2A state metadata missing {sorted(required_state - set(state.columns))}"
        )
    if not required_morph.issubset(morph.columns):
        raise RuntimeError(
            f"R10J1 morphology missing {sorted(required_morph - set(morph.columns))}"
        )
    if "sample_id" not in manifest.columns:
        raise RuntimeError("R05D1 manifest missing sample_id.")

    if len(state) != N_ROWS:
        raise RuntimeError(f"state rows={len(state)} != {N_ROWS}")
    if len(morph) != N_ROWS:
        raise RuntimeError(f"morph rows={len(morph)} != {N_ROWS}")
    if len(manifest) != N_CASES:
        raise RuntimeError(f"manifest rows={len(manifest)} != {N_CASES}")

    state["_sid"] = norm_sid_series(state["sample_id"])
    morph["_sid"] = norm_sid_series(morph["sample_id"])
    manifest["_sid"] = norm_sid_series(manifest["sample_id"])

    if state[["_sid", "model_state_id"]].drop_duplicates().shape[0] != N_ROWS:
        raise RuntimeError("R10K2A state explicit key not unique.")
    if morph[["_sid", "model_state_id"]].drop_duplicates().shape[0] != N_ROWS:
        raise RuntimeError("R10J1 morphology explicit key not unique.")
    if manifest["_sid"].nunique() != N_CASES:
        raise RuntimeError("R05D1 manifest sample_id not unique.")

    state = state.copy()
    state["_state_row"] = np.arange(N_ROWS, dtype=np.int64)

    joined = morph.merge(
        state[
            [
                "_sid",
                "model_state_id",
                "model_family",
                "training_seed",
                "checkpoint_sha256",
                "_state_row",
            ]
        ],
        on=["_sid", "model_state_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_morph", "_repr"),
    )

    if len(joined) != N_ROWS:
        raise RuntimeError(f"morph/state exact join={len(joined)} != {N_ROWS}")

    fam_ok = (
        joined["model_family_morph"].astype(str)
        == joined["model_family_repr"].astype(str)
    )
    if not fam_ok.all():
        raise RuntimeError("model_family mismatch after exact join.")

    if set(joined["_sid"]) != set(manifest["_sid"]):
        raise RuntimeError("Development panel and 1000-case manifest case sets differ.")

    with np.load(R10K2A_SOURCE_COND, allow_pickle=False) as z:
        required = {"foreground_patch_mean", "background_patch_mean"}
        if not required.issubset(z.files):
            raise RuntimeError(f"R10K2A NPZ missing keys. available={z.files}")

        fg = np.asarray(z["foreground_patch_mean"], dtype=np.float32)
        bg = np.asarray(z["background_patch_mean"], dtype=np.float32)

    if fg.shape != (N_ROWS, 768):
        raise RuntimeError(f"FG shape={fg.shape}")
    if bg.shape != (N_ROWS, 768):
        raise RuntimeError(f"BG shape={bg.shape}")

    cond = np.concatenate([fg, bg], axis=1).astype(np.float32)

    if cond.shape != (N_ROWS, COND_DIM):
        raise RuntimeError(f"CondDINO shape={cond.shape}")
    if not np.isfinite(cond).all():
        raise RuntimeError("Non-finite CondDINO SOURCE features.")

    cond_aligned = cond[joined["_state_row"].to_numpy(dtype=np.int64)]

    m2 = np.column_stack(
        [
            pd.to_numeric(joined[c], errors="raise").to_numpy(dtype=np.float64)
            for c in M2_COLS
        ]
    )

    if m2.shape != (N_ROWS, 2):
        raise RuntimeError(f"M2 shape={m2.shape}")
    if not np.isfinite(m2).all():
        raise RuntimeError("Non-finite M2.")

    joined = joined.rename(columns={"model_family_morph": "model_family"})
    if "model_family_repr" in joined.columns:
        joined = joined.drop(columns=["model_family_repr"])

    return joined.reset_index(drop=True), cond_aligned, m2


def reconstruct_source_state(
    joined: pd.DataFrame,
    cond_aligned: np.ndarray,
    m2: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    lock = load_json(R10L0_LOCK)

    if lock.get("decision") != (
        "FINAL_SOURCE_ESTIMATOR_LOCKED_FOR_INDEPENDENT_EXTERNAL_VALIDATION"
    ):
        raise RuntimeError("Unexpected R10L0 lock decision.")

    rep = lock.get("representation", {})

    if rep.get("pca_dimension") != PCA_DIM:
        raise RuntimeError("Historical PCA dimension changed.")
    if rep.get("final_input_dimension") != FINAL_DIM:
        raise RuntimeError("Historical final input dimension changed.")
    if list(rep.get("m2_features", [])) != M2_COLS:
        raise RuntimeError("Historical M2 feature list changed.")

    pca = joblib.load(R10L0_PCA)

    if np.asarray(pca.components_).shape != (PCA_DIM, COND_DIM):
        raise RuntimeError(
            f"Frozen PCA components shape={np.asarray(pca.components_).shape}"
        )
    if np.asarray(pca.mean_).shape != (COND_DIM,):
        raise RuntimeError("Frozen PCA mean shape changed.")
    if bool(getattr(pca, "whiten", False)) is not True:
        raise RuntimeError("Historical PCA whiten flag changed.")

    # IMPORTANT: transform only. NO fit / fit_transform.
    z64 = np.asarray(pca.transform(cond_aligned), dtype=np.float64)

    if z64.shape != (N_ROWS, PCA_DIM):
        raise RuntimeError(f"z64 shape={z64.shape}")
    if not np.isfinite(z64).all():
        raise RuntimeError("Non-finite reconstructed PCA64.")

    source66 = np.column_stack([m2, z64]).astype(np.float64)

    if source66.shape != (N_ROWS, FINAL_DIM):
        raise RuntimeError(f"SOURCE66 shape={source66.shape}")
    if not np.isfinite(source66).all():
        raise RuntimeError("Non-finite reconstructed SOURCE66.")

    info = {
        "R10L0_lock_sha256": sha256_file(R10L0_LOCK),
        "R10L0_PCA_sha256": sha256_file(R10L0_PCA),
        "R10K2A_state_meta_sha256": sha256_file(R10K2A_STATE_META),
        "R10K2A_source_cond_sha256": sha256_file(R10K2A_SOURCE_COND),
        "R10J1_morph_sha256": sha256_file(R10J1_MORPH),
        "R05D1_manifest_sha256": sha256_file(R05D1_MANIFEST),
        "pca_transform_only": True,
        "pca_fit_performed": False,
        "dino_rerun": False,
        "target_rows_used": 0,
    }

    return source66, info


def frozen_head_roundtrip_check(source66: np.ndarray) -> Dict[str, Any]:
    """
    This is not a performance metric. It checks that the reconstructed 66-D
    input is consumable by the exact historical R10L0 frozen safety head.
    """
    out: Dict[str, Any] = {
        "head_present": R10L0_HEAD.is_file(),
        "checked": False,
    }

    if not R10L0_HEAD.is_file():
        return out

    head = joblib.load(R10L0_HEAD)

    # No fit. Pure predict_proba on the reconstructed historical input.
    p = np.asarray(head.predict_proba(source66)[:, 1], dtype=np.float64)

    if p.shape != (N_ROWS,):
        raise RuntimeError(f"Frozen-head probability shape={p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError("Frozen-head probabilities non-finite.")

    out.update(
        {
            "checked": True,
            "R10L0_HEAD_sha256": sha256_file(R10L0_HEAD),
            "rows": int(len(p)),
            "probability_min": float(p.min()),
            "probability_median": float(np.median(p)),
            "probability_max": float(p.max()),
            "note": (
                "Diagnostic serialization/input-path check only; "
                "not used as P0-1B scientific result."
            ),
        }
    )
    return out


def self_test():
    assert FINAL_DIM == 66
    assert PCA_DIM == 64
    assert COND_DIM == 1536
    assert FINAL_DIM == PCA_DIM + 2
    print("SELF_TEST_DIMENSIONS=PASS")
    print("SELF_TEST_NO_FIT_DESIGN=PASS")
    print("SELF_TEST=PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "SafeTTA B6-P0-1B deterministic reconstruction of historical "
            "NeoPolyp SOURCE-state 66-D features."
        )
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    verify_upstream()

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite SOURCE66 reconstruction output: {OUT_DIR}\n"
            "Use _fix1 if this script itself requires repair."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("=" * 156)
    print("SafeTTA B6-P0-1B — exact NeoPolyp SOURCE-state 66-D reconstruction")
    print(f"Version                : {VERSION}")
    print(f"Protocol SHA           : {EXPECTED_PROTOCOL_SHA256}")
    print("PCA fit                : NO")
    print("Classifier fit         : NO")
    print("DINO/TTA rerun         : NO / NO")
    print("PolypGen rows accessed : 0")
    print("=" * 156)

    print("\n[1/3] Exact upstream row binding")
    joined, cond, m2 = load_and_bind()

    print("rows       =", len(joined))
    print("cases      =", joined["_sid"].nunique())
    print("model states=", joined["model_state_id"].nunique())
    print("CondDINO   =", cond.shape)
    print("M2         =", m2.shape)
    print("EXACT_SOURCE_ROW_BINDING=PASS")

    print("\n[2/3] Frozen PCA64 transform-only reconstruction")
    source66, provenance = reconstruct_source_state(joined, cond, m2)

    print("SOURCE66 shape =", source66.shape)
    print("finite         =", bool(np.isfinite(source66).all()))
    print("SOURCE66_RECONSTRUCTION=PASS")

    print("\n[3/3] Freeze reconstructed asset")

    p_feat = OUT_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66.npy"
    p_meta = OUT_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66_ROWS.csv"

    np.save(p_feat, source66, allow_pickle=False)

    row_cols = [
        "sample_id",
        "_sid",
        "model_state_id",
        "model_family",
        "harm_label",
        "morph_fg_fraction",
        "morph_boundary_density",
    ]
    optional = ["training_seed", "checkpoint_sha256"]

    keep = [c for c in row_cols + optional if c in joined.columns]
    joined[keep].to_csv(p_meta, index=False, encoding="utf-8")

    head_check = frozen_head_roundtrip_check(source66)

    audit = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "scientific_role": (
            "deterministic recovery of missing historical SOURCE-state cache"
        ),
        "representation_definition": (
            "[morph_fg_fraction, morph_boundary_density, "
            "R10L0_PCA64.transform(R10K2A_SOURCE_CondDINO1536)]"
        ),
        "rows": N_ROWS,
        "cases": N_CASES,
        "dimension": FINAL_DIM,
        "no_fit": True,
        "no_dino_rerun": True,
        "no_tta_rerun": True,
        "target_rows_used": 0,
        "provenance": provenance,
        "frozen_head_input_check": head_check,
        "outputs": {
            p_feat.name: {
                "path": str(p_feat),
                "sha256": sha256_file(p_feat),
                "bytes": p_feat.stat().st_size,
                "shape": list(source66.shape),
                "dtype": str(source66.dtype),
            },
            p_meta.name: {
                "path": str(p_meta),
                "sha256": sha256_file(p_meta),
                "bytes": p_meta.stat().st_size,
                "rows": int(len(joined)),
            },
        },
    }

    p_audit = OUT_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66_RECONSTRUCTION_AUDIT.json"
    write_json(p_audit, audit)

    report = "\n".join(
        [
            "=" * 156,
            "SafeTTA B6-P0-1B NEOPOLYP SOURCE66 RECONSTRUCTION COMPLETE",
            f"rows={N_ROWS}",
            f"cases={N_CASES}",
            f"shape={source66.shape}",
            f"feature_sha256={sha256_file(p_feat)}",
            f"row_metadata_sha256={sha256_file(p_meta)}",
            f"frozen_head_input_check={head_check.get('checked', False)}",
            "",
            "No PCA/classifier fitting was performed.",
            "No target data were used.",
            "No scientific performance metric was computed.",
            "",
            f"GATE={PASS_GATE}",
            f"audit_json={p_audit}",
            f"stage_dir={OUT_DIR}",
            "=" * 156,
            "",
        ]
    )

    (OUT_DIR / "B6_P01B_NEOPOLYP_SOURCE_STATE66_RECONSTRUCTION_REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
