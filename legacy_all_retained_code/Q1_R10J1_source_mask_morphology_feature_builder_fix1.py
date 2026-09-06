
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10J1_source_mask_morphology_feature_builder_fix1.py

Build a frozen richer PRE-ADAPTATION morphology representation from the
R05D3-locked binary SOURCE masks only.

Feature construction reads ONLY source_masks_packed.  a1_masks_packed is
checked only as a schema key and its values are never accessed.

Outcome labels are joined only AFTER the no-label morphology table has been
constructed and audited, via explicit key sample_id + model_state_id.

No model fitting or performance evaluation occurs here.
"""

import argparse
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage


MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = MASK_PIXELS // 8
BITORDER = "little"
ATOL = 1e-12

DEFAULT_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2_v1/"
    "model_case_prediction_manifest_relocated_fix2.csv"
)

DEFAULT_OUTCOMES = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
    "model_case_outcomes.csv"
)

DEFAULT_R10A0 = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10A0_invariant_feature_table_v1/"
    "invariant_feature_table.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1"
)

FEATURES = [
    "morph_fg_fraction",
    "morph_boundary_density",
    "morph_component_count_8",
    "morph_largest_component_ratio",
    "morph_largest_component_extent",
    "morph_largest_component_eccentricity",
    "morph_largest_component_circularity",
    "morph_largest_component_perimeter_pixels",
    "morph_largest_component_perimeter_area_ratio",
    "morph_hole_count_4",
    "morph_euler_number_8_4",
]

STRUCT8 = np.ones((3, 3), dtype=np.uint8)
STRUCT4 = np.asarray(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    dtype=np.uint8,
)


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def boundary_density(mask):
    m = np.asarray(mask, dtype=bool)
    h = np.mean(m[:, 1:] != m[:, :-1]) if m.shape[1] > 1 else 0.0
    v = np.mean(m[1:, :] != m[:-1, :]) if m.shape[0] > 1 else 0.0
    return float(0.5 * (h + v))


def grid_perimeter_pixels(mask):
    m = np.asarray(mask, dtype=np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    vertical = np.abs(np.diff(p.astype(np.int16), axis=0)).sum()
    horizontal = np.abs(np.diff(p.astype(np.int16), axis=1)).sum()
    return float(vertical + horizontal)


def component_eccentricity(component):
    coords = np.argwhere(component)
    if len(coords) < 2:
        return 0.0

    centered = coords.astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(len(centered))
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))

    major = float(vals[-1])
    minor = float(vals[0])

    if major <= 0.0:
        return 0.0

    return float(math.sqrt(max(0.0, 1.0 - minor / major)))


def component_extent(component):
    coords = np.argwhere(component)
    if len(coords) == 0:
        return 0.0

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_area = int(r1 - r0 + 1) * int(c1 - c0 + 1)

    if bbox_area <= 0:
        return 0.0

    return float(len(coords) / bbox_area)


def hole_count_4(mask):
    bg = ~np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(bg, structure=STRUCT4)

    if n == 0:
        return 0

    border_labels = set(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ]).tolist()
    )
    border_labels.discard(0)

    holes = 0
    for lab in range(1, n + 1):
        if lab not in border_labels:
            holes += 1

    return int(holes)


def extract_morphology(mask):
    m = np.asarray(mask, dtype=bool)
    area = int(m.sum())
    fg_fraction = float(area / MASK_PIXELS)
    bd = boundary_density(m)

    labels, n_components = ndimage.label(m, structure=STRUCT8)

    if n_components == 0 or area == 0:
        return {
            "morph_fg_fraction": fg_fraction,
            "morph_boundary_density": bd,
            "morph_component_count_8": 0.0,
            "morph_largest_component_ratio": 0.0,
            "morph_largest_component_extent": 0.0,
            "morph_largest_component_eccentricity": 0.0,
            "morph_largest_component_circularity": 0.0,
            "morph_largest_component_perimeter_pixels": 0.0,
            "morph_largest_component_perimeter_area_ratio": 0.0,
            "morph_hole_count_4": 0.0,
            "morph_euler_number_8_4": 0.0,
        }

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest_area = int(counts[largest_label])
    largest = labels == largest_label

    largest_perimeter = grid_perimeter_pixels(largest)
    largest_ratio = float(largest_area / area)
    extent = component_extent(largest)
    ecc = component_eccentricity(largest)

    if largest_perimeter > 0.0 and largest_area > 0:
        circularity = float(
            4.0 * math.pi * largest_area / (largest_perimeter ** 2)
        )
        perim_area = float(largest_perimeter / largest_area)
    else:
        circularity = 0.0
        perim_area = 0.0

    holes = hole_count_4(m)
    euler = int(n_components) - int(holes)

    return {
        "morph_fg_fraction": fg_fraction,
        "morph_boundary_density": bd,
        "morph_component_count_8": float(n_components),
        "morph_largest_component_ratio": largest_ratio,
        "morph_largest_component_extent": extent,
        "morph_largest_component_eccentricity": ecc,
        "morph_largest_component_circularity": circularity,
        "morph_largest_component_perimeter_pixels": largest_perimeter,
        "morph_largest_component_perimeter_area_ratio": perim_area,
        "morph_hole_count_4": float(holes),
        "morph_euler_number_8_4": float(euler),
    }


def unpack_source_mask(packed_row):
    p = np.asarray(packed_row, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise AssertionError(
            f"Packed row shape={p.shape}, expected={(PACKED_BYTES,)}"
        )

    bits = np.unpackbits(p, bitorder=BITORDER)
    if bits.shape != (MASK_PIXELS,):
        raise AssertionError(
            f"Unpacked bits={bits.shape}, expected={(MASK_PIXELS,)}"
        )

    return bits.reshape(MASK_H, MASK_W).astype(bool)


def vector_multiset_match(A, B, atol=ATOL):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape:
        return False, np.inf

    if A.ndim != 2 or A.shape[1] != 2:
        raise AssertionError("Expected Nx2 morphology vectors.")

    oa = np.lexsort((A[:, 1], A[:, 0]))
    ob = np.lexsort((B[:, 1], B[:, 0]))
    A2 = A[oa]
    B2 = B[ob]

    d = np.abs(A2 - B2)
    maxdiff = float(d.max()) if d.size else 0.0
    return bool(np.all(d <= atol)), maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    ap.add_argument("--r10a0", default=DEFAULT_R10A0)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10J1 SOURCE-MASK MORPHOLOGY FEATURE BUILDER FIX1 =====")
    print("STATUS: FROZEN FEATURE CONSTRUCTION")
    print("MASK RESOLUTION: 352x352")
    print("PACKBITS BITORDER: little")
    print("READ source_masks_packed: YES")
    print("READ a1_masks_packed VALUES: NO")
    print("GT USED IN FEATURE CONSTRUCTION: NO")
    print("DICE USED IN FEATURE CONSTRUCTION: NO")
    print("HARM LABEL USED IN FEATURE CONSTRUCTION: NO")
    print("MODEL FITTING: NONE")
    print("THRESHOLD TUNING: NONE")
    print("FEATURE COUNT:", len(FEATURES))
    for i, c in enumerate(FEATURES, 1):
        print(f"  {i:02d}. {c}")
    print()

    manifest_path = Path(args.manifest)
    outcomes_path = Path(args.outcomes)
    r10a0_path = Path(args.r10a0)

    print("manifest exists:", manifest_path.exists(), manifest_path)
    print("outcomes exists:", outcomes_path.exists(), outcomes_path)
    print("R10A0 exists:", r10a0_path.exists(), r10a0_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)
    if not r10a0_path.exists():
        raise FileNotFoundError(r10a0_path)

    man = pd.read_csv(manifest_path, low_memory=False)
    outc = pd.read_csv(outcomes_path, low_memory=False)
    old = pd.read_csv(r10a0_path, low_memory=False)

    required_man = [
        "sample_id",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "state_prediction_npz",
        "state_row_index",
        "source_foreground_pixels",
    ]
    required_out = [
        "sample_id",
        "model_state_id",
        "model_family",
        "training_seed",
        "checkpoint_sha256",
        "harm_label",
        "adaptation_outcome",
    ]
    required_old = [
        "sample_id",
        "model_family",
        "source_fg_fraction",
        "source_boundary_density",
    ]

    for c in required_man:
        if c not in man.columns:
            raise AssertionError(f"Manifest missing: {c}")
    for c in required_out:
        if c not in outc.columns:
            raise AssertionError(f"Outcomes missing: {c}")
    for c in required_old:
        if c not in old.columns:
            raise AssertionError(f"R10A0 missing: {c}")

    man["_sid"] = norm_sid(man["sample_id"])
    outc["_sid"] = norm_sid(outc["sample_id"])
    old["_sid"] = norm_sid(old["sample_id"])

    key = ["_sid", "model_state_id"]

    print("\n===== MANIFEST CARDINALITY =====")
    print("Rows:", len(man))
    print("Cases:", man["_sid"].nunique())
    print("States:", man["model_state_id"].nunique())
    print("Unique sample+state:", man[key].drop_duplicates().shape[0])

    if len(man) != 9000:
        raise AssertionError("Expected 9000 manifest rows.")
    if man["_sid"].nunique() != 1000:
        raise AssertionError("Expected 1000 cases.")
    if man["model_state_id"].nunique() != 9:
        raise AssertionError("Expected 9 model states.")
    if man[key].drop_duplicates().shape[0] != 9000:
        raise AssertionError("sample_id+model_state_id is not unique.")

    print("\n===== STATE ROW-INDEX AUDIT =====")
    for state_id, g in man.groupby("model_state_id", sort=True):
        idx = pd.to_numeric(
            g["state_row_index"], errors="raise"
        ).astype(int).to_numpy()

        ok = np.array_equal(np.sort(idx), np.arange(1000))
        print(
            state_id,
            "rows=", len(g),
            "row_index_0_999=", ok,
            "npz=", g["state_prediction_npz"].nunique(),
        )

        if len(g) != 1000 or not ok:
            raise AssertionError(f"State row-index audit failed: {state_id}")
        if g["state_prediction_npz"].nunique() != 1:
            raise AssertionError(f"Multiple NPZ paths for state: {state_id}")

    feature_rows = []
    fg_pixel_agree = 0
    fg_pixel_total = 0

    for state_id, g in man.groupby("model_state_id", sort=True):
        g = g.copy()
        path = Path(g["state_prediction_npz"].iloc[0])

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise AssertionError(f"{state_id}: source_masks_packed missing")
            if "a1_masks_packed" not in z.files:
                raise AssertionError(f"{state_id}: expected locked A1 key absent")

            # Values from the adapted mask array are intentionally never accessed.
            source_packed = z["source_masks_packed"]

            if source_packed.shape != (1000, PACKED_BYTES):
                raise AssertionError(
                    f"{state_id}: source shape={source_packed.shape}"
                )

            print(
                f"\nSTATE {state_id}: "
                f"source_masks_packed shape={source_packed.shape}; "
                "a1 array value access=NO"
            )

            pbar = tqdm(
                g.itertuples(index=False),
                total=len(g),
                desc=f"Extract morphology {state_id}",
                unit="mask",
                dynamic_ncols=True,
            )

            for r in pbar:
                row_index = int(getattr(r, "state_row_index"))
                mask = unpack_source_mask(source_packed[row_index])

                observed_fg = int(mask.sum())
                expected_fg = int(getattr(r, "source_foreground_pixels"))

                fg_pixel_total += 1
                if observed_fg == expected_fg:
                    fg_pixel_agree += 1
                else:
                    raise AssertionError(
                        f"Foreground-pixel mismatch "
                        f"{getattr(r, 'sample_id')} {state_id} "
                        f"row={row_index}: decoded={observed_fg} "
                        f"manifest={expected_fg}"
                    )

                feats = extract_morphology(mask)

                item = {
                    "sample_id": getattr(r, "sample_id"),
                    "model_family": getattr(r, "model_family"),
                    "model_state_id": getattr(r, "model_state_id"),
                    "training_seed": getattr(r, "training_seed"),
                    "checkpoint_sha256": getattr(r, "checkpoint_sha256"),
                    "state_row_index": row_index,
                    "source_foreground_pixels_manifest": expected_fg,
                    "source_prediction_npz": str(path),
                }
                item.update(feats)
                feature_rows.append(item)

    feat = pd.DataFrame(feature_rows)

    print("\n===== DECODE AUDIT =====")
    print("Feature rows:", len(feat))
    print("Foreground-pixel agreement:", fg_pixel_agree, "/", fg_pixel_total)

    if len(feat) != 9000:
        raise AssertionError("Expected 9000 morphology rows.")
    if fg_pixel_agree != 9000:
        raise AssertionError("Foreground-pixel audit incomplete.")

    feat["_sid"] = norm_sid(feat["sample_id"])

    old_groups = {
        k: g for k, g in old.groupby(
            ["_sid", "model_family"], sort=False
        )
    }
    new_groups = {
        k: g for k, g in feat.groupby(
            ["_sid", "model_family"], sort=False
        )
    }

    print("\n===== EXISTING R10I2 BACKBONE REPRODUCTION =====")
    print("Old groups:", len(old_groups))
    print("New groups:", len(new_groups))
    print("Group-key sets equal:", set(old_groups) == set(new_groups))

    if set(old_groups) != set(new_groups):
        raise AssertionError("R10A0/new morphology group sets differ.")

    matched_groups = 0
    maxdiff = 0.0

    for k in old_groups:
        og = old_groups[k]
        ng = new_groups[k]

        if len(og) != 3 or len(ng) != 3:
            raise AssertionError(f"{k}: expected 3 states/family.")

        A = ng[
            ["morph_fg_fraction", "morph_boundary_density"]
        ].to_numpy(float)

        B = og[
            ["source_fg_fraction", "source_boundary_density"]
        ].apply(pd.to_numeric, errors="raise").to_numpy(float)

        ok, d = vector_multiset_match(A, B, ATOL)

        if ok:
            matched_groups += 1
        maxdiff = max(maxdiff, d)

    print(
        "Sample-family groups reproducing old morphology multiset:",
        matched_groups,
        "/",
        len(old_groups),
    )
    print("Maximum 2-D backbone difference:", maxdiff)

    if matched_groups != len(old_groups) or maxdiff > ATOL:
        raise AssertionError(
            "Decoded SOURCE masks do not reproduce the frozen R10I2 backbone."
        )

    print("\n===== FEATURE SANITY AUDIT BEFORE LABEL JOIN =====")
    sanity_rows = []

    for c in FEATURES:
        x = pd.to_numeric(feat[c], errors="coerce").to_numpy(float)
        finite = np.isfinite(x)

        sanity_rows.append({
            "feature": c,
            "finite": int(finite.sum()),
            "nonfinite": int((~finite).sum()),
            "min": float(np.min(x[finite])) if finite.any() else np.nan,
            "median": float(np.median(x[finite])) if finite.any() else np.nan,
            "mean": float(np.mean(x[finite])) if finite.any() else np.nan,
            "max": float(np.max(x[finite])) if finite.any() else np.nan,
            "unique": int(np.unique(x[finite]).size) if finite.any() else 0,
        })

    sanity = pd.DataFrame(sanity_rows)
    print(sanity.to_string(index=False))

    if not (sanity["nonfinite"] == 0).all():
        raise AssertionError("Non-finite morphology features detected.")

    # Freeze no-label panel BEFORE any outcome join.
    no_label = feat.drop(columns=["_sid"]).copy()
    no_label_path = out / "source_mask_morphology_features_no_labels.csv"
    no_label.to_csv(no_label_path, index=False)

    print("\n===== POST-CONSTRUCTION LABEL JOIN =====")

    join_out = outc[
        [
            "_sid",
            "model_state_id",
            "model_family",
            "training_seed",
            "checkpoint_sha256",
            "harm_label",
            "adaptation_outcome",
        ]
    ].copy()

    labeled = feat.merge(
        join_out,
        on=["_sid", "model_state_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_feature", "_outcome"),
    )

    if len(labeled) != 9000:
        raise AssertionError("Label join changed row count.")
    if labeled["harm_label"].isna().any():
        raise AssertionError("Missing harm labels after explicit-key join.")

    fam_ok = (
        labeled["model_family_feature"].astype(str)
        == labeled["model_family_outcome"].astype(str)
    )
    seed_ok = (
        labeled["training_seed_feature"].astype(str)
        == labeled["training_seed_outcome"].astype(str)
    )
    sha_ok = (
        labeled["checkpoint_sha256_feature"].astype(str)
        == labeled["checkpoint_sha256_outcome"].astype(str)
    )

    print("model_family agreement:", int(fam_ok.sum()), "/ 9000")
    print("training_seed agreement:", int(seed_ok.sum()), "/ 9000")
    print("checkpoint agreement:", int(sha_ok.sum()), "/ 9000")
    print(
        "HARM/NON-HARM:",
        int(labeled["harm_label"].astype(int).sum()),
        "/",
        int((labeled["harm_label"].astype(int) == 0).sum()),
    )

    if not (fam_ok.all() and seed_ok.all() and sha_ok.all()):
        raise AssertionError("Outcome metadata cross-check failed.")

    labeled = labeled.rename(columns={
        "model_family_feature": "model_family",
        "training_seed_feature": "training_seed",
        "checkpoint_sha256_feature": "checkpoint_sha256",
    })

    labeled = labeled.drop(columns=[
        "_sid",
        "model_family_outcome",
        "training_seed_outcome",
        "checkpoint_sha256_outcome",
    ])

    labeled_path = out / "source_mask_morphology_features_labeled.csv"
    labeled.to_csv(labeled_path, index=False)

    sanity_path = out / "R10J1_feature_sanity_summary.csv"
    sanity.to_csv(sanity_path, index=False)

    audit = {
        "status": "PASS",
        "decision": "SOURCE_MASK_MORPHOLOGY_PANEL_LOCKED",
        "rows": 9000,
        "cases": 1000,
        "model_states": 9,
        "mask_height": MASK_H,
        "mask_width": MASK_W,
        "bitorder": BITORDER,
        "source_array_key": "source_masks_packed",
        "a1_array_value_accessed": False,
        "gt_used_for_feature_construction": False,
        "labels_used_for_feature_construction": False,
        "foreground_pixel_agreement": fg_pixel_agree,
        "backbone_reproduction_groups": matched_groups,
        "backbone_reproduction_groups_total": len(old_groups),
        "backbone_max_abs_difference": maxdiff,
        "feature_names": FEATURES,
        "no_label_feature_table": str(no_label_path),
        "labeled_feature_table": str(labeled_path),
    }

    with open(
        out / "R10J1_source_mask_morphology_lock.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL DECISION =====")
    print("DECISION: SOURCE_MASK_MORPHOLOGY_PANEL_LOCKED")
    print("Rows: 9000")
    print("Cases: 1000")
    print("States: 9")
    print("SOURCE mask array used: source_masks_packed")
    print("A1 mask array value accessed: NO")
    print("GT used in feature construction: NO")
    print("HARM label used in feature construction: NO")
    print("Old 2-feature backbone reproduction: PASS")
    print("Maximum backbone difference:", maxdiff)

    print("\nOutputs:")
    print(no_label_path)
    print(labeled_path)
    print(sanity_path)
    print(out / "R10J1_source_mask_morphology_lock.json")
    print("PASS")


if __name__ == "__main__":
    main()
