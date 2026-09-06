
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10K0_spatial_mask_subspace_LOMO_feasibility_fix1.py

Purpose
-------
R10J2 rejected the frozen 11-feature handcrafted morphology panel as a
universal cross-model representation. The next route must therefore stop
adding handcrafted shape descriptors.

R10K0 asks a narrower feasibility question:

    Does the PRE-ADAPTATION binary mask contain transferable spatial safety
    information beyond the two robust scalar morphology features?

Representation
--------------
1) Decode only `source_masks_packed`.
2) Convert each 352x352 binary mask to a 32x32 occupancy map using exact
   11x11 non-overlapping block means (352 = 32 * 11).
3) Flatten the 32x32 map to 1024 dimensions.
4) Inside each outer source-training fold only:
   - fit PCA(n_components=32, svd_solver="randomized", whiten=True);
   - transform source-train/source-test/held-out target-family masks.
5) Train the same balanced logistic-regression safety head.

Primary candidate:
    M2_plus_SpatialPCA32

Controls:
    M2_backbone
    M11_richer_morphology
    SpatialPCA32

This is a feasibility probe for a higher-level spatial mask representation,
NOT a final method.

Strict boundary
---------------
- source mask values only;
- A1 mask values never accessed;
- no GT/Dice/DeltaDice in representation construction;
- target labels evaluation-only;
- target features are not used to fit PCA;
- target held-out cases are never used for training;
- no PCA dimension sweep;
- no classifier sweep.
"""

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


MASK_H = 352
MASK_W = 352
BLOCK = 11
GRID = 32
SPATIAL_DIM = GRID * GRID
PCA_DIM = 32
BITORDER = "little"

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
OUTER_FOLDS = 5
INNER_FOLDS = 4
TARGET_RECALL = 0.90
PPV_PREVALENCE = 0.01

FAMILIES = [
    "DeepLabV3-R50",
    "PraNet",
    "SegFormer-B0",
]

M2 = [
    "morph_fg_fraction",
    "morph_boundary_density",
]

M11 = [
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

DEFAULT_MANIFEST = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J0_source_prediction_npz_relocation_schema_audit_fix2_v1/"
    "model_case_prediction_manifest_relocated_fix2.csv"
)

DEFAULT_MORPH = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10J1_source_mask_morphology_feature_builder_fix1_v1/"
    "source_mask_morphology_features_labeled.csv"
)

DEFAULT_OUTPUT = (
    "F:/MEDSEG_SAFETTA/outputs/"
    "Q1_R10K0_spatial_mask_subspace_LOMO_feasibility_fix1_v1"
)


def norm_sid(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )


def unpack_mask(packed_row):
    x = np.asarray(packed_row, dtype=np.uint8)
    bits = np.unpackbits(x, bitorder=BITORDER)
    if bits.size != MASK_H * MASK_W:
        raise AssertionError(
            f"Unexpected unpacked size: {bits.size}"
        )
    return bits.reshape(MASK_H, MASK_W).astype(np.float32)


def mask_to_occupancy32(mask):
    """
    Exact 11x11 block-average occupancy map:
        352 x 352 -> 32 x 32.
    """
    if mask.shape != (MASK_H, MASK_W):
        raise AssertionError(f"Unexpected mask shape: {mask.shape}")

    if MASK_H != GRID * BLOCK or MASK_W != GRID * BLOCK:
        raise AssertionError("Frozen block geometry is inconsistent.")

    x = mask.reshape(GRID, BLOCK, GRID, BLOCK)
    x = x.mean(axis=(1, 3))
    return x.astype(np.float32).reshape(-1)


def build_lr(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def matrix(df, cols):
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in cols
    ])


def threshold_for_recall(y, p, target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise AssertionError("No positive source samples.")

    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def grouped_oof_lr(X, y, groups, seed):
    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    p = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Inner group leakage.")

        m = build_lr(seed + fold)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]

    if np.isnan(p).any():
        raise AssertionError("Incomplete inner OOF predictions.")

    return p


def op_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    empirical_ppv = tp / (tp + fp) if tp + fp else np.nan

    pi = PPV_PREVALENCE
    denom = recall * pi + fpr * (1 - pi)
    ppv1 = recall * pi / denom if denom > 0 else np.nan

    return {
        "Recall": float(recall),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "EmpiricalPPV": float(empirical_ppv),
        "PPV1pct": float(ppv1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--morphology", default=DEFAULT_MORPH)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10K0 SPATIAL MASK SUBSPACE LOMO FEASIBILITY FIX1 =====")
    print("STATUS: HIGHER-LEVEL MASK REPRESENTATION FEASIBILITY")
    print("HANDCRAFTED FEATURE ADDITION AFTER R10J2: NO")
    print("SOURCE ARRAY: source_masks_packed")
    print("A1 ARRAY VALUES ACCESSED: NO")
    print("GT/DICE USED IN REPRESENTATION: NO")
    print("SPATIAL MAP: 32x32 block occupancy")
    print("BLOCK SIZE: 11x11")
    print("PCA COMPONENTS:", PCA_DIM)
    print("PCA DIMENSION SWEEP: NO")
    print("TARGET FEATURES USED TO FIT PCA: NO")
    print("TARGET LABELS USED FOR FITTING: NO")
    print("TARGET LABELS USED FOR THRESHOLD SELECTION: NO")
    print("TARGET LABELS USED FOR EVALUATION ONLY: YES\n")

    manifest_path = Path(args.manifest)
    morph_path = Path(args.morphology)

    print("manifest exists:", manifest_path.exists(), manifest_path)
    print("morphology exists:", morph_path.exists(), morph_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not morph_path.exists():
        raise FileNotFoundError(morph_path)

    man = pd.read_csv(manifest_path, low_memory=False)
    morph = pd.read_csv(morph_path, low_memory=False)

    required_man = [
        "sample_id",
        "model_family",
        "model_state_id",
        "state_prediction_npz",
        "state_row_index",
    ]
    required_morph = [
        "sample_id",
        "model_family",
        "model_state_id",
        "harm_label",
    ] + M11

    for c in required_man:
        if c not in man.columns:
            raise AssertionError(f"Manifest missing: {c}")
    for c in required_morph:
        if c not in morph.columns:
            raise AssertionError(f"Morphology table missing: {c}")

    man["_sid"] = norm_sid(man["sample_id"])
    morph["_sid"] = norm_sid(morph["sample_id"])

    if len(man) != 9000 or len(morph) != 9000:
        raise AssertionError("Expected 9000 rows.")

    if man[["_sid", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise AssertionError("Manifest explicit key not unique.")
    if morph[["_sid", "model_state_id"]].drop_duplicates().shape[0] != 9000:
        raise AssertionError("Morphology explicit key not unique.")

    if set(map(tuple, man[["_sid", "model_state_id"]].astype(str).to_numpy())) != \
       set(map(tuple, morph[["_sid", "model_state_id"]].astype(str).to_numpy())):
        raise AssertionError("Manifest/morphology state key sets differ.")

    # ------------------------------------------------------------------
    # Build frozen 32x32 occupancy vectors from SOURCE masks only.
    # No outcome labels are accessed in this section.
    # ------------------------------------------------------------------
    print("\n===== BUILDING 32x32 SOURCE-MASK OCCUPANCY VECTORS =====")

    spatial_rows = []
    spatial_vectors = []

    for state_id, g in man.groupby("model_state_id", sort=True):
        g = g.copy()
        npz_path = Path(g["state_prediction_npz"].iloc[0])

        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise AssertionError(
                    f"{state_id}: source_masks_packed missing."
                )
            if "a1_masks_packed" not in z.files:
                raise AssertionError(
                    f"{state_id}: locked A1 key missing."
                )

            source_packed = z["source_masks_packed"]

            if source_packed.shape[0] != 1000:
                raise AssertionError(
                    f"{state_id}: expected 1000 source masks."
                )

            print(
                f"{state_id}: source={source_packed.shape}; "
                "a1 values accessed=NO"
            )

            for r in tqdm(
                g.itertuples(index=False),
                total=len(g),
                desc=f"Occupancy {state_id}",
                unit="mask",
                dynamic_ncols=True,
            ):
                idx = int(getattr(r, "state_row_index"))
                mask = unpack_mask(source_packed[idx])
                occ = mask_to_occupancy32(mask)

                if occ.shape != (SPATIAL_DIM,):
                    raise AssertionError("Unexpected occupancy vector size.")
                if not np.isfinite(occ).all():
                    raise AssertionError("Non-finite occupancy vector.")
                if occ.min() < 0.0 or occ.max() > 1.0:
                    raise AssertionError("Occupancy outside [0,1].")

                spatial_rows.append({
                    "_sid": getattr(r, "_sid"),
                    "sample_id": getattr(r, "sample_id"),
                    "model_family": getattr(r, "model_family"),
                    "model_state_id": getattr(r, "model_state_id"),
                })
                spatial_vectors.append(occ)

    spatial_meta = pd.DataFrame(spatial_rows)
    Xsp = np.asarray(spatial_vectors, dtype=np.float32)

    print("\nSpatial rows:", len(spatial_meta))
    print("Spatial matrix:", Xsp.shape)

    if Xsp.shape != (9000, SPATIAL_DIM):
        raise AssertionError(
            f"Expected spatial matrix (9000,{SPATIAL_DIM}), got {Xsp.shape}"
        )

    # ------------------------------------------------------------------
    # Explicit-key join to the already frozen labeled morphology table.
    # ------------------------------------------------------------------
    spatial_meta["_row_spatial"] = np.arange(len(spatial_meta), dtype=int)

    joined = spatial_meta.merge(
        morph,
        on=["_sid", "sample_id", "model_family", "model_state_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(joined) != 9000:
        raise AssertionError("Spatial/morphology join failed.")

    # Reorder spatial vectors to joined row order.
    Xsp = Xsp[joined["_row_spatial"].to_numpy(int)]

    y_all = pd.to_numeric(
        joined["harm_label"], errors="raise"
    ).astype(int).to_numpy()
    groups_all = joined["_sid"].to_numpy()
    family_all = joined["model_family"].astype(str).to_numpy()

    print("Joined HARM/NON-HARM:", int(y_all.sum()), int((y_all == 0).sum()))
    print("Families:")
    print(joined["model_family"].value_counts().to_string())

    # Save source-only spatial basis input for audit/reuse.
    np.savez_compressed(
        out / "R10K0_source_mask_occupancy32.npz",
        occupancy32=Xsp.astype(np.float32),
        sample_id=joined["sample_id"].astype(str).to_numpy(),
        model_state_id=joined["model_state_id"].astype(str).to_numpy(),
        model_family=joined["model_family"].astype(str).to_numpy(),
    )

    methods = [
        "M2_backbone",
        "M11_richer_morphology",
        "SpatialPCA32",
        "M2_plus_SpatialPCA32",
    ]

    result_rows = []
    fold_rows = []

    for target_family in FAMILIES:
        source_idx = np.flatnonzero(family_all != target_family)
        target_idx = np.flatnonzero(family_all == target_family)

        source = joined.iloc[source_idx].reset_index(drop=True)
        target = joined.iloc[target_idx].reset_index(drop=True)

        Xsp_source = Xsp[source_idx]
        Xsp_target = Xsp[target_idx]

        y_source = y_all[source_idx]
        y_target = y_all[target_idx]
        g_source = groups_all[source_idx]
        g_target = groups_all[target_idx]

        if set(g_source) != set(g_target):
            raise AssertionError(
                f"Source/target case pools differ: {target_family}"
            )

        print("\n\n============================================================")
        print("TARGET FAMILY:", target_family)
        print("SOURCE FAMILIES:", [f for f in FAMILIES if f != target_family])
        print("Source rows:", len(source))
        print("Target rows:", len(target))
        print("============================================================")

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            outer = StratifiedGroupKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=seed,
            )

            splits = list(
                outer.split(
                    np.zeros((len(source), 1)),
                    y_source,
                    g_source,
                )
            )

            probs = {
                m: np.full(len(target), np.nan, dtype=float)
                for m in methods
            }
            thresholds = {
                m: np.full(len(target), np.nan, dtype=float)
                for m in methods
            }

            for fold, (tr_s, te_s) in enumerate(splits):
                train_cases = set(g_source[tr_s])
                test_cases = set(g_source[te_s])

                if train_cases & test_cases:
                    raise AssertionError("Outer group leakage.")

                te_t = np.flatnonzero(
                    np.isin(g_target, list(test_cases))
                )

                if set(g_target[te_t]) != test_cases:
                    raise AssertionError(
                        "Held-out target case set mismatch."
                    )

                # ------------------------------------------------------
                # M2 / M11
                # ------------------------------------------------------
                tabular_panels = {
                    "M2_backbone": M2,
                    "M11_richer_morphology": M11,
                }

                for method, cols in tabular_panels.items():
                    Xtr = matrix(source.iloc[tr_s], cols)
                    Xte = matrix(target.iloc[te_t], cols)

                    inner_p = grouped_oof_lr(
                        Xtr,
                        y_source[tr_s],
                        g_source[tr_s],
                        seed + 1000 + fold,
                    )

                    thr = threshold_for_recall(
                        y_source[tr_s],
                        inner_p,
                        TARGET_RECALL,
                    )

                    clf = build_lr(seed + 10000 + fold)
                    clf.fit(Xtr, y_source[tr_s])

                    probs[method][te_t] = clf.predict_proba(Xte)[:, 1]
                    thresholds[method][te_t] = thr

                # ------------------------------------------------------
                # Frozen spatial PCA fitted SOURCE-TRAIN ONLY.
                # No target feature is used to fit PCA.
                # ------------------------------------------------------
                pca = PCA(
                    n_components=PCA_DIM,
                    svd_solver="randomized",
                    whiten=True,
                    random_state=seed + fold,
                )

                Ztr = pca.fit_transform(Xsp_source[tr_s])
                Zte_t = pca.transform(Xsp_target[te_t])

                explained = float(
                    np.sum(pca.explained_variance_ratio_)
                )

                # SpatialPCA32.
                inner_p = grouped_oof_lr(
                    Ztr,
                    y_source[tr_s],
                    g_source[tr_s],
                    seed + 2000 + fold,
                )
                thr = threshold_for_recall(
                    y_source[tr_s],
                    inner_p,
                    TARGET_RECALL,
                )
                clf = build_lr(seed + 20000 + fold)
                clf.fit(Ztr, y_source[tr_s])
                probs["SpatialPCA32"][te_t] = clf.predict_proba(
                    Zte_t
                )[:, 1]
                thresholds["SpatialPCA32"][te_t] = thr

                # M2 + SpatialPCA32.
                Xtr_m2 = matrix(source.iloc[tr_s], M2)
                Xte_m2 = matrix(target.iloc[te_t], M2)

                Ctr = np.column_stack([Xtr_m2, Ztr])
                Cte = np.column_stack([Xte_m2, Zte_t])

                inner_p = grouped_oof_lr(
                    Ctr,
                    y_source[tr_s],
                    g_source[tr_s],
                    seed + 3000 + fold,
                )
                thr = threshold_for_recall(
                    y_source[tr_s],
                    inner_p,
                    TARGET_RECALL,
                )
                clf = build_lr(seed + 30000 + fold)
                clf.fit(Ctr, y_source[tr_s])
                probs["M2_plus_SpatialPCA32"][te_t] = clf.predict_proba(
                    Cte
                )[:, 1]
                thresholds["M2_plus_SpatialPCA32"][te_t] = thr

                fold_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "fold": fold,
                    "train_cases": len(train_cases),
                    "test_cases": len(test_cases),
                    "pca_explained_variance_32": explained,
                    "pca_fit_source_train_rows": len(tr_s),
                    "target_rows_used_to_fit_pca": 0,
                })

            print(f"\n######## TARGET={target_family} SEED={seed} ########")

            for method in methods:
                if np.isnan(probs[method]).any():
                    raise AssertionError(
                        f"Incomplete probabilities: "
                        f"{target_family}/{seed}/{method}"
                    )
                if np.isnan(thresholds[method]).any():
                    raise AssertionError(
                        f"Incomplete thresholds: "
                        f"{target_family}/{seed}/{method}"
                    )

                p = probs[method]

                auc = float(roc_auc_score(y_target, p))
                auprc = float(average_precision_score(y_target, p))

                pred = (p >= thresholds[method]).astype(int)
                tp = int(np.sum((pred == 1) & (y_target == 1)))
                fp = int(np.sum((pred == 1) & (y_target == 0)))
                tn = int(np.sum((pred == 0) & (y_target == 0)))
                fn = int(np.sum((pred == 0) & (y_target == 1)))

                recall = tp / (tp + fn)
                fpr = fp / (fp + tn)

                pi = PPV_PREVALENCE
                denom = recall * pi + fpr * (1 - pi)
                ppv1 = recall * pi / denom if denom > 0 else np.nan

                oracle_thr = threshold_for_recall(
                    y_target,
                    p,
                    TARGET_RECALL,
                )
                oracle = op_metrics(
                    y_target,
                    p,
                    oracle_thr,
                )

                print(
                    f"{method:24s} "
                    f"AUROC={auc:.6f} "
                    f"AUPRC={auprc:.6f} "
                    f"TRANSFER_RECALL={recall:.6f} "
                    f"TRANSFER_FPR={fpr:.6f} "
                    f"PPV1%={ppv1:.6f} "
                    f"ORACLE_R90_FPR={oracle['FPR']:.6f}"
                )

                result_rows.append({
                    "target_family": target_family,
                    "seed": seed,
                    "method": method,
                    "target_AUROC": auc,
                    "target_AUPRC": auprc,
                    "transfer_Recall": recall,
                    "transfer_FPR": fpr,
                    "transfer_PPV1pct": ppv1,
                    "oracle_R90_FPR": oracle["FPR"],
                    "oracle_R90_PPV1pct": oracle["PPV1pct"],
                })

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)

    by_family = (
        results.groupby(["target_family", "method"], sort=False)
        .agg(
            seeds=("seed", "count"),
            target_AUROC_mean=("target_AUROC", "mean"),
            target_AUROC_std=("target_AUROC", "std"),
            target_AUPRC_mean=("target_AUPRC", "mean"),
            target_AUPRC_std=("target_AUPRC", "std"),
            transfer_Recall_mean=("transfer_Recall", "mean"),
            transfer_Recall_std=("transfer_Recall", "std"),
            transfer_FPR_mean=("transfer_FPR", "mean"),
            transfer_FPR_std=("transfer_FPR", "std"),
            transfer_PPV1pct_mean=("transfer_PPV1pct", "mean"),
            transfer_PPV1pct_std=("transfer_PPV1pct", "std"),
            oracle_R90_FPR_mean=("oracle_R90_FPR", "mean"),
            oracle_R90_FPR_std=("oracle_R90_FPR", "std"),
        )
        .reset_index()
    )

    macro_seed = (
        results.groupby(["method", "seed"], sort=False)
        .agg(
            macro_AUROC=("target_AUROC", "mean"),
            macro_AUPRC=("target_AUPRC", "mean"),
            macro_transfer_Recall=("transfer_Recall", "mean"),
            macro_transfer_FPR=("transfer_FPR", "mean"),
            macro_PPV1pct=("transfer_PPV1pct", "mean"),
            macro_oracle_R90_FPR=("oracle_R90_FPR", "mean"),
        )
        .reset_index()
    )

    macro = (
        macro_seed.groupby("method", sort=False)
        .agg(
            seeds=("seed", "count"),
            macro_AUROC_mean=("macro_AUROC", "mean"),
            macro_AUROC_std=("macro_AUROC", "std"),
            macro_AUPRC_mean=("macro_AUPRC", "mean"),
            macro_AUPRC_std=("macro_AUPRC", "std"),
            macro_transfer_Recall_mean=("macro_transfer_Recall", "mean"),
            macro_transfer_Recall_std=("macro_transfer_Recall", "std"),
            macro_transfer_FPR_mean=("macro_transfer_FPR", "mean"),
            macro_transfer_FPR_std=("macro_transfer_FPR", "std"),
            macro_PPV1pct_mean=("macro_PPV1pct", "mean"),
            macro_PPV1pct_std=("macro_PPV1pct", "std"),
            macro_oracle_R90_FPR_mean=("macro_oracle_R90_FPR", "mean"),
            macro_oracle_R90_FPR_std=("macro_oracle_R90_FPR", "std"),
        )
        .reset_index()
    )

    print("\n\n===== R10K0 BY-FAMILY SUMMARY =====")
    print(by_family.to_string(index=False))

    print("\n===== R10K0 MACRO SUMMARY =====")
    print(macro.to_string(index=False))

    base = macro[macro["method"] == "M2_backbone"].iloc[0]
    cand = macro[
        macro["method"] == "M2_plus_SpatialPCA32"
    ].iloc[0]

    base_family = by_family[
        by_family["method"] == "M2_backbone"
    ].set_index("target_family")
    cand_family = by_family[
        by_family["method"] == "M2_plus_SpatialPCA32"
    ].set_index("target_family")

    delta_auc = float(
        cand["macro_AUROC_mean"] - base["macro_AUROC_mean"]
    )
    delta_fpr = float(
        cand["macro_transfer_FPR_mean"]
        - base["macro_transfer_FPR_mean"]
    )
    delta_oracle = float(
        cand["macro_oracle_R90_FPR_mean"]
        - base["macro_oracle_R90_FPR_mean"]
    )
    worst_auc = float(cand_family["target_AUROC_mean"].min())
    worst_delta = float(
        (
            cand_family["target_AUROC_mean"]
            - base_family["target_AUROC_mean"]
        ).min()
    )

    print("\n===== PRIMARY SPATIAL DELTAS: M2+PCA32 vs M2 =====")
    print("Delta macro AUROC:", delta_auc)
    print("Delta transferred FPR:", delta_fpr)
    print("Delta oracle R90 FPR:", delta_oracle)
    print("Worst-family candidate AUROC:", worst_auc)
    print("Worst-family AUROC delta:", worst_delta)

    if (
        delta_auc >= 0.02
        and delta_fpr <= -0.04
        and delta_oracle <= -0.03
        and float(cand["macro_transfer_Recall_mean"]) >= 0.85
        and worst_auc >= 0.75
        and worst_delta >= -0.01
    ):
        decision = "SPATIAL_MASK_SUBSPACE_PROMISING"
    elif (
        (
            delta_auc >= 0.01
            or delta_fpr <= -0.03
            or delta_oracle <= -0.03
        )
        and float(cand["macro_transfer_Recall_mean"]) >= 0.85
        and worst_delta >= -0.02
    ):
        decision = "SPATIAL_MASK_SUBSPACE_PARTIALLY_SUPPORTED"
    else:
        decision = "SPATIAL_MASK_SUBSPACE_NOT_SUPPORTED"

    print("\n===== FINAL DECISION =====")
    print("DECISION:", decision)

    results.to_csv(
        out / "R10K0_per_seed_family_metrics.csv",
        index=False,
    )
    by_family.to_csv(
        out / "R10K0_by_family_summary.csv",
        index=False,
    )
    macro.to_csv(
        out / "R10K0_macro_summary.csv",
        index=False,
    )
    folds.to_csv(
        out / "R10K0_fold_audit.csv",
        index=False,
    )

    print("\nOutputs:")
    print(out / "R10K0_source_mask_occupancy32.npz")
    print(out / "R10K0_per_seed_family_metrics.csv")
    print(out / "R10K0_by_family_summary.csv")
    print(out / "R10K0_macro_summary.csv")
    print(out / "R10K0_fold_audit.csv")
    print("PASS")


if __name__ == "__main__":
    main()
