#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM7A_promise12_external_ranking_baseline_audit_fix1.py

SafeTTA Q1 enhancement — CM7A.

Purpose
-------
Now that CM6 has completed the FIRST prospective PROMISE12 GT reveal, perform
one fixed-protocol POST-GT baseline audit to answer the main remaining Q1
reviewer question:

    Is the frozen prediction-conditioned Final risk score actually more useful
    than simple uncertainty / morphology / unconditioned semantic baselines on
    the independent MRI target?

Important scope
---------------
CM7A is NOT another prospective endpoint and must not be described as such.
The prospective target endpoint remains the already locked CM6 result.

CM7A does not tune:
- segmentation models;
- TTA;
- HARM definition;
- Final safety estimator;
- any deployed threshold.

Baseline definitions are frozen in this script before execution and use no
PROMISE12 GT during baseline-score construction.

Baselines
---------
1) SOURCE_ENTROPY
   Mean pixelwise Bernoulli entropy from frozen SOURCE eval logits.

2) SOURCE_LOW_CONFIDENCE
   Mean [1 - max(p,1-p)] from frozen SOURCE eval logits.

3) M2
   source_fg_fraction + source_boundary_density.
   SOURCE-only median imputer + StandardScaler + balanced LR.

4) DINO_CLS
   frozen DINOv2-base [CLS] 768.
   SOURCE-only median imputer + randomized whiten PCA64 +
   StandardScaler + balanced LR.

5) CondDINO
   frozen prediction-conditioned FG768+BG768.
   SOURCE-only median imputer + randomized whiten PCA64 +
   StandardScaler + balanced LR.

6) Final
   Prospectively frozen CM5A/CM4C score.
   No refitting here.

For M2 / DINO_CLS / CondDINO, one pooled all-SOURCE head is fitted across all
three model-family rows, matching CM4B's pooled safety-head design. Target GT is
not loaded until all baseline target scores have been saved and hashed.

Primary comparison
------------------
Target macro-family HARM AUROC.

Statistics
----------
- family AUROC/AUPRC/Spearman;
- macro-family AUROC/AUPRC;
- pooled AUROC/AUPRC/Spearman;
- 2000x paired patient-cluster bootstrap;
- paired Delta AUROC = Final - baseline with 95% CI.

No threshold analysis occurs in CM7A.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM7A-v1-fix2"
BUILD = "Q1X_CM7A_PROMISE12_EXTERNAL_RANKING_BASELINE_AUDIT_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM3_LOCK = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
    / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
)
EXPECTED_CM3_LOCK_SHA = (
    "4cfbe8f108f53e014973111d96244c666bfabd13eb022675fa31a34edaae7cec"
)

CM4B_LOCK = (
    OUT
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1_v1"
    / "CM4B_MRI_SOURCE_SAFETY_OOF_LOCK.json"
)
EXPECTED_CM4B_LOCK_SHA = (
    "7903f8dd84338be8051883b1a905607daec25e55539205f7c6fb78ca3cc41a7e"
)

CM4D_LOCK = (
    OUT
    / "Q1X_CM4D_final_all_source_segmentation_models_lock_fix2_v1"
    / "CM4D_FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCK.json"
)
EXPECTED_CM4D_LOCK_SHA = (
    "fbc55b3d09276e214491e115ca90dff72599222fa7ee92a584c8083de0afbfa7"
)

CM5A_LOCK = (
    OUT
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1"
    / "CM5A_PROMISE12_GT_FREE_PREDICTION_SAFETY_LOCK.json"
)
EXPECTED_CM5A_LOCK_SHA = (
    "18198bf417ec30434fedcab39f8f50bff270fade676e47fccfa1e397fc2c93fd"
)

CM6_LOCK = (
    OUT
    / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
    / "CM6_PROMISE12_GT_REVEAL_LOCKED_EVALUATION_LOCK.json"
)
EXPECTED_CM6_LOCK_SHA = (
    "151b067b73ab690694f40f8f72d06fc0414de56b5763f1bc31a8eb478db4f4a7"
)

CM3_HELPER = (
    CODE
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py"
)
EXPECTED_CM3_HELPER_SHA = (
    "443e371c1a71173cc9fd3eaa4876cb8721a00497836e06d902b50da12cb9f57a"
)

CM4A_HELPER = (
    CODE
    / "Q1X_CM4A_source_oof_tent1_outcome_asset_lock_fix4.py"
)
EXPECTED_CM4A_HELPER_SHA = (
    "cd4af7a78fcfc72ff4a12df43578757f095d6a0cb3a6445c8b6c25e50e6ca6c0"
)

CM4B_HELPER = (
    CODE
    / "Q1X_CM4B_mri_source_prediction_conditioned_safety_grouped_oof_fix1.py"
)
EXPECTED_CM4B_HELPER_SHA = (
    "0b894a2db0d58dda6fa9fe26bf0b1482a3da89d25918405e45e39346d132cc40"
)

CM5A_HELPER = (
    CODE
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1.py"
)
EXPECTED_CM5A_HELPER_SHA = (
    "e7373152880068157b88cba31f091258932c9366e53313476692ab81c5c50f3b"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM7A_promise12_external_ranking_baseline_audit_fix2_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]

N_SOURCE_PATIENTS = 139
N_SOURCE_SLICES = 3553
N_TARGET_PATIENTS = 50
N_TARGET_SLICES = 1377

RAW_DIM = 1538
COND_DIM = 1536
M2_DIM = 2
DINO_DIM = 768
PCA_COMPONENTS = 64

SEG_BATCH_SIZE = 8
DINO_BATCH_SIZE = 16

LR_MAX_ITER = 3000
SEED = 20260904

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260907

METHODS = [
    "SOURCE_ENTROPY",
    "SOURCE_LOW_CONFIDENCE",
    "M2",
    "DINO_CLS",
    "CondDINO",
    "Final",
]

PASS_DECISION = (
    "PROMISE12_POST_GT_BASELINE_AUDIT_COMPLETE_READY_FOR_CM7B_MATCHED_COVERAGE_UTILITY"
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


def import_helper(label, path, expected_sha, module_name):
    if not path.is_file():
        raise FileNotFoundError(path)

    got = sha256_file(path)
    print(label, got, "PASS" if got == expected_sha else "FAIL")
    if got != expected_sha:
        raise RuntimeError(f"{label} SHA mismatch.")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {label}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_lineage():
    for label, path, expected in [
        ("CM3_LOCK", CM3_LOCK, EXPECTED_CM3_LOCK_SHA),
        ("CM4B_LOCK", CM4B_LOCK, EXPECTED_CM4B_LOCK_SHA),
        ("CM4D_LOCK", CM4D_LOCK, EXPECTED_CM4D_LOCK_SHA),
        ("CM5A_LOCK", CM5A_LOCK, EXPECTED_CM5A_LOCK_SHA),
        ("CM6_LOCK", CM6_LOCK, EXPECTED_CM6_LOCK_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    cm4b = load_json(CM4B_LOCK)
    cm4d = load_json(CM4D_LOCK)
    cm5a = load_json(CM5A_LOCK)
    cm6 = load_json(CM6_LOCK)

    if cm3.get("status") != "PASS":
        raise RuntimeError("CM3 status changed.")
    if cm4b.get("status") != "PASS":
        raise RuntimeError("CM4B status changed.")
    if cm4d.get("status") != "PASS":
        raise RuntimeError("CM4D status changed.")
    if cm5a.get("status") != "PASS":
        raise RuntimeError("CM5A status changed.")
    if cm6.get("status") != "PASS":
        raise RuntimeError("CM6 status changed.")

    return cm3, cm4b, cm4d, cm5a, cm6


def load_source_raw_features_and_labels(cm4b):
    score_path = Path(cm4b["artifacts"]["scores"])
    if sha256_file(score_path) != cm4b["artifacts"]["scores_sha256"]:
        raise RuntimeError("CM4B SOURCE score SHA mismatch.")

    source_rows = pd.read_csv(score_path)

    required = {
        "family",
        "global_index",
        "case_key",
        "harmful",
    }
    missing = required.difference(source_rows.columns)
    if missing:
        raise RuntimeError(
            f"Missing CM4B SOURCE columns: {sorted(missing)}"
        )

    if len(source_rows) != len(FAMILIES) * N_SOURCE_SLICES:
        raise RuntimeError("Unexpected CM4B SOURCE row count.")
    if source_rows["case_key"].nunique() != N_SOURCE_PATIENTS:
        raise RuntimeError("Unexpected CM4B SOURCE patient count.")

    feature_meta_path = Path(cm4b["artifacts"]["feature_meta"])
    if sha256_file(feature_meta_path) != cm4b[
        "artifacts"
    ]["feature_meta_sha256"]:
        raise RuntimeError("CM4B feature-meta SHA mismatch.")

    feature_meta = load_json(feature_meta_path)
    feature_path = Path(feature_meta["feature_file"])

    if sha256_file(feature_path) != feature_meta["feature_sha256"]:
        raise RuntimeError("CM4B SOURCE feature SHA mismatch.")

    raw = np.load(feature_path, mmap_mode="r")
    expected_shape = (
        len(FAMILIES),
        N_SOURCE_SLICES,
        RAW_DIM,
    )
    if tuple(raw.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SOURCE feature shape {raw.shape}"
        )

    X = np.empty(
        (len(source_rows), RAW_DIM),
        dtype=np.float32,
    )

    family_to_idx = {
        family: i for i, family in enumerate(FAMILIES)
    }

    for family in FAMILIES:
        mask = source_rows["family"].to_numpy() == family
        gi = source_rows.loc[
            mask,
            "global_index",
        ].to_numpy(dtype=np.int64)

        if not np.all((gi >= 0) & (gi < N_SOURCE_SLICES)):
            raise RuntimeError(
                f"{family} invalid SOURCE global_index."
            )

        X[np.flatnonzero(mask)] = np.asarray(
            raw[family_to_idx[family], gi, :],
            dtype=np.float32,
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite SOURCE raw features.")

    return source_rows, X


def load_target_raw_features_and_rows(cm5a):
    score_path = Path(cm5a["artifacts"]["target_scores"])
    if sha256_file(score_path) != cm5a[
        "artifacts"
    ]["target_scores_sha256"]:
        raise RuntimeError("CM5A target score SHA mismatch.")

    target_rows = pd.read_csv(score_path)

    required = {
        "family",
        "global_index",
        "case_key",
        "slice_index",
        "risk_score",
    }
    missing = required.difference(target_rows.columns)
    if missing:
        raise RuntimeError(
            f"Missing CM5A target columns: {sorted(missing)}"
        )

    forbidden = {
        "harmful",
        "beneficial",
        "delta_dice",
        "source_dice",
        "tent1_dice",
    }
    if forbidden.intersection(target_rows.columns):
        raise RuntimeError(
            "CM5A target score table unexpectedly contains GT outcomes."
        )

    if len(target_rows) != len(FAMILIES) * N_TARGET_SLICES:
        raise RuntimeError("Unexpected CM5A target row count.")
    if target_rows["case_key"].nunique() != N_TARGET_PATIENTS:
        raise RuntimeError("Unexpected CM5A target patient count.")

    feature_path = Path(cm5a["artifacts"]["target_features"])
    if sha256_file(feature_path) != cm5a[
        "artifacts"
    ]["target_features_sha256"]:
        raise RuntimeError("CM5A target feature SHA mismatch.")

    raw = np.load(feature_path, mmap_mode="r")
    expected_shape = (
        len(FAMILIES),
        N_TARGET_SLICES,
        RAW_DIM,
    )
    if tuple(raw.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected target feature shape {raw.shape}"
        )

    X = np.empty(
        (len(target_rows), RAW_DIM),
        dtype=np.float32,
    )

    family_to_idx = {
        family: i for i, family in enumerate(FAMILIES)
    }

    for family in FAMILIES:
        mask = target_rows["family"].to_numpy() == family
        gi = target_rows.loc[
            mask,
            "global_index",
        ].to_numpy(dtype=np.int64)

        if not np.all((gi >= 0) & (gi < N_TARGET_SLICES)):
            raise RuntimeError(
                f"{family} invalid target global_index."
            )

        X[np.flatnonzero(mask)] = np.asarray(
            raw[family_to_idx[family], gi, :],
            dtype=np.float32,
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite target raw features.")

    return target_rows, X


def load_source_image_cache(cm3):
    meta_path = Path(cm3["source_cache_meta"])
    if sha256_file(meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("CM3 SOURCE cache-meta SHA mismatch.")

    meta = load_json(meta_path)
    cache_dir = meta_path.parent
    image_path = cache_dir / "source_images_f16.npy"

    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if sha256_file(image_path) != meta["images_sha256"]:
        raise RuntimeError("CM3 SOURCE image-cache SHA mismatch.")

    images = np.load(image_path, mmap_mode="r")
    if images.shape != (N_SOURCE_SLICES, 352, 352):
        raise RuntimeError(
            f"Unexpected SOURCE image cache shape {images.shape}"
        )

    return images


def load_target_image_cache(cm5a):
    meta_path = Path(cm5a["artifacts"]["target_cache_meta"])
    if sha256_file(meta_path) != cm5a[
        "artifacts"
    ]["target_cache_meta_sha256"]:
        raise RuntimeError("CM5A target cache-meta SHA mismatch.")

    meta = load_json(meta_path)
    image_path = Path(meta["images"])

    if sha256_file(image_path) != meta["images_sha256"]:
        raise RuntimeError("CM5A target image-cache SHA mismatch.")

    images = np.load(image_path, mmap_mode="r")
    if images.shape != (N_TARGET_SLICES, 352, 352):
        raise RuntimeError(
            f"Unexpected target image cache shape {images.shape}"
        )

    return images


def extract_dino_cls(
    cm4b_helper,
    dino_lock,
    source_images,
    target_images,
    stage_dir,
    torch,
    F,
    device,
):
    source_path = stage_dir / "CM7A_SOURCE_DINO_CLS_F16.npy"
    target_path = stage_dir / "CM7A_TARGET_DINO_CLS_F16.npy"
    meta_path = stage_dir / "CM7A_DINO_CLS_META.json"

    if (
        source_path.is_file()
        and target_path.is_file()
        and meta_path.is_file()
    ):
        meta = load_json(meta_path)
        valid = (
            meta.get("status") == "PASS"
            and meta.get("source_sha256") == sha256_file(source_path)
            and meta.get("target_sha256") == sha256_file(target_path)
        )
        if valid:
            print("DINO CLS=REUSE PASS")
            return (
                np.load(source_path, mmap_mode="r"),
                np.load(target_path, mmap_mode="r"),
                meta,
            )

    model = cm4b_helper.load_dino_model(
        dino_lock,
        device,
    )

    src = np.lib.format.open_memmap(
        source_path,
        mode="w+",
        dtype=np.float16,
        shape=(N_SOURCE_SLICES, DINO_DIM),
    )
    tgt = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float16,
        shape=(N_TARGET_SLICES, DINO_DIM),
    )

    for label, images, out_arr, n in [
        ("SOURCE", source_images, src, N_SOURCE_SLICES),
        ("TARGET", target_images, tgt, N_TARGET_SLICES),
    ]:
        for start in tqdm(
            range(0, n, DINO_BATCH_SIZE),
            total=(n + DINO_BATCH_SIZE - 1) // DINO_BATCH_SIZE,
            desc=f"DINO CLS {label}",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + DINO_BATCH_SIZE, n)
            x = cm4b_helper.dino_preprocess(
                images[start:end],
                torch,
                F,
                device,
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                    dtype=(
                        torch.float16
                        if device.type == "cuda"
                        else torch.bfloat16
                    ),
                ):
                    out = model(pixel_values=x)

            cls = out.last_hidden_state[:, 0, :].float()

            if cls.shape != (end - start, DINO_DIM):
                raise RuntimeError(
                    f"Unexpected DINO CLS shape {cls.shape}"
                )
            if not torch.isfinite(cls).all():
                raise RuntimeError("Nonfinite DINO CLS.")

            out_arr[start:end] = (
                cls.cpu().numpy().astype(np.float16)
            )

    src.flush()
    tgt.flush()

    meta = {
        "status": "PASS",
        "version": VERSION,
        "representation": "frozen DINOv2-base CLS token",
        "source_shape": [N_SOURCE_SLICES, DINO_DIM],
        "target_shape": [N_TARGET_SLICES, DINO_DIM],
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "target_file": str(target_path),
        "target_sha256": sha256_file(target_path),
        "target_gt_used": False,
    }
    save_json(meta_path, meta)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        np.load(source_path, mmap_mode="r"),
        np.load(target_path, mmap_mode="r"),
        meta,
    )


def compute_source_uncertainty_baselines(
    cm4a_helper,
    cm5a_helper,
    cm3_helper_module,
    cm4d,
    cm5a,
    target_images,
    stage_dir,
    torch,
    device,
):
    score_path = (
        stage_dir
        / "CM7A_TARGET_SOURCE_UNCERTAINTY_SCORES.csv"
    )
    meta_path = (
        stage_dir
        / "CM7A_SOURCE_UNCERTAINTY_META.json"
    )

    if score_path.is_file() and meta_path.is_file():
        meta = load_json(meta_path)
        if (
            meta.get("status") == "PASS"
            and meta.get("score_sha256")
            == sha256_file(score_path)
        ):
            print("SOURCE uncertainty baselines=REUSE PASS")
            return pd.read_csv(score_path), meta

    family_pred = cm5a["artifacts"]["family_predictions"]
    rows = []

    eps = 1e-7

    for family in FAMILIES:
        ckpt_path = Path(
            cm4d["final_models"][family]["checkpoint"]
        )
        ckpt_sha = cm4d["final_models"][family][
            "checkpoint_sha256"
        ]

        model, _ = cm4a_helper.build_loaded_model(
            cm3_helper_module,
            family,
            ckpt_path,
            ckpt_sha,
            torch,
            device,
        )
        model.eval()
        model.requires_grad_(False)

        packed_path = Path(
            family_pred[family]["source_masks"]
        )
        if sha256_file(packed_path) != family_pred[
            family
        ]["source_masks_sha256"]:
            raise RuntimeError(
                f"{family} CM5A SOURCE-mask SHA mismatch."
            )
        packed = np.load(packed_path, mmap_mode="r")

        mismatch_total = 0

        for start in tqdm(
            range(0, N_TARGET_SLICES, SEG_BATCH_SIZE),
            total=(N_TARGET_SLICES + SEG_BATCH_SIZE - 1) // SEG_BATCH_SIZE,
            desc=f"{family} SOURCE uncertainty",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + SEG_BATCH_SIZE, N_TARGET_SLICES)

            x = cm5a_helper.model_input_from_numpy(
                target_images[start:end],
                torch,
                device,
            )

            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                    dtype=(
                        torch.float16
                        if device.type == "cuda"
                        else torch.bfloat16
                    ),
                ):
                    logits = model(x)

            logits_f = logits.detach().float()
            prob = torch.sigmoid(logits_f)

            entropy = -(
                prob * torch.log(prob.clamp_min(eps))
                + (1.0 - prob)
                * torch.log((1.0 - prob).clamp_min(eps))
            ).mean(dim=(1, 2, 3))

            low_conf = (
                1.0
                - torch.maximum(prob, 1.0 - prob)
            ).mean(dim=(1, 2, 3))

            pred = (
                logits_f[:, 0].cpu().numpy() >= 0.0
            )
            locked = cm5a_helper.unpack_masks(
                np.asarray(packed[start:end])
            ).astype(bool)

            mismatch_total += int(
                np.count_nonzero(pred != locked)
            )

            ent_np = entropy.cpu().numpy()
            low_np = low_conf.cpu().numpy()

            for j, gi in enumerate(range(start, end)):
                rows.append({
                    "family": family,
                    "global_index": gi,
                    "SOURCE_ENTROPY": float(ent_np[j]),
                    "SOURCE_LOW_CONFIDENCE": float(low_np[j]),
                })

        if mismatch_total != 0:
            raise RuntimeError(
                f"{family} SOURCE inference does not exactly reproduce "
                f"CM5A masks: pixel_mismatch={mismatch_total}"
            )

        print(
            f"{family} SOURCE mask replay: PASS mismatch=0"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(
        ["family", "global_index"]
    ).reset_index(drop=True)

    if len(df) != len(FAMILIES) * N_TARGET_SLICES:
        raise RuntimeError("Uncertainty-score row count changed.")

    df.to_csv(score_path, index=False)

    meta = {
        "status": "PASS",
        "version": VERSION,
        "score_file": str(score_path),
        "score_sha256": sha256_file(score_path),
        "source_mask_replay_pixel_mismatch": 0,
        "definitions": {
            "SOURCE_ENTROPY": (
                "mean pixelwise Bernoulli entropy of frozen SOURCE eval logits"
            ),
            "SOURCE_LOW_CONFIDENCE": (
                "mean [1-max(p,1-p)] of frozen SOURCE eval logits"
            ),
        },
        "target_gt_used": False,
    }
    save_json(meta_path, meta)

    return df, meta


def fit_source_only_probability_head(
    X_source,
    y_source,
    X_target,
    representation,
):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if representation == "M2":
        imp = SimpleImputer(strategy="median")
        tr = imp.fit_transform(
            X_source[:, COND_DIM:RAW_DIM]
        )
        te = imp.transform(
            X_target[:, COND_DIM:RAW_DIM]
        )

        scaler = StandardScaler()
        tr = scaler.fit_transform(tr)
        te = scaler.transform(te)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=SEED,
        )
        clf.fit(tr, y_source)
        return clf.predict_proba(te)[:, 1]

    if representation == "CondDINO":
        imp = SimpleImputer(strategy="median")
        tr_cond = imp.fit_transform(
            X_source[:, :COND_DIM]
        )
        te_cond = imp.transform(
            X_target[:, :COND_DIM]
        )

    elif representation == "DINO_CLS":
        imp = SimpleImputer(strategy="median")
        tr_cond = imp.fit_transform(X_source)
        te_cond = imp.transform(X_target)

    else:
        raise ValueError(representation)

    pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )
    tr = pca.fit_transform(tr_cond)
    te = pca.transform(te_cond)

    scaler = StandardScaler()
    tr = scaler.fit_transform(tr)
    te = scaler.transform(te)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    clf.fit(tr, y_source)

    return clf.predict_proba(te)[:, 1]


def make_cls_row_features(
    rows,
    cls_by_slice,
    n_slices,
):
    X = np.empty(
        (len(rows), DINO_DIM),
        dtype=np.float32,
    )

    gi = rows["global_index"].to_numpy(dtype=np.int64)
    if not np.all((gi >= 0) & (gi < n_slices)):
        raise RuntimeError("Invalid CLS row global_index.")

    X[:] = np.asarray(
        cls_by_slice[gi],
        dtype=np.float32,
    )

    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite row-aligned CLS features.")

    return X


def construct_target_baseline_scores_before_gt_loading(
    source_rows,
    source_raw,
    target_rows,
    target_raw,
    source_cls,
    target_cls,
    uncertainty_df,
    stage_dir,
):
    """
    IMPORTANT: this function is called before the CM6 outcome table is loaded.
    It uses SOURCE HARM labels but no target GT/outcome columns.
    """
    y_source = source_rows["harmful"].to_numpy(dtype=np.int64)

    source_cls_rows = make_cls_row_features(
        source_rows,
        source_cls,
        N_SOURCE_SLICES,
    )
    target_cls_rows = make_cls_row_features(
        target_rows,
        target_cls,
        N_TARGET_SLICES,
    )

    m2_score = fit_source_only_probability_head(
        source_raw,
        y_source,
        target_raw,
        "M2",
    )
    cls_score = fit_source_only_probability_head(
        source_cls_rows,
        y_source,
        target_cls_rows,
        "DINO_CLS",
    )
    cond_score = fit_source_only_probability_head(
        source_raw,
        y_source,
        target_raw,
        "CondDINO",
    )

    key_cols = [
        "family",
        "global_index",
        "case_key",
        "slice_index",
    ]
    out = target_rows[key_cols].copy()

    # Merge frozen SOURCE uncertainty scores.
    out = out.merge(
        uncertainty_df,
        on=["family", "global_index"],
        how="left",
        validate="one_to_one",
    )

    out["M2"] = m2_score
    out["DINO_CLS"] = cls_score
    out["CondDINO"] = cond_score
    out["Final"] = target_rows[
        "risk_score"
    ].to_numpy(dtype=np.float64)

    for method in METHODS:
        vals = out[method].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise RuntimeError(
                f"Nonfinite target baseline score: {method}"
            )

    score_path = (
        stage_dir
        / "CM7A_TARGET_BASELINE_SCORES_LOCKED_BEFORE_OUTCOME_LOAD.csv"
    )
    out.to_csv(score_path, index=False)

    score_lock = {
        "status": "PASS",
        "version": VERSION,
        "build": BUILD,
        "scope": "POST_GT_FIXED_PROTOCOL_BASELINE_AUDIT",
        "important": (
            "Target baseline scores were constructed without loading CM6 "
            "target outcomes in this script. This does not convert CM7A into "
            "a prospective experiment because PROMISE12 GT had already been "
            "revealed in CM6."
        ),
        "fix2_engineering_correction": {
            "issue": (
                "Fix1 passed the parsed CM3 lock dictionary into "
                "CM4A build_loaded_model(), whose first argument must be "
                "the imported CM3 helper module exposing build_model()."
            ),
            "correction": (
                "Import the exact SHA-locked CM3 helper module and pass that "
                "module to CM4A build_loaded_model()."
            ),
            "scientific_method_changed": False,
            "target_gt_reused_for_baseline_construction": False,
        },
        "methods": METHODS,
        "target_rows": int(len(out)),
        "target_patients": int(out["case_key"].nunique()),
        "target_gt_used_for_score_construction": False,
        "target_threshold_tuning": False,
        "score_file": str(score_path),
        "score_sha256": sha256_file(score_path),
    }

    lock_path = (
        stage_dir
        / "CM7A_TARGET_BASELINE_SCORE_CONSTRUCTION_LOCK.json"
    )
    save_json(lock_path, score_lock)

    print("\n===== CM7A BASELINE SCORE CONSTRUCTION LOCK =====")
    print("Target GT loaded for baseline-score construction: NO")
    print("Target tuning: NO")
    print("Methods:", METHODS)
    print("SCORE SHA256:", score_lock["score_sha256"])
    print("PASS")

    return out, score_lock, lock_path


def load_cm6_outcomes_after_score_lock(cm6):
    outcome_path = Path(cm6["artifacts"]["outcomes"])
    if sha256_file(outcome_path) != cm6[
        "artifacts"
    ]["outcomes_sha256"]:
        raise RuntimeError("CM6 outcome SHA mismatch.")

    outcomes = pd.read_csv(outcome_path)

    required = {
        "family",
        "global_index",
        "case_key",
        "slice_index",
        "harmful",
        "delta_dice",
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise RuntimeError(
            f"Missing CM6 outcome columns: {sorted(missing)}"
        )

    if len(outcomes) != len(FAMILIES) * N_TARGET_SLICES:
        raise RuntimeError("Unexpected CM6 outcome row count.")
    if outcomes["case_key"].nunique() != N_TARGET_PATIENTS:
        raise RuntimeError("Unexpected CM6 outcome patient count.")

    return outcomes


def metric_table(merged):
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    rows = []

    for method in METHODS:
        for family in FAMILIES:
            sub = merged[merged["family"] == family]
            y = sub["harmful"].to_numpy(dtype=np.int64)
            s = sub[method].to_numpy(dtype=np.float64)

            rows.append({
                "method": method,
                "scope": family,
                "rows": len(sub),
                "patients": sub["case_key"].nunique(),
                "harm_count": int(y.sum()),
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(
                    average_precision_score(y, s)
                ),
                "spearman_score_vs_minus_delta": float(
                    spearmanr(
                        s,
                        -sub["delta_dice"].to_numpy(
                            dtype=np.float64
                        ),
                    ).statistic
                ),
            })

        family_rows = [
            r
            for r in rows
            if r["method"] == method
            and r["scope"] in FAMILIES
        ]

        y_all = merged["harmful"].to_numpy(dtype=np.int64)
        s_all = merged[method].to_numpy(dtype=np.float64)

        rows.append({
            "method": method,
            "scope": "MACRO_FAMILY",
            "rows": len(merged),
            "patients": merged["case_key"].nunique(),
            "harm_count": int(y_all.sum()),
            "auroc": float(
                np.mean([r["auroc"] for r in family_rows])
            ),
            "auprc": float(
                np.mean([r["auprc"] for r in family_rows])
            ),
            "spearman_score_vs_minus_delta": np.nan,
        })

        rows.append({
            "method": method,
            "scope": "POOLED",
            "rows": len(merged),
            "patients": merged["case_key"].nunique(),
            "harm_count": int(y_all.sum()),
            "auroc": float(roc_auc_score(y_all, s_all)),
            "auprc": float(
                average_precision_score(y_all, s_all)
            ),
            "spearman_score_vs_minus_delta": float(
                spearmanr(
                    s_all,
                    -merged["delta_dice"].to_numpy(
                        dtype=np.float64
                    ),
                ).statistic
            ),
        })

    return pd.DataFrame(rows)


def paired_patient_bootstrap(merged):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.asarray(
        sorted(merged["case_key"].astype(str).unique())
    )
    if len(patients) != N_TARGET_PATIENTS:
        raise RuntimeError(
            "Unexpected target patient count for bootstrap."
        )

    case = merged["case_key"].astype(str).to_numpy()
    family = merged["family"].astype(str).to_numpy()
    y = merged["harmful"].to_numpy(dtype=np.int64)

    score_map = {
        method: merged[method].to_numpy(dtype=np.float64)
        for method in METHODS
    }

    patient_to_rows = {
        p: np.flatnonzero(case == p)
        for p in patients
    }

    method_values = {
        method: []
        for method in METHODS
    }
    delta_values = {
        baseline: []
        for baseline in METHODS
        if baseline != "Final"
    }

    valid_reps = 0

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="CM7A paired patient bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        idx = np.concatenate(
            [patient_to_rows[p] for p in sampled]
        )

        macro = {}
        valid = True

        for method in METHODS:
            aucs = []

            for fam in FAMILIES:
                m = family[idx] == fam
                yy = y[idx][m]
                ss = score_map[method][idx][m]

                if len(np.unique(yy)) != 2:
                    valid = False
                    break

                aucs.append(
                    float(roc_auc_score(yy, ss))
                )

            if not valid:
                break

            macro[method] = float(np.mean(aucs))

        if not valid:
            continue

        valid_reps += 1

        for method in METHODS:
            method_values[method].append(macro[method])

        for baseline in delta_values:
            delta_values[baseline].append(
                macro["Final"] - macro[baseline]
            )

    if valid_reps < int(0.95 * BOOTSTRAP_REPS):
        raise RuntimeError(
            f"Too many invalid bootstrap reps: "
            f"{valid_reps}/{BOOTSTRAP_REPS}"
        )

    method_ci_rows = []
    for method, vals in method_values.items():
        arr = np.asarray(vals, dtype=np.float64)
        method_ci_rows.append({
            "method": method,
            "reps_valid": valid_reps,
            "bootstrap_mean_macro_auroc": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
        })

    delta_rows = []
    for baseline, vals in delta_values.items():
        arr = np.asarray(vals, dtype=np.float64)
        delta_rows.append({
            "comparison": f"Final - {baseline}",
            "baseline": baseline,
            "reps_valid": valid_reps,
            "bootstrap_mean_delta_auroc": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "final_significantly_better_95ci": bool(
                np.quantile(arr, 0.025) > 0.0
            ),
        })

    return (
        pd.DataFrame(method_ci_rows),
        pd.DataFrame(delta_rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM7A FIX2 PROMISE12 POST-GT EXTERNAL RANKING BASELINE AUDIT ====="
    )
    print(
        "FIX2_CHANGE=PASS_EXACT_CM3_HELPER_MODULE_TO_CM4A_BUILD_LOADED_MODEL"
    )
    print("SCIENTIFIC_METHOD_CHANGE=NO")
    print("PROSPECTIVE_ENDPOINT=NO")
    print("CM6_PROSPECTIVE_RESULT_REMAINS_PRIMARY=YES")
    print("TARGET_GT_USED_FOR_BASELINE_SCORE_CONSTRUCTION=NO")
    print("TARGET_TUNING=NO")
    print("THRESHOLD_ANALYSIS=NO")
    print("METHODS=", METHODS)
    print("PRIMARY_COMPARISON=paired patient-bootstrap macro-family AUROC")
    print("BOOTSTRAP_REPS=", BOOTSTRAP_REPS)

    cm3, cm4b, cm4d, cm5a, cm6 = verify_lineage()

    cm3_helper_module = import_helper(
        "CM3_HELPER",
        CM3_HELPER,
        EXPECTED_CM3_HELPER_SHA,
        "q1x_cm3_fix3_cm7a",
    )
    cm4a_helper = import_helper(
        "CM4A_HELPER",
        CM4A_HELPER,
        EXPECTED_CM4A_HELPER_SHA,
        "q1x_cm4a_fix4_cm7a",
    )
    cm4b_helper = import_helper(
        "CM4B_HELPER",
        CM4B_HELPER,
        EXPECTED_CM4B_HELPER_SHA,
        "q1x_cm4b_fix1_cm7a",
    )
    cm5a_helper = import_helper(
        "CM5A_HELPER",
        CM5A_HELPER,
        EXPECTED_CM5A_HELPER_SHA,
        "q1x_cm5a_fix1_cm7a",
    )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_rows, source_raw = (
        load_source_raw_features_and_labels(cm4b)
    )
    target_rows, target_raw = (
        load_target_raw_features_and_rows(cm5a)
    )

    source_images = load_source_image_cache(cm3)
    target_images = load_target_image_cache(cm5a)

    import torch
    import torch.nn.functional as F

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    source_cls, target_cls, cls_meta = extract_dino_cls(
        cm4b_helper,
        cm4b["dino_lock"],
        source_images,
        target_images,
        args.output_dir,
        torch,
        F,
        device,
    )

    uncertainty_df, uncertainty_meta = (
        compute_source_uncertainty_baselines(
            cm4a_helper,
            cm5a_helper,
            cm3_helper_module,
            cm4d,
            cm5a,
            target_images,
            args.output_dir,
            torch,
            device,
        )
    )

    # CRITICAL ORDER:
    # Construct and hash all target baseline scores BEFORE reading CM6 outcomes.
    baseline_scores, baseline_score_lock, baseline_score_lock_path = (
        construct_target_baseline_scores_before_gt_loading(
            source_rows,
            source_raw,
            target_rows,
            target_raw,
            source_cls,
            target_cls,
            uncertainty_df,
            args.output_dir,
        )
    )

    # Only now load target outcomes.
    outcomes = load_cm6_outcomes_after_score_lock(cm6)

    merge_keys = [
        "family",
        "global_index",
        "case_key",
        "slice_index",
    ]

    merged = baseline_scores.merge(
        outcomes[
            merge_keys
            + [
                "harmful",
                "beneficial",
                "delta_dice",
                "source_dice",
                "tent1_dice",
            ]
        ],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(FAMILIES) * N_TARGET_SLICES:
        raise RuntimeError("CM7A merged row count changed.")

    merged_path = (
        args.output_dir
        / "CM7A_TARGET_BASELINE_SCORES_WITH_LOCKED_OUTCOMES.csv"
    )
    merged.to_csv(merged_path, index=False)

    metrics = metric_table(merged)
    metric_path = (
        args.output_dir
        / "CM7A_PROMISE12_BASELINE_RANKING_METRICS.csv"
    )
    metrics.to_csv(metric_path, index=False)

    method_ci, delta_ci = paired_patient_bootstrap(merged)

    method_ci_path = (
        args.output_dir
        / "CM7A_METHOD_MACRO_AUROC_PATIENT_BOOTSTRAP.csv"
    )
    delta_ci_path = (
        args.output_dir
        / "CM7A_FINAL_VS_BASELINE_PAIRED_DELTA_AUROC_BOOTSTRAP.csv"
    )

    method_ci.to_csv(method_ci_path, index=False)
    delta_ci.to_csv(delta_ci_path, index=False)

    macro = (
        metrics[metrics["scope"] == "MACRO_FAMILY"]
        .sort_values("auroc", ascending=False)
        .reset_index(drop=True)
    )

    print("\n===== CM7A PROMISE12 MACRO-FAMILY RANKING =====")
    print(
        macro[
            ["method", "auroc", "auprc"]
        ].to_string(index=False)
    )

    print("\n===== CM7A METHOD BOOTSTRAP 95% CI =====")
    print(method_ci.to_string(index=False))

    print("\n===== CM7A PAIRED FINAL - BASELINE DELTA AUROC =====")
    print(delta_ci.to_string(index=False))

    final_macro = float(
        macro.loc[
            macro["method"] == "Final",
            "auroc",
        ].iloc[0]
    )

    lock = {
        "status": "PASS",
        "decision": PASS_DECISION,
        "version": VERSION,
        "build": BUILD,
        "scope": "POST_GT_FIXED_PROTOCOL_BASELINE_AUDIT",
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm4b_lock_sha256": EXPECTED_CM4B_LOCK_SHA,
        "cm4d_lock_sha256": EXPECTED_CM4D_LOCK_SHA,
        "cm5a_lock_sha256": EXPECTED_CM5A_LOCK_SHA,
        "cm6_lock_sha256": EXPECTED_CM6_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "cm4a_helper_sha256": EXPECTED_CM4A_HELPER_SHA,
        "cm4b_helper_sha256": EXPECTED_CM4B_HELPER_SHA,
        "cm5a_helper_sha256": EXPECTED_CM5A_HELPER_SHA,
        "methods": METHODS,
        "baseline_construction": {
            "target_gt_used": False,
            "target_tuning": False,
            "source_labels_used": True,
            "pooled_all_source_head_for": [
                "M2",
                "DINO_CLS",
                "CondDINO",
            ],
            "final_score": (
                "prospectively frozen CM5A Final risk_score"
            ),
            "source_uncertainty": (
                "recomputed from exact frozen SOURCE eval logits with "
                "binary-mask replay mismatch required to be zero"
            ),
        },
        "statistics": {
            "primary_metric": "macro-family HARM AUROC",
            "bootstrap": (
                "2000x paired target patient-cluster bootstrap"
            ),
            "final_macro_auroc": final_macro,
        },
        "interpretation_boundary": (
            "CM7A is a post-GT fixed-protocol baseline audit. "
            "It can support method-specific comparative evidence but "
            "does not replace the prospective CM6 endpoint."
        ),
        "artifacts": {
            "dino_cls_meta": str(
                args.output_dir / "CM7A_DINO_CLS_META.json"
            ),
            "dino_cls_meta_sha256": sha256_file(
                args.output_dir / "CM7A_DINO_CLS_META.json"
            ),
            "uncertainty_meta": str(
                args.output_dir / "CM7A_SOURCE_UNCERTAINTY_META.json"
            ),
            "uncertainty_meta_sha256": sha256_file(
                args.output_dir / "CM7A_SOURCE_UNCERTAINTY_META.json"
            ),
            "baseline_score_lock": str(baseline_score_lock_path),
            "baseline_score_lock_sha256": sha256_file(
                baseline_score_lock_path
            ),
            "merged_scores_outcomes": str(merged_path),
            "merged_scores_outcomes_sha256": sha256_file(
                merged_path
            ),
            "metrics": str(metric_path),
            "metrics_sha256": sha256_file(metric_path),
            "method_bootstrap": str(method_ci_path),
            "method_bootstrap_sha256": sha256_file(
                method_ci_path
            ),
            "paired_delta_bootstrap": str(delta_ci_path),
            "paired_delta_bootstrap_sha256": sha256_file(
                delta_ci_path
            ),
        },
        "next_stage": (
            "CM7B_MATCHED_COVERAGE_SELECTIVE_ADAPTATION_UTILITY_AUDIT"
        ),
    }

    lock_path = (
        args.output_dir
        / "CM7A_PROMISE12_EXTERNAL_RANKING_BASELINE_AUDIT_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM7A FINAL =====")
    print("Fix2 helper-module wiring correction only: YES")
    print("Scientific method change: NO")
    print("Prospective endpoint: NO (CM6 remains primary)")
    print("Target GT used to construct baseline scores: NO")
    print("Target tuning: NO")
    print("Threshold changes: NO")
    print("Final macro-family AUROC:", final_macro)
    print("Decision=", PASS_DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
