#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM4D_final_all_source_segmentation_models_lock_fix1.py

SafeTTA Q1 enhancement — CM4D.

Purpose
-------
Train exactly one final all-SOURCE prostate segmentation model per frozen
family before any PROMISE12 image is used.

CM2B already froze:
- one final model per family;
- all 139 SOURCE patients;
- final duration derived from SOURCE OOF only;
- PROMISE12 cannot be used for epoch/model selection.

CM4D now freezes the duration rule prospectively:
  final_epochs(family) = median of the five CM3 OOF best epochs

Frozen result from the locked CM3 lineage:
- UNet            [30,19,15,24,39] -> 24 epochs
- DeepLabV3_R50   [16,34,32,24,13] -> 24 epochs
- SegFormer_B0    [39,34,50,28,48] -> 39 epochs

No early stopping and no SOURCE validation/model selection are performed in
the all-SOURCE refit. A descriptive all-SOURCE training-set sanity Dice is
computed only after the fixed-duration training is finished and cannot change
the model.

PROMISE12 access = NO.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM4D-v1-fix1"
BUILD = "Q1X_CM4D_FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

CM3_DIR = (
    OUT
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3_v1"
)
CM3_LOCK = CM3_DIR / "CM3_SOURCE_OOF_SEGMENTATION_PANEL_LOCK.json"
EXPECTED_CM3_LOCK_SHA = (
    "4cfbe8f108f53e014973111d96244c666bfabd13eb022675fa31a34edaae7cec"
)

CM4C_DIR = (
    OUT
    / "Q1X_CM4C_final_source_safety_estimator_and_threshold_lock_fix2_v1"
)
CM4C_LOCK = CM4C_DIR / "CM4C_FINAL_SOURCE_SAFETY_ESTIMATOR_LOCK.json"
EXPECTED_CM4C_LOCK_SHA = (
    "5ddf4a6a4b44d19a1e3c8cccc8a2a9cdba71da18850a7df58cbbee83a40527ff"
)

CM3_HELPER = (
    CODE
    / "Q1X_CM3_source_oof_segmentation_training_and_prediction_lock_fix3.py"
)
EXPECTED_CM3_HELPER_SHA = (
    "443e371c1a71173cc9fd3eaa4876cb8721a00497836e06d902b50da12cb9f57a"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM4D_final_all_source_segmentation_models_lock_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]

EXPECTED_BEST_EPOCHS = {
    "UNet": [30, 19, 15, 24, 39],
    "DeepLabV3_R50": [16, 34, 32, 24, 13],
    "SegFormer_B0": [39, 34, 50, 28, 48],
}

EXPECTED_FINAL_EPOCHS = {
    "UNet": 24,
    "DeepLabV3_R50": 24,
    "SegFormer_B0": 39,
}

N_SOURCE_PATIENTS = 139
N_SOURCE_SLICES = 3553
BATCH_SIZE = 8

PASS_DECISION = (
    "FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCKED_READY_FOR_"
    "CM5_PROMISE12_GT_FREE_INFERENCE"
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


def import_cm3_helper():
    if not CM3_HELPER.is_file():
        raise FileNotFoundError(CM3_HELPER)

    got = sha256_file(CM3_HELPER)
    print(
        "CM3_HELPER",
        got,
        "PASS" if got == EXPECTED_CM3_HELPER_SHA else "FAIL",
    )
    if got != EXPECTED_CM3_HELPER_SHA:
        raise RuntimeError("CM3 helper SHA mismatch.")

    spec = importlib.util.spec_from_file_location(
        "q1x_cm3_fix3_helper",
        str(CM3_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import exact CM3 helper.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_lineage():
    for label, path, expected in [
        ("CM3_LOCK", CM3_LOCK, EXPECTED_CM3_LOCK_SHA),
        ("CM4C_LOCK", CM4C_LOCK, EXPECTED_CM4C_LOCK_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    cm3 = load_json(CM3_LOCK)
    cm4c = load_json(CM4C_LOCK)

    if cm3.get("status") != "PASS":
        raise RuntimeError("CM3 status changed.")
    if cm4c.get("status") != "PASS":
        raise RuntimeError("CM4C status changed.")

    if cm3.get("source_patient_count") != N_SOURCE_PATIENTS:
        raise RuntimeError("CM3 source patient count changed.")
    if cm3.get("source_slice_count") != N_SOURCE_SLICES:
        raise RuntimeError("CM3 source slice count changed.")
    if cm3.get("batch_size") != BATCH_SIZE:
        raise RuntimeError("CM3 frozen batch size changed.")
    if cm3.get("promises12_gt_access") is not False:
        raise RuntimeError("CM3 information boundary changed.")

    if cm4c.get("information_boundary", {}).get("promises12_access") is not False:
        raise RuntimeError("CM4C information boundary changed.")
    if cm4c.get("information_boundary", {}).get("target_gt") is not False:
        raise RuntimeError("CM4C target-GT boundary changed.")

    return cm3, cm4c


def freeze_final_epoch_rule(cm3):
    print("\n===== FINAL TRAINING DURATION LOCK =====")
    print("RULE=integer median of five CM3 SOURCE OOF best epochs")

    derived = {}

    for family in FAMILIES:
        got = [
            int(v)
            for v in cm3["family_oof_artifacts"][family]["best_epochs"]
        ]

        expected = EXPECTED_BEST_EPOCHS[family]
        if got != expected:
            raise RuntimeError(
                f"{family}: best-epoch lineage changed: "
                f"{got} != {expected}"
            )

        med = float(np.median(np.asarray(got, dtype=np.float64)))
        if not med.is_integer():
            raise RuntimeError(
                f"{family}: non-integer median best epoch {med}"
            )

        final_epoch = int(med)
        if final_epoch != EXPECTED_FINAL_EPOCHS[family]:
            raise RuntimeError(
                f"{family}: derived final epoch {final_epoch} != "
                f"{EXPECTED_FINAL_EPOCHS[family]}"
            )

        derived[family] = final_epoch

        print(
            f"{family}: best_epochs={got} "
            f"median={final_epoch} PASS"
        )

    return derived


def load_source_cache(cm3):
    meta_path = Path(cm3["source_cache_meta"])

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if sha256_file(meta_path) != cm3["source_cache_meta_sha256"]:
        raise RuntimeError("SOURCE cache-meta SHA mismatch.")

    meta = load_json(meta_path)
    cache_dir = meta_path.parent

    images_path = cache_dir / "source_images_f16.npy"
    masks_path = cache_dir / "source_masks_u8.npy"
    slices_path = cache_dir / "source_slices.csv"

    for path, key in [
        (images_path, "images_sha256"),
        (masks_path, "masks_sha256"),
        (slices_path, "slices_sha256"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != meta[key]:
            raise RuntimeError(
                f"SOURCE cache SHA mismatch: {path.name}"
            )

    slice_df = pd.read_csv(slices_path)

    if len(slice_df) != N_SOURCE_SLICES:
        raise RuntimeError(
            f"Unexpected SOURCE slices: {len(slice_df)}"
        )
    if slice_df["case_key"].nunique() != N_SOURCE_PATIENTS:
        raise RuntimeError("Unexpected SOURCE patient count.")

    return images_path, masks_path, slices_path, slice_df, meta_path


def final_family_seed(helper, family):
    """
    Reuse the already-frozen family fold-0 seed.
    This is a SOURCE-only deterministic convention and is not selected by
    final-model performance or target data.
    """
    if int(helper.BASE_SEED) != 20260904:
        raise RuntimeError(
            f"Unexpected CM3 BASE_SEED {helper.BASE_SEED}"
        )

    return int(
        helper.BASE_SEED
        + helper.FAMILIES.index(family) * 100
    )


def source_training_sanity_eval(
    helper,
    torch,
    DataLoader,
    model,
    DatasetClass,
    images_path,
    masks_path,
    slice_df,
    device,
):
    """
    Descriptive in-sample evaluation after fixed training.
    It is never used for epoch/model selection.
    """
    indices = np.arange(len(slice_df), dtype=np.int64)
    ds = DatasetClass(
        images_path,
        masks_path,
        slice_df,
        indices,
        training=False,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    patient_counts = {
        str(k): {"tp": 0, "fp": 0, "fn": 0}
        for k in sorted(slice_df["case_key"].astype(str).unique())
    }

    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="all-SOURCE sanity inference",
            unit="batch",
            dynamic_ncols=True,
        ):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].cpu().numpy()[:, 0] > 0

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

            pred = (
                logits.detach().float().cpu().numpy()[:, 0] >= 0.0
            )

            case_keys = [str(v) for v in batch["case_key"]]

            for i, case_key in enumerate(case_keys):
                p = pred[i]
                t = y[i]

                patient_counts[case_key]["tp"] += int(
                    np.logical_and(p, t).sum()
                )
                patient_counts[case_key]["fp"] += int(
                    np.logical_and(p, ~t).sum()
                )
                patient_counts[case_key]["fn"] += int(
                    np.logical_and(~p, t).sum()
                )

    rows = []
    for case_key, c in patient_counts.items():
        den = 2 * c["tp"] + c["fp"] + c["fn"]
        dice = 1.0 if den == 0 else (2.0 * c["tp"]) / den
        rows.append(
            {
                "case_key": case_key,
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "dice_3d": float(dice),
            }
        )

    df = pd.DataFrame(rows)
    return float(df["dice_3d"].mean()), df


def train_final_family(
    helper,
    torch,
    DataLoader,
    DatasetClass,
    family,
    final_epochs,
    images_path,
    masks_path,
    slice_df,
    device,
    stage_dir,
):
    family_dir = stage_dir / "models" / family
    family_dir.mkdir(parents=True, exist_ok=True)

    last_path = family_dir / "last.pt"
    final_path = family_dir / "final.pt"
    history_path = family_dir / "history.csv"
    patient_path = family_dir / "all_source_training_sanity_patient_metrics.csv"
    complete_path = family_dir / "COMPLETE.json"

    if complete_path.is_file() and final_path.is_file():
        done = load_json(complete_path)
        valid = (
            done.get("status") == "PASS"
            and done.get("family") == family
            and int(done.get("fixed_epochs", -1)) == int(final_epochs)
            and done.get("final_checkpoint_sha256")
            == sha256_file(final_path)
        )
        if valid:
            print(
                f"{family}: REUSE COMPLETE "
                f"epochs={final_epochs} "
                f"sanity_dice={done['all_source_training_sanity_mean_patient_dice']:.6f}"
            )
            return done

    indices = np.arange(len(slice_df), dtype=np.int64)

    train_ds = DatasetClass(
        images_path,
        masks_path,
        slice_df,
        indices,
        training=True,
    )

    seed = final_family_seed(helper, family)

    generator = torch.Generator()
    generator.manual_seed(seed)
    helper.set_global_seed(seed)

    model = helper.build_model(family).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=helper.LR,
        weight_decay=helper.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    start_epoch = 1
    history = []

    if last_path.is_file():
        ckpt = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        if ckpt.get("family") != family:
            raise RuntimeError(f"{family}: invalid resume family.")
        if int(ckpt.get("fixed_epochs", -1)) != int(final_epochs):
            raise RuntimeError(f"{family}: resume epoch plan changed.")
        if int(ckpt.get("seed", -1)) != int(seed):
            raise RuntimeError(f"{family}: resume seed changed.")

        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history", []))

        helper.restore_rng_state(ckpt["rng_state"])
        generator.set_state(ckpt["dataloader_generator_state"])

        print(
            f"{family}: RESUME epoch={start_epoch}/{final_epochs}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    for epoch in range(start_epoch, final_epochs + 1):
        model.train()

        running = 0.0
        n_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"{family} FINAL E{epoch:02d}/{final_epochs:02d}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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
                loss = helper.combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = int(x.shape[0])
            running += float(loss.detach().cpu()) * bs
            n_seen += bs

            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}"
            )

        train_loss = running / max(1, n_seen)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

        torch.save(
            {
                "status": "IN_PROGRESS",
                "version": VERSION,
                "family": family,
                "epoch": epoch,
                "fixed_epochs": final_epochs,
                "seed": seed,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "rng_state": helper.capture_rng_state(),
                "dataloader_generator_state": generator.get_state(),
            },
            last_path,
        )

        print(
            f"{family} final epoch {epoch:02d}/{final_epochs:02d}: "
            f"train_loss={train_loss:.6f}"
        )

    if len(history) != final_epochs:
        raise RuntimeError(
            f"{family}: training history length "
            f"{len(history)} != {final_epochs}"
        )

    # Save the fixed-duration model before any descriptive sanity metric.
    total_params, trainable_params = helper.model_parameter_count(model)

    torch.save(
        {
            "status": "PASS",
            "version": VERSION,
            "build": BUILD,
            "family": family,
            "epoch": final_epochs,
            "fixed_epochs": final_epochs,
            "duration_rule": "median of five CM3 SOURCE OOF best epochs",
            "source_patients": N_SOURCE_PATIENTS,
            "source_slices": N_SOURCE_SLICES,
            "seed": seed,
            "model": model.state_dict(),
            "params": total_params,
            "trainable_params": trainable_params,
            "source_only": True,
            "promises12_access": False,
        },
        final_path,
    )

    final_sha = sha256_file(final_path)

    print(
        f"\n===== {family} FIXED MODEL SAVED BEFORE SANITY EVAL ====="
    )
    print("final checkpoint:", final_path)
    print("SHA256:", final_sha)

    sanity_dice, patient_df = source_training_sanity_eval(
        helper,
        torch,
        DataLoader,
        model,
        DatasetClass,
        images_path,
        masks_path,
        slice_df,
        device,
    )
    patient_df.to_csv(patient_path, index=False)

    done = {
        "status": "PASS",
        "version": VERSION,
        "build": BUILD,
        "family": family,
        "seed": seed,
        "best_epochs_from_cm3": EXPECTED_BEST_EPOCHS[family],
        "duration_rule": "integer median of five CM3 SOURCE OOF best epochs",
        "fixed_epochs": final_epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": float(helper.LR),
        "weight_decay": float(helper.WEIGHT_DECAY),
        "early_stopping": False,
        "validation_selection": False,
        "source_patients": N_SOURCE_PATIENTS,
        "source_slices": N_SOURCE_SLICES,
        "params": int(total_params),
        "trainable_params": int(trainable_params),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_sha,
        "history": str(history_path),
        "history_sha256": sha256_file(history_path),
        "all_source_training_sanity_mean_patient_dice": sanity_dice,
        "all_source_training_sanity_patient_metrics": str(patient_path),
        "all_source_training_sanity_patient_metrics_sha256": (
            sha256_file(patient_path)
        ),
        "sanity_metric_selection_role": (
            "DESCRIPTIVE_ONLY_COMPUTED_AFTER_FIXED_CHECKPOINT_SAVE"
        ),
        "promises12_access": False,
    }
    save_json(complete_path, done)

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM4D FINAL ALL-SOURCE SEGMENTATION MODEL TRAINING ====="
    )
    print("SOURCE_ONLY=YES")
    print("PROMISE12_ACCESS=NO")
    print("TARGET_GT=NO")
    print("TARGET_TUNING=NO")
    print("MODEL_PANEL=", FAMILIES)
    print("BATCH_SIZE=", BATCH_SIZE)
    print("FINAL_EPOCH_RULE=median(CM3 five OOF best epochs)")
    print("EARLY_STOPPING=NO")
    print("VALIDATION_SELECTION=NO")

    cm3, cm4c = verify_lineage()
    helper = import_cm3_helper()

    final_epochs = freeze_final_epoch_rule(cm3)

    (
        images_path,
        masks_path,
        slices_path,
        slice_df,
        cache_meta_path,
    ) = load_source_cache(cm3)

    (
        nib,
        torch,
        nn,
        F,
        DataLoader,
        Dataset,
        InterpolationMode,
        TF,
        _tqdm,
    ) = helper.import_runtime()

    if int(helper.BATCH_SIZE) != BATCH_SIZE:
        raise RuntimeError("CM3 helper batch size changed.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===== RUNTIME =====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    DatasetClass = helper.build_dataset_class(
        Dataset,
        TF,
        InterpolationMode,
    )

    if args.output_dir.exists():
        stage_dir = args.output_dir
        print("STAGE_DIR=RESUME", stage_dir)
    else:
        stage_dir = args.output_dir
        stage_dir.mkdir(parents=True, exist_ok=False)
        print("STAGE_DIR=NEW", stage_dir)

    results = {}

    for family in FAMILIES:
        print(
            f"\n\n########## FINAL ALL-SOURCE {family} "
            f"({final_epochs[family]} epochs) ##########"
        )

        results[family] = train_final_family(
            helper,
            torch,
            DataLoader,
            DatasetClass,
            family,
            final_epochs[family],
            images_path,
            masks_path,
            slice_df,
            device,
            stage_dir,
        )

    summary_rows = []
    for family in FAMILIES:
        d = results[family]
        summary_rows.append(
            {
                "family": family,
                "fixed_epochs": d["fixed_epochs"],
                "seed": d["seed"],
                "params": d["params"],
                "all_source_training_sanity_mean_patient_dice": (
                    d["all_source_training_sanity_mean_patient_dice"]
                ),
                "checkpoint_sha256": d["final_checkpoint_sha256"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = stage_dir / "CM4D_FINAL_MODEL_SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)

    lock = {
        "status": "PASS",
        "decision": PASS_DECISION,
        "version": VERSION,
        "build": BUILD,
        "cm3_lock_sha256": EXPECTED_CM3_LOCK_SHA,
        "cm4c_lock_sha256": EXPECTED_CM4C_LOCK_SHA,
        "cm3_helper_sha256": EXPECTED_CM3_HELPER_SHA,
        "source_patients": N_SOURCE_PATIENTS,
        "source_slices": N_SOURCE_SLICES,
        "batch_size": BATCH_SIZE,
        "model_families": FAMILIES,
        "duration_selection": {
            "rule": "integer median of five CM3 SOURCE OOF best epochs",
            "best_epochs": EXPECTED_BEST_EPOCHS,
            "final_epochs": final_epochs,
            "chosen_before_promises12_image_access": True,
        },
        "final_models": {
            family: {
                "checkpoint": results[family]["final_checkpoint"],
                "checkpoint_sha256": results[family][
                    "final_checkpoint_sha256"
                ],
                "fixed_epochs": results[family]["fixed_epochs"],
                "seed": results[family]["seed"],
                "params": results[family]["params"],
                "sanity_dice": results[family][
                    "all_source_training_sanity_mean_patient_dice"
                ],
                "sanity_metric_selection_role": (
                    "DESCRIPTIVE_ONLY_COMPUTED_AFTER_FIXED_CHECKPOINT_SAVE"
                ),
            }
            for family in FAMILIES
        },
        "information_boundary": {
            "promises12_access": False,
            "target_gt": False,
            "target_tuning": False,
            "target_model_selection": False,
        },
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "source_cache_meta": str(cache_meta_path),
        "source_cache_meta_sha256": sha256_file(cache_meta_path),
        "next_stage": (
            "CM5_PROMISE12_GT_FREE_SOURCE_AND_TENT1_PREDICTION_"
            "AND_SAFETY_SCORING"
        ),
    }

    lock_path = (
        stage_dir
        / "CM4D_FINAL_ALL_SOURCE_SEGMENTATION_MODELS_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM4D FINAL =====")
    print(summary_df.to_string(index=False))
    print("PROMISE12 access: NO")
    print("Target tuning: NO")
    print("Decision=", PASS_DECISION)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
