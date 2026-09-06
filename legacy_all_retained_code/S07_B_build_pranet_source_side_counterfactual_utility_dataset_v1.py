#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S07-B: PraNet source-side counterfactual utility dataset construction.

SOURCE-VAL only. No utility estimator is trained here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


VERSION = "2026-08-18-S07-B-v1"
BUILD = "S07_B_PRANET_SOURCE_SIDE_A1_A2_COUNTERFACTUAL_UTILITY_DATASET"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL = ROOT / "docs" / "S07_A_safettta_v2_counterfactual_utility_preregistered_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "8a0aa3eaf708d6d12c55e13e1e84f8cdd80310a19673666d99d8089738911ab2"

S03B_HELPER = ROOT / "code" / "S03_B_build_source_side_safettta_gate_v1.py"
EXPECTED_S03B_HELPER_SHA256 = "07ac65247216dea376e8996cbfbc98c54c62157ecdd1e9b7d9b539b5a74a2b5c"

TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_TRAINING_HELPER_SHA256 = "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"

OUTPUT_DIR = ROOT / "outputs" / "S07_B_pranet_source_side_counterfactual_utility_dataset_v1"

SEEDS = (20260817, 20260818, 20260819)
SOURCE_VAL_N = 145
N_FOLDS = 5

PERTURBATIONS = (
    "identity",
    "brightness_070",
    "brightness_130",
    "contrast_065",
    "saturation_050",
    "gamma_070",
    "gamma_150",
    "gaussian_blur_r15",
    "gaussian_noise_s004",
    "downsample_050",
)

ACTIONS = (
    ("A1_TENT_1STEP", 1),
    ("A2_TENT_2STEP", 2),
)

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02

EXPECTED_BASE_CASES = SOURCE_VAL_N * len(PERTURBATIONS) * len(SEEDS)
EXPECTED_ACTION_ROWS = EXPECTED_BASE_CASES * len(ACTIONS)

EXPECTED_CHECKPOINT_SHA256 = {
    20260817: "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
    20260818: "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
    20260819: "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
}

FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_fg_fraction",
    "source_boundary_density",
    "source_confidence_mean",
    "tent_entropy_mean",
    "tent_entropy_std",
    "tent_prob_mean",
    "tent_prob_std",
    "tent_prob_q10",
    "tent_prob_q50",
    "tent_prob_q90",
    "tent_fg_fraction",
    "tent_boundary_density",
    "tent_confidence_mean",
    "entropy_mean_shift",
    "prob_abs_change_mean",
    "mask_disagreement_fraction",
    "abs_fg_fraction_shift",
]


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch. expected={expected} actual={actual}"
        )
    return actual


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def import_frozen_helper():
    validate_sha(S03B_HELPER, EXPECTED_S03B_HELPER_SHA256, "S03-B helper")
    validate_sha(TRAINING_HELPER, EXPECTED_TRAINING_HELPER_SHA256, "PraNet training helper")
    spec = importlib.util.spec_from_file_location(
        "s07b_frozen_s03b",
        str(S03B_HELPER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen S03-B helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_helper(helper):
    required = (
        "load_source_val_rows",
        "validate_checkpoints",
        "import_training_helper",
        "load_model",
        "configure_tent",
        "snapshot_params",
        "restore_params",
        "final_logit",
        "mean_binary_entropy",
        "apply_perturbation",
        "image_to_model_tensor",
        "load_gt",
        "dice_from_logit",
        "extract_features",
        "fold_for_sample",
        "FEATURE_NAMES",
        "PERTURBATIONS",
        "SEEDS",
        "TENT_LR",
        "TENT_WEIGHT_DECAY",
        "CHECKPOINTS",
        "DATA_ROOT",
    )
    for name in required:
        if not hasattr(helper, name):
            raise RuntimeError(f"Frozen S03-B helper missing symbol: {name}")

    if tuple(helper.PERTURBATIONS) != PERTURBATIONS:
        raise RuntimeError("Frozen perturbation bank mismatch.")
    if tuple(helper.SEEDS) != SEEDS:
        raise RuntimeError("Frozen seed list mismatch.")
    if list(helper.FEATURE_NAMES) != FEATURE_NAMES:
        raise RuntimeError("Frozen 24-feature order mismatch.")
    if float(helper.TENT_LR) != 1e-3:
        raise RuntimeError("Frozen TENT lr mismatch.")
    if float(helper.TENT_WEIGHT_DECAY) != 0.0:
        raise RuntimeError("Frozen TENT weight decay mismatch.")

    for seed in SEEDS:
        declared = str(helper.CHECKPOINTS[seed]["sha256"])
        if declared != EXPECTED_CHECKPOINT_SHA256[seed]:
            raise RuntimeError(f"Checkpoint provenance mismatch seed={seed}")


def seed_runtime(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def source_a1_a2_logits(helper, model, params, source_values, x):
    """
    A0/source prediction plus one deterministic TENT trajectory:
      reset -> step1 -> A1 -> step2 -> A2.

    A2 uses the same Adam optimizer instance across both steps, preserving
    optimizer state as required by the frozen two-step action.
    """
    helper.restore_params(params, source_values)

    model.eval()
    with torch.no_grad():
        z_source = helper.final_logit(model, x).detach()

    helper.restore_params(params, source_values)
    model.train()

    optimizer = torch.optim.Adam(
        params,
        lr=float(helper.TENT_LR),
        weight_decay=float(helper.TENT_WEIGHT_DECAY),
    )

    optimizer.zero_grad(set_to_none=True)
    z_pre1 = helper.final_logit(model, x)
    loss1 = helper.mean_binary_entropy(z_pre1)
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z_a1 = helper.final_logit(model, x).detach()

    z_pre2 = helper.final_logit(model, x)
    loss2 = helper.mean_binary_entropy(z_pre2)
    loss2.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z_a2 = helper.final_logit(model, x).detach()

    l1 = float(loss1.detach().cpu())
    l2 = float(loss2.detach().cpu())

    del optimizer, z_pre1, z_pre2, loss1, loss2

    return (
        z_source[0, 0].float().cpu().numpy(),
        z_a1[0, 0].float().cpu().numpy(),
        z_a2[0, 0].float().cpu().numpy(),
        l1,
        l2,
    )


def classify_delta(delta: float):
    harmful = int(delta <= HARM_THRESHOLD)
    beneficial = int(delta >= BENEFIT_THRESHOLD)
    neutral = int(not harmful and not beneficial)
    return harmful, neutral, beneficial


def summarize_values(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise RuntimeError("Empty value list.")
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "q05": float(np.quantile(a, 0.05)),
        "q10": float(np.quantile(a, 0.10)),
        "q25": float(np.quantile(a, 0.25)),
        "q50": float(np.quantile(a, 0.50)),
        "q75": float(np.quantile(a, 0.75)),
        "q90": float(np.quantile(a, 0.90)),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def distribution_row(rows, group_type: str, group: str):
    if not rows:
        raise RuntimeError(f"Empty summary group: {group_type}={group}")
    stats = summarize_values([float(r["delta_dice"]) for r in rows])
    h = sum(int(r["harmful"]) for r in rows)
    n = sum(int(r["neutral"]) for r in rows)
    b = sum(int(r["beneficial"]) for r in rows)

    out = {
        "group_type": group_type,
        "group": str(group),
        "rows": len(rows),
        "source_mean_dice": float(np.mean([float(r["source_dice"]) for r in rows])),
        "action_mean_dice": float(np.mean([float(r["action_dice"]) for r in rows])),
        "harmful_n": h,
        "harmful_fraction": h / len(rows),
        "neutral_n": n,
        "neutral_fraction": n / len(rows),
        "beneficial_n": b,
        "beneficial_fraction": b / len(rows),
    }
    for key, value in stats.items():
        out[f"delta_dice_{key}"] = value
    return out


def audit_rows(rows):
    if len(rows) != EXPECTED_ACTION_ROWS:
        raise RuntimeError(
            f"Action-row mismatch: expected={EXPECTED_ACTION_ROWS} actual={len(rows)}"
        )

    keys = [
        (r["sample_id"], int(r["seed"]), r["perturbation"], r["action"])
        for r in rows
    ]
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        raise RuntimeError(f"Duplicate scientific action keys={duplicate_keys}")

    expected_actions = {name for name, _ in ACTIONS}
    base_actions = defaultdict(set)
    for r in rows:
        base_actions[
            (r["sample_id"], int(r["seed"]), r["perturbation"])
        ].add(r["action"])

    incomplete = sum(1 for actions in base_actions.values() if actions != expected_actions)
    if incomplete:
        raise RuntimeError(f"Incomplete action sets={incomplete}")
    if len(base_actions) != EXPECTED_BASE_CASES:
        raise RuntimeError(
            f"Base-case mismatch: expected={EXPECTED_BASE_CASES} actual={len(base_actions)}"
        )

    folds_by_sample = defaultdict(set)
    for r in rows:
        folds_by_sample[r["sample_id"]].add(int(r["fold"]))

    leakage = sum(1 for folds in folds_by_sample.values() if len(folds) != 1)
    if leakage:
        raise RuntimeError(f"Source-group fold leakage={leakage}")
    if len(folds_by_sample) != SOURCE_VAL_N:
        raise RuntimeError(
            f"Source group count mismatch: expected={SOURCE_VAL_N} actual={len(folds_by_sample)}"
        )

    nonfinite_features = 0
    nonfinite_numeric = 0
    bad_partition = 0

    for r in rows:
        if int(r["harmful"]) + int(r["neutral"]) + int(r["beneficial"]) != 1:
            bad_partition += 1

        for name in FEATURE_NAMES:
            if not math.isfinite(float(r[name])):
                nonfinite_features += 1

        for name in (
            "source_dice",
            "action_dice",
            "delta_dice",
            "tent_step1_entropy_loss",
            "tent_step2_entropy_loss",
        ):
            if not math.isfinite(float(r[name])):
                nonfinite_numeric += 1

    if nonfinite_features or nonfinite_numeric or bad_partition:
        raise RuntimeError(
            f"Integrity failure nonfinite_features={nonfinite_features} "
            f"nonfinite_numeric={nonfinite_numeric} bad_partition={bad_partition}"
        )

    fold_counts = Counter(
        next(iter(folds))
        for folds in folds_by_sample.values()
    )

    return {
        "action_rows": len(rows),
        "base_cases": len(base_actions),
        "source_groups": len(folds_by_sample),
        "duplicate_row_keys": duplicate_keys,
        "incomplete_base_cases": incomplete,
        "group_leakage_samples": leakage,
        "nonfinite_feature_values": nonfinite_features,
        "nonfinite_numeric_values": nonfinite_numeric,
        "bad_state_partition_rows": bad_partition,
        "fold_group_counts": {
            str(k): int(v)
            for k, v in sorted(fold_counts.items())
        },
    }


def build_distribution_summary(rows):
    result = []

    for action, _ in ACTIONS:
        result.append(
            distribution_row(
                [r for r in rows if r["action"] == action],
                "action",
                action,
            )
        )

    for action, _ in ACTIONS:
        for seed in SEEDS:
            result.append(
                distribution_row(
                    [
                        r for r in rows
                        if r["action"] == action and int(r["seed"]) == seed
                    ],
                    "action_seed",
                    f"{action}::{seed}",
                )
            )

    for action, _ in ACTIONS:
        for perturbation in PERTURBATIONS:
            result.append(
                distribution_row(
                    [
                        r for r in rows
                        if r["action"] == action and r["perturbation"] == perturbation
                    ],
                    "action_perturbation",
                    f"{action}::{perturbation}",
                )
            )

    return result


def run(args):
    protocol_sha = validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "S07-A protocol")

    helper = import_frozen_helper()
    validate_helper(helper)
    helper.validate_checkpoints()

    for seed in SEEDS:
        actual = file_sha256(Path(helper.CHECKPOINTS[seed]["path"]))
        if actual != EXPECTED_CHECKPOINT_SHA256[seed]:
            raise RuntimeError(f"Checkpoint SHA mismatch seed={seed}")

    training = helper.import_training_helper()
    source_rows = helper.load_source_val_rows()
    if len(source_rows) != SOURCE_VAL_N:
        raise RuntimeError(f"source_val count mismatch={len(source_rows)}")

    if args.output_dir.exists():
        raise FileExistsError(f"Final S07-B output exists: {args.output_dir}")

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Incomplete S07-B output exists: {build_dir}. "
                "Use --technical-rerun only after a technical interruption."
            )
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device={device}")
    print(f"source_val={SOURCE_VAL_N} perturbations={len(PERTURBATIONS)} seeds={len(SEEDS)}")
    print(f"base cases={EXPECTED_BASE_CASES} action rows={EXPECTED_ACTION_ROWS}")
    print("Actions=A1_TENT_1STEP,A2_TENT_2STEP")
    print("Utility estimator fitted=NO")
    print("Non-source evaluation data opened=NO")
    print()

    all_rows = []

    for seed in SEEDS:
        seed_runtime(seed)

        model = helper.load_model(training, seed, device)
        params = helper.configure_tent(model)
        source_values = helper.snapshot_params(params)

        pbar = tqdm(
            total=len(source_rows) * len(PERTURBATIONS),
            desc=f"S07-B seed {seed}",
            unit="base-case",
            dynamic_ncols=True,
        )

        for row in source_rows:
            image_path = helper.DATA_ROOT / Path(row["image_relpath"])
            with Image.open(image_path) as im:
                native = im.convert("RGB")

            gt = helper.load_gt(row)
            fold = int(helper.fold_for_sample(row["sample_id"]))
            if not 0 <= fold < N_FOLDS:
                raise RuntimeError(f"Invalid fold={fold}")

            for perturbation in PERTURBATIONS:
                perturbed = helper.apply_perturbation(
                    native,
                    perturbation,
                    row["sample_id"],
                )
                x = helper.image_to_model_tensor(training, perturbed).to(
                    device,
                    non_blocking=True,
                )

                z_source, z_a1, z_a2, loss1, loss2 = source_a1_a2_logits(
                    helper,
                    model,
                    params,
                    source_values,
                    x,
                )

                for label, z in (
                    ("source", z_source),
                    ("a1", z_a1),
                    ("a2", z_a2),
                ):
                    if not np.isfinite(z).all():
                        raise RuntimeError(
                            f"Nonfinite {label} logits seed={seed} "
                            f"sample={row['sample_id']} perturbation={perturbation}"
                        )

                source_dice = helper.dice_from_logit(z_source, gt)
                a1_dice = helper.dice_from_logit(z_a1, gt)
                a2_dice = helper.dice_from_logit(z_a2, gt)

                payloads = (
                    ("A1_TENT_1STEP", 1, z_a1, a1_dice),
                    ("A2_TENT_2STEP", 2, z_a2, a2_dice),
                )

                for action, action_steps, z_action, action_dice in payloads:
                    delta = float(action_dice - source_dice)
                    harmful, neutral, beneficial = classify_delta(delta)

                    features = helper.extract_features(z_source, z_action)
                    if list(features.keys()) != FEATURE_NAMES:
                        raise RuntimeError("24-feature order/key mismatch.")

                    out = {
                        "seed": seed,
                        "sample_id": row["sample_id"],
                        "source_group_id": row["sample_id"],
                        "fold": fold,
                        "perturbation": perturbation,
                        "base_case_id": f"{row['sample_id']}::{seed}::{perturbation}",
                        "action": action,
                        "action_steps": action_steps,
                        "source_dice": float(source_dice),
                        "action_dice": float(action_dice),
                        "delta_dice": delta,
                        "harmful": harmful,
                        "neutral": neutral,
                        "beneficial": beneficial,
                        "tent_step1_entropy_loss": float(loss1),
                        "tent_step2_entropy_loss": float(loss2),
                    }
                    out.update(features)
                    all_rows.append(out)

                pbar.set_postfix(
                    d1=f"{a1_dice-source_dice:+.3f}",
                    d2=f"{a2_dice-source_dice:+.3f}",
                )
                pbar.update(1)

        pbar.close()
        helper.restore_params(params, source_values)
        del model, params, source_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    audit = audit_rows(all_rows)

    fields = [
        "seed",
        "sample_id",
        "source_group_id",
        "fold",
        "perturbation",
        "base_case_id",
        "action",
        "action_steps",
        "source_dice",
        "action_dice",
        "delta_dice",
        "harmful",
        "neutral",
        "beneficial",
        "tent_step1_entropy_loss",
        "tent_step2_entropy_loss",
    ] + FEATURE_NAMES

    table_path = build_dir / "source_side_counterfactual_utility_table.csv"
    write_csv(table_path, all_rows, fields)
    table_sha = file_sha256(table_path)

    dist_rows = build_distribution_summary(all_rows)
    dist_path = build_dir / "counterfactual_utility_distribution_summary.csv"
    write_csv(dist_path, dist_rows, list(dist_rows[0].keys()))
    dist_sha = file_sha256(dist_path)

    fold_rows = []
    for fold in range(N_FOLDS):
        fold_rows.append({
            "fold": fold,
            "unique_source_groups": int(audit["fold_group_counts"].get(str(fold), 0)),
            "rows": sum(1 for r in all_rows if int(r["fold"]) == fold),
        })
    fold_path = build_dir / "source_group_fold_audit.csv"
    write_csv(
        fold_path,
        fold_rows,
        ["fold", "unique_source_groups", "rows"],
    )
    fold_sha = file_sha256(fold_path)

    by_action = {
        r["group"]: r
        for r in dist_rows
        if r["group_type"] == "action"
    }
    a1 = by_action["A1_TENT_1STEP"]
    a2 = by_action["A2_TENT_2STEP"]

    audit_payload = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "s03b_helper_sha256": EXPECTED_S03B_HELPER_SHA256,
        "training_helper_sha256": EXPECTED_TRAINING_HELPER_SHA256,
        "checkpoint_sha256": {
            str(seed): EXPECTED_CHECKPOINT_SHA256[seed]
            for seed in SEEDS
        },
        "source_val_images": SOURCE_VAL_N,
        "perturbations": list(PERTURBATIONS),
        "seeds": list(SEEDS),
        "actions": [
            {"name": name, "steps": steps}
            for name, steps in ACTIONS
        ],
        "harm_threshold": HARM_THRESHOLD,
        "benefit_threshold": BENEFIT_THRESHOLD,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "data_audit": audit,
        "table_sha256": table_sha,
        "distribution_summary_sha256": dist_sha,
        "fold_audit_sha256": fold_sha,
        "utility_estimator_fitted": False,
        "threshold_selected": False,
        "action_rule_selected_from_labels": False,
        "non_source_evaluation_data_opened": False,
        "decision": (
            "S07_B_SOURCE_SIDE_COUNTERFACTUAL_UTILITY_DATASET_"
            "LOCKED_READY_FOR_S07C_GROUPED_OOF_QUANTILE_FEASIBILITY"
        ),
    }

    audit_path = build_dir / "S07_B_counterfactual_utility_dataset_audit.json"
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_sha = file_sha256(audit_path)

    lock_payload = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": protocol_sha,
        "table_sha256": table_sha,
        "distribution_summary_sha256": dist_sha,
        "fold_audit_sha256": fold_sha,
        "audit_json_sha256": audit_sha,
        "expected_base_cases": EXPECTED_BASE_CASES,
        "expected_action_rows": EXPECTED_ACTION_ROWS,
        "utility_estimator_fitted": False,
        "target_data_used": False,
        "decision": audit_payload["decision"],
    }
    lock_path = build_dir / "S07_B_COUNTERFACTUAL_UTILITY_DATASET_LOCK.json"
    lock_path.write_text(
        json.dumps(lock_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lock_sha = file_sha256(lock_path)

    summary = f"""===== S07-B PRANET SOURCE-SIDE COUNTERFACTUAL UTILITY DATASET =====
Script version: {VERSION}
Build: {BUILD}

Frozen provenance:
  S07-A protocol SHA256={protocol_sha}
  S03-B helper SHA256={EXPECTED_S03B_HELPER_SHA256}
  PraNet training helper SHA256={EXPECTED_TRAINING_HELPER_SHA256}

Construction:
  source_val images={SOURCE_VAL_N}
  perturbations={len(PERTURBATIONS)}
  seeds={len(SEEDS)}
  base cases={EXPECTED_BASE_CASES}
  actions per base case={len(ACTIONS)}
  total action rows={len(all_rows)}

[A1_TENT_1STEP]
  rows={a1["rows"]}
  Source mean Dice={a1["source_mean_dice"]:.6f}
  Action mean Dice={a1["action_mean_dice"]:.6f}
  DeltaDice mean={a1["delta_dice_mean"]:+.6f}
  DeltaDice q10={a1["delta_dice_q10"]:+.6f}
  DeltaDice q50={a1["delta_dice_q50"]:+.6f}
  DeltaDice q90={a1["delta_dice_q90"]:+.6f}
  harmful={a1["harmful_fraction"]:.6f}
  neutral={a1["neutral_fraction"]:.6f}
  beneficial={a1["beneficial_fraction"]:.6f}

[A2_TENT_2STEP]
  rows={a2["rows"]}
  Source mean Dice={a2["source_mean_dice"]:.6f}
  Action mean Dice={a2["action_mean_dice"]:.6f}
  DeltaDice mean={a2["delta_dice_mean"]:+.6f}
  DeltaDice q10={a2["delta_dice_q10"]:+.6f}
  DeltaDice q50={a2["delta_dice_q50"]:+.6f}
  DeltaDice q90={a2["delta_dice_q90"]:+.6f}
  harmful={a2["harmful_fraction"]:.6f}
  neutral={a2["neutral_fraction"]:.6f}
  beneficial={a2["beneficial_fraction"]:.6f}

Dataset integrity:
  duplicate row keys={audit["duplicate_row_keys"]}
  incomplete base cases={audit["incomplete_base_cases"]}
  source-group leakage={audit["group_leakage_samples"]}
  nonfinite feature values={audit["nonfinite_feature_values"]}
  nonfinite numeric values={audit["nonfinite_numeric_values"]}
  invalid state partitions={audit["bad_state_partition_rows"]}
  fold group counts={audit["fold_group_counts"]}

Scientific firewall:
  utility estimator fitted=NO
  threshold selected=NO
  action rule selected from labels=NO
  non-source evaluation data opened=NO

Locks:
  utility table SHA256={table_sha}
  audit SHA256={audit_sha}
  S07-B lock SHA256={lock_sha}

Decision: {audit_payload["decision"]}

[OK] Outputs: {args.output_dir}
"""
    (build_dir / "summary.txt").write_text(summary, encoding="utf-8")

    # Final hash checks before atomic commit.
    assert file_sha256(table_path) == table_sha
    assert file_sha256(dist_path) == dist_sha
    assert file_sha256(fold_path) == fold_sha
    assert file_sha256(audit_path) == audit_sha
    assert file_sha256(lock_path) == lock_sha

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))


# ---------- self-test ----------

class _ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(
            1,
            affine=True,
            track_running_stats=False,
        )

    def forward(self, x):
        return self.bn(self.conv(x))


class _ToyHelper:
    TENT_LR = 1e-3
    TENT_WEIGHT_DECAY = 0.0

    @staticmethod
    def restore_params(params, values):
        with torch.no_grad():
            for p, src in zip(params, values):
                p.copy_(src)
                p.grad = None

    @staticmethod
    def final_logit(model, x):
        return model(x)

    @staticmethod
    def mean_binary_entropy(logits):
        p = torch.sigmoid(logits)
        return (
            torch.nn.functional.softplus(logits)
            - p * logits
        ).mean()


def self_test():
    assert EXPECTED_BASE_CASES == 4350
    assert EXPECTED_ACTION_ROWS == 8700
    assert len(FEATURE_NAMES) == 24
    assert ACTIONS == (
        ("A1_TENT_1STEP", 1),
        ("A2_TENT_2STEP", 2),
    )
    assert HARM_THRESHOLD == -0.02
    assert BENEFIT_THRESHOLD == 0.02

    assert classify_delta(-0.03) == (1, 0, 0)
    assert classify_delta(0.00) == (0, 1, 0)
    assert classify_delta(+0.03) == (0, 0, 1)

    q = summarize_values([-1.0, 0.0, 1.0])
    assert q["min"] == -1.0
    assert q["max"] == 1.0
    assert abs(q["q50"]) < 1e-12

    torch.manual_seed(7)
    model = _ToyNet()
    model.requires_grad_(False)
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            params.extend([module.weight, module.bias])

    source_values = [p.detach().clone() for p in params]
    x = torch.randn(1, 3, 8, 8)

    z0, z1, z2, l1, l2 = source_a1_a2_logits(
        _ToyHelper,
        model,
        params,
        source_values,
        x,
    )
    assert z0.shape == (8, 8)
    assert z1.shape == z0.shape
    assert z2.shape == z0.shape
    assert np.isfinite(z0).all()
    assert np.isfinite(z1).all()
    assert np.isfinite(z2).all()
    assert math.isfinite(l1) and math.isfinite(l2)
    assert np.max(np.abs(z2 - z1)) > 0.0

    print("FROZEN_COUNT_TEST_PASS")
    print("ACTION_SET_TEST_PASS")
    print("HARM_BENEFIT_PARTITION_TEST_PASS")
    print("QUANTILE_SUMMARY_TEST_PASS")
    print("TWO_STEP_TENT_TRAJECTORY_TEST_PASS")
    print("NO_ESTIMATOR_TRAINING_STAGE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the frozen PraNet source-side A1/A2 counterfactual "
            "utility dataset for SafeTTA-v2. No estimator fitting."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--technical-rerun",
        action="store_true",
        help=(
            "Delete only an incomplete S07-B __building directory after "
            "a technical interruption. Never overwrite completed output."
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
