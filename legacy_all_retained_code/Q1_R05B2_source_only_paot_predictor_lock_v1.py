#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Q1-R05B2 — Freeze source-only PAOT HARM/BENEFIT predictors.

No target data.
No target images.
No target masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm


VERSION = "2026-08-19-Q1-R05B2-v1"
BUILD = "Q1_R05B2_SOURCE_ONLY_PAOT_PREDICTOR_LOCK"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = (
    ROOT / "docs"
    / "Q1_R05B2_source_only_paot_predictor_lock_preregistered_protocol_v1.md"
)
EXPECTED_PROTOCOL_SHA256 = "85ce6e60fa8002895709e3eccfbaeea24930af00bed46f3cfc0bebd010e89993"

R03_DIR = (
    ROOT / "outputs"
    / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2"
)
R03_LOCK = R03_DIR / "Q1_R03_NINE_STATE_SOURCE_UTILITY_LOCK.json"
EXPECTED_R03_LOCK_SHA256 = (
    "2c773f5eb70bedb50e2c231ef2512c06dff5ecca425eb43b2719637c1bc06dad"
)
R03_TABLE = R03_DIR / "nine_state_source_utility_table.csv"
EXPECTED_R03_TABLE_SHA256 = (
    "969ef666c4e54d0b6f3152709b99a601ff3b82ce6b9f26d46efeb23e53960642"
)

R04_LOCK = (
    ROOT / "outputs"
    / "Q1_R04_model_relative_prospective_utility_feasibility_loao_v1_fix1"
    / "Q1_R04_MODEL_RELATIVE_UTILITY_LOCK.json"
)
EXPECTED_R04_LOCK_SHA256 = (
    "9147a3b3f9f3d7526c057c4c049ac3119ad0f82dfc7b529b32883164ee03cc3a"
)
EXPECTED_R04_DECISION = (
    "STOP_MODEL_RELATIVE_UTILITY_NO_ROBUST_LOAO_SIGNAL"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "Q1_R05B2_source_only_paot_predictor_lock_v1"
)

FAMILIES = (
    "PraNet",
    "DeepLabV3-R50",
    "SegFormer-B0",
)

SOURCE_FEATURE_NAMES = [
    "source_entropy_mean",
    "source_entropy_std",
    "source_entropy_q10",
    "source_entropy_q50",
    "source_entropy_q90",
    "source_prob_mean",
    "source_prob_std",
    "source_prob_q10",
    "source_prob_q50",
    "source_prob_q90",
    "source_confidence_mean",
    "source_confidence_std",
    "source_fg_fraction",
    "source_boundary_density",
    "source_uncertain_fraction_040_060",
    "source_high_entropy_fraction_050",
    "source_logit_abs_mean",
    "source_logit_abs_std",
    "tent_preupdate_entropy_loss",
]

REPRESENTATIONS = ("MRZ19", "RAW19")
TASKS = ("HARM", "BENEFIT")

REFERENCE_STD_FLOOR = 1e-6
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
RANDOM_STATE = 20260819

EXPECTED_ROWS = 13050
EXPECTED_STATES = 9
EXPECTED_ROWS_PER_STATE = 1450
EXPECTED_IDENTITY_PER_STATE = 145
EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY = 8700
EXPECTED_PREDICTORS = 12

DECISION_READY = "SOURCE_ONLY_PAOT_PREDICTORS_LOCKED"
DECISION_FAIL = "SOURCE_ONLY_PAOT_PREDICTOR_LOCK_FAILED"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
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
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch: expected={expected} actual={actual}"
        )
    return actual


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_upstream():
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "R05B2 protocol")
    validate_sha(R03_LOCK, EXPECTED_R03_LOCK_SHA256, "R03 lock")
    validate_sha(R03_TABLE, EXPECTED_R03_TABLE_SHA256, "R03 table")
    validate_sha(R04_LOCK, EXPECTED_R04_LOCK_SHA256, "R04 lock")

    r03 = json.loads(R03_LOCK.read_text(encoding="utf-8"))
    r04 = json.loads(R04_LOCK.read_text(encoding="utf-8"))

    if r03.get("decision") != "NINE_STATE_SOURCE_UTILITY_ASSET_READY":
        raise RuntimeError("Unexpected R03 decision.")
    if bool(r03.get("target_data_used", True)):
        raise RuntimeError("R03 target boundary violated.")

    if r04.get("decision") != EXPECTED_R04_DECISION:
        raise RuntimeError(
            f"Unexpected R04 decision: {r04.get('decision')}"
        )
    if bool(r04.get("target_data_used", True)):
        raise RuntimeError("R04 target boundary violated.")

    return r03, r04


def load_rows():
    rows, fields = read_csv(R03_TABLE)

    required = {
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "perturbation",
        "harmful",
        "beneficial",
        *SOURCE_FEATURE_NAMES,
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"R03 table missing fields: {missing}")

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"R03 rows={len(rows)} expected={EXPECTED_ROWS}"
        )

    states = Counter(r["model_state_id"] for r in rows)
    if len(states) != EXPECTED_STATES:
        raise RuntimeError(f"states={len(states)} expected={EXPECTED_STATES}")
    if any(v != EXPECTED_ROWS_PER_STATE for v in states.values()):
        raise RuntimeError(f"state cardinality mismatch: {dict(states)}")

    families = Counter(r["model_family"] for r in rows)
    if set(families) != set(FAMILIES):
        raise RuntimeError(f"family set mismatch: {dict(families)}")

    return rows


def rows_to_arrays(rows):
    X = np.asarray(
        [
            [float(r[name]) for name in SOURCE_FEATURE_NAMES]
            for r in rows
        ],
        dtype=np.float64,
    )
    family = np.asarray(
        [r["model_family"] for r in rows],
        dtype=object,
    )
    state = np.asarray(
        [r["model_state_id"] for r in rows],
        dtype=object,
    )
    perturb = np.asarray(
        [r["perturbation"] for r in rows],
        dtype=object,
    )
    harm = np.asarray(
        [int(r["harmful"]) for r in rows],
        dtype=np.int64,
    )
    benefit = np.asarray(
        [int(r["beneficial"]) for r in rows],
        dtype=np.int64,
    )

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite R03 feature matrix.")

    return X, family, state, perturb, harm, benefit


def build_reference_statistics(rows, X, state, perturb):
    refs = {}
    csv_rows = []

    state_meta = {}
    for r in rows:
        state_meta.setdefault(
            r["model_state_id"],
            {
                "model_family": r["model_family"],
                "training_seed": int(r["training_seed"]),
                "checkpoint_sha256": r["checkpoint_sha256"],
            },
        )

    for sid in sorted(set(state.tolist())):
        mask = (state == sid) & (perturb == "identity")
        ref = X[mask]

        if len(ref) != EXPECTED_IDENTITY_PER_STATE:
            raise RuntimeError(
                f"identity reference count state={sid}: "
                f"{len(ref)} != {EXPECTED_IDENTITY_PER_STATE}"
            )

        mu = ref.mean(axis=0)
        std = ref.std(axis=0, ddof=0)
        eff = np.maximum(std, REFERENCE_STD_FLOOR)

        refs[sid] = {
            "mean": mu,
            "std": std,
            "effective_std": eff,
        }

        meta = state_meta[sid]
        for j, feature in enumerate(SOURCE_FEATURE_NAMES):
            csv_rows.append({
                "model_family": meta["model_family"],
                "model_state_id": sid,
                "training_seed": meta["training_seed"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "reference_count": len(ref),
                "feature": feature,
                "mean": float(mu[j]),
                "std": float(std[j]),
                "effective_std": float(eff[j]),
                "std_floored": int(std[j] < REFERENCE_STD_FLOOR),
            })

    if len(refs) != EXPECTED_STATES:
        raise RuntimeError("Reference state count mismatch.")

    return refs, csv_rows


def transform_mrz(X, state, refs):
    Z = np.empty_like(X)

    for sid in sorted(set(state.tolist())):
        mask = state == sid
        if sid not in refs:
            raise RuntimeError(f"Missing reference state: {sid}")
        ref = refs[sid]
        Z[mask] = (
            X[mask] - ref["mean"]
        ) / ref["effective_std"]

    if not np.isfinite(Z).all():
        raise RuntimeError("Non-finite MRZ19 matrix.")

    return Z


def sigmoid(v):
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    pos = v >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
    ev = np.exp(v[~pos])
    out[~pos] = ev / (1.0 + ev)
    return out


def fit_binary_package(X, y, held_family, representation, task, train_families):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"Degenerate labels: held={held_family} rep={representation} task={task}"
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(Xs, y)

    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

    if len(coef) != len(SOURCE_FEATURE_NAMES):
        raise RuntimeError("Coefficient length mismatch.")

    # Verify saved explicit deployment math is equivalent to sklearn.
    logits = Xs @ coef + intercept
    p_explicit = sigmoid(logits)
    p_sklearn = model.predict_proba(Xs)[:, 1]
    max_prob_diff = float(
        np.max(np.abs(p_explicit - p_sklearn))
    )
    if max_prob_diff > 1e-12:
        raise RuntimeError(
            f"Explicit deployment probability mismatch: {max_prob_diff}"
        )

    package = {
        "held_out_future_architecture": held_family,
        "training_architecture_families": list(train_families),
        "representation": representation,
        "task": task,
        "positive_class": (
            "delta_dice <= -0.02"
            if task == "HARM"
            else "delta_dice >= +0.02"
        ),
        "feature_names": list(SOURCE_FEATURE_NAMES),
        "training_rows": int(len(y)),
        "standard_scaler_mean": [
            float(v) for v in scaler.mean_
        ],
        "standard_scaler_scale": [
            float(v) for v in scaler.scale_
        ],
        "logistic_coefficient": [
            float(v) for v in coef
        ],
        "logistic_intercept": intercept,
        "C": LOGISTIC_C,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": LOGISTIC_MAX_ITER,
        "random_state": RANDOM_STATE,
        "deployment_probability_equivalence_max_abs_error":
            max_prob_diff,
    }

    diag = {
        "held_out_future_architecture": held_family,
        "training_architecture_families": ";".join(train_families),
        "representation": representation,
        "task": task,
        "training_rows": len(y),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "positive_prevalence": float(y.mean()),
        "in_sample_auroc": float(
            roc_auc_score(y, p_explicit)
        ),
        "in_sample_auprc": float(
            average_precision_score(y, p_explicit)
        ),
        "deployment_probability_equivalence_max_abs_error":
            max_prob_diff,
    }

    return package, diag


def preflight():
    validate_upstream()
    rows = load_rows()
    X, family, state, perturb, harm, benefit = rows_to_arrays(rows)
    refs, ref_rows = build_reference_statistics(
        rows, X, state, perturb
    )
    Z = transform_mrz(X, state, refs)

    print("===== Q1-R05B2 PREFLIGHT =====")
    print(f"protocol_sha256={EXPECTED_PROTOCOL_SHA256}")
    print(f"r03_lock_sha256={EXPECTED_R03_LOCK_SHA256}")
    print(f"r03_table_sha256={EXPECTED_R03_TABLE_SHA256}")
    print(f"r04_lock_sha256={EXPECTED_R04_LOCK_SHA256}")
    print(f"rows={len(rows)}")
    print(f"states={len(set(state.tolist()))}")
    print(f"families={len(set(family.tolist()))}")
    print(f"features={X.shape[1]}")
    print(f"reference_states={len(refs)}")
    print(f"reference_stat_rows={len(ref_rows)}")
    print(f"representations={REPRESENTATIONS}")
    print(f"tasks={TASKS}")
    print(f"expected_predictors={EXPECTED_PREDICTORS}")
    print("target_data=NO")
    print("target_pixels_opened=NO")
    print("target_masks_opened=NO")
    print("predictor_fit=NO")
    print("PREFLIGHT_PASS")


def run(args):
    r03_lock, r04_lock = validate_upstream()
    rows = load_rows()
    X_raw, family, state, perturb, harm, benefit = rows_to_arrays(rows)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        raise FileExistsError(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)

    protocol_copy = build_dir / "preregistered_protocol_copy.md"
    shutil.copy2(PROTOCOL, protocol_copy)
    validate_sha(
        protocol_copy,
        EXPECTED_PROTOCOL_SHA256,
        "R05B2 protocol copy",
    )

    refs, ref_rows = build_reference_statistics(
        rows,
        X_raw,
        state,
        perturb,
    )
    X_mrz = transform_mrz(
        X_raw,
        state,
        refs,
    )

    ref_path = build_dir / "model_reference_statistics.csv"
    write_csv(
        ref_path,
        ref_rows,
        [
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
            "reference_count",
            "feature",
            "mean",
            "std",
            "effective_std",
            "std_floored",
        ],
    )

    packages = []
    diagnostics = []
    manifest_rows = []

    pbar = tqdm(
        total=EXPECTED_PREDICTORS,
        desc="Q1-R05B2 freeze PAOT predictors",
        unit="model",
        dynamic_ncols=True,
    )

    for held_family in FAMILIES:
        train_families = tuple(
            f for f in FAMILIES
            if f != held_family
        )
        train_mask = family != held_family

        if int(train_mask.sum()) != EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY:
            raise RuntimeError(
                f"held={held_family}: train rows={train_mask.sum()} "
                f"expected={EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY}"
            )

        for representation in REPRESENTATIONS:
            X_rep = X_mrz if representation == "MRZ19" else X_raw
            X_train = X_rep[train_mask]

            for task in TASKS:
                y = (
                    harm[train_mask]
                    if task == "HARM"
                    else benefit[train_mask]
                )

                package, diag = fit_binary_package(
                    X_train,
                    y,
                    held_family,
                    representation,
                    task,
                    train_families,
                )
                package_id = (
                    f"{held_family}__{representation}__{task}"
                )
                package["package_id"] = package_id
                packages.append(package)
                diagnostics.append(diag)

                manifest_rows.append({
                    "package_id": package_id,
                    "held_out_future_architecture": held_family,
                    "training_architecture_families":
                        ";".join(train_families),
                    "representation": representation,
                    "task": task,
                    "features": len(SOURCE_FEATURE_NAMES),
                    "training_rows": len(y),
                    "positive_count": int(y.sum()),
                    "negative_count": int((1 - y).sum()),
                })

                pbar.update(1)
                pbar.set_postfix(
                    held=held_family,
                    rep=representation,
                    task=task,
                )

    pbar.close()

    if len(packages) != EXPECTED_PREDICTORS:
        raise RuntimeError(
            f"predictors={len(packages)} expected={EXPECTED_PREDICTORS}"
        )

    package_path = build_dir / "predictor_packages.json"
    write_json(
        package_path,
        {
            "script_version": VERSION,
            "build": BUILD,
            "feature_names": SOURCE_FEATURE_NAMES,
            "representations": list(REPRESENTATIONS),
            "tasks": list(TASKS),
            "packages": packages,
        },
    )

    manifest_path = build_dir / "predictor_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "package_id",
            "held_out_future_architecture",
            "training_architecture_families",
            "representation",
            "task",
            "features",
            "training_rows",
            "positive_count",
            "negative_count",
        ],
    )

    diag_path = build_dir / "source_fit_diagnostics.csv"
    write_csv(
        diag_path,
        diagnostics,
        [
            "held_out_future_architecture",
            "training_architecture_families",
            "representation",
            "task",
            "training_rows",
            "positive_count",
            "negative_count",
            "positive_prevalence",
            "in_sample_auroc",
            "in_sample_auprc",
            "deployment_probability_equivalence_max_abs_error",
        ],
    )

    upstream = {
        "script_version": VERSION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "r03_decision": r03_lock.get("decision"),
        "r04_decision": r04_lock.get("decision"),
        "continuous_delta_regression_retired": True,
        "paot_classification_only": True,
        "target_data_used": False,
        "target_image_pixels_opened": False,
        "target_mask_pixels_opened": False,
        "target_probabilities_generated": False,
        "target_tta_run": False,
        "hyperparameter_sweep": False,
        "feature_selection": False,
    }
    upstream_path = build_dir / "upstream_audit.json"
    write_json(upstream_path, upstream)

    checks = {
        "upstream_hashes_verified": True,
        "r03_rows_13050": len(rows) == EXPECTED_ROWS,
        "reference_states_9": len(refs) == EXPECTED_STATES,
        "reference_count_each_145": all(
            int(r["reference_count"]) == EXPECTED_IDENTITY_PER_STATE
            for r in ref_rows
        ),
        "predictor_packages_12": len(packages) == EXPECTED_PREDICTORS,
        "all_predictors_19_features": all(
            len(p["feature_names"]) == len(SOURCE_FEATURE_NAMES)
            for p in packages
        ),
        "all_predictors_8700_training_rows": all(
            int(p["training_rows"]) == EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY
            for p in packages
        ),
        "target_data_not_used": True,
    }

    decision = (
        DECISION_READY
        if all(checks.values())
        else DECISION_FAIL
    )

    decision_path = build_dir / "decision.txt"
    decision_path.write_text(decision + "\n", encoding="utf-8")

    lines = [
        "===== Q1-R05B2 SOURCE-ONLY PAOT PREDICTOR LOCK =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "Frozen object:",
        "  continuous Delta-Dice regression=RETIRED",
        "  PAOT classification=HARM + BENEFIT",
        "",
        "Source asset:",
        f"  rows={len(rows)}",
        f"  model states={len(refs)}",
        f"  features={len(SOURCE_FEATURE_NAMES)}",
        "",
        "Final predictor construction:",
        "  future target architectures=3",
        "  representations=MRZ19, RAW19",
        "  tasks=HARM, BENEFIT",
        f"  predictor packages={len(packages)}",
        f"  training rows per package={EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY}",
        "",
        "Information boundary:",
        "  target data=NO",
        "  target image pixels opened=NO",
        "  target mask pixels opened=NO",
        "  target probabilities generated=NO",
        "  target TTA=NO",
        "  hyperparameter sweep=NO",
        "",
        "Checks:",
    ]

    for key, passed in checks.items():
        lines.append(f"  {key}={'PASS' if passed else 'FAIL'}")

    lines += [
        "",
        "Decision:",
        f"  {decision}",
    ]

    if decision == DECISION_READY:
        lines += [
            "",
            "Predictors are now frozen before new confirmatory target access.",
        ]

    run_log_path = build_dir / "run_log.txt"
    run_log_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "protocol_copy": protocol_copy,
        "upstream_audit": upstream_path,
        "model_reference_statistics": ref_path,
        "predictor_manifest": manifest_path,
        "predictor_packages": package_path,
        "source_fit_diagnostics": diag_path,
        "decision": decision_path,
        "run_log": run_log_path,
    }

    lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "r03_lock_sha256": EXPECTED_R03_LOCK_SHA256,
        "r03_table_sha256": EXPECTED_R03_TABLE_SHA256,
        "r04_lock_sha256": EXPECTED_R04_LOCK_SHA256,
        "continuous_delta_regression_retired": True,
        "primary_representation": "MRZ19",
        "comparator_representation": "RAW19",
        "tasks": list(TASKS),
        "feature_names": list(SOURCE_FEATURE_NAMES),
        "model_reference_states": EXPECTED_STATES,
        "predictor_packages": EXPECTED_PREDICTORS,
        "target_data_used": False,
        "target_image_pixels_opened": False,
        "target_mask_pixels_opened": False,
        "target_probabilities_generated": False,
        "target_tta_run": False,
        "decision": decision,
        "artifacts": {
            name: {
                "relative_path": str(path.relative_to(build_dir)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    }

    lock_path = build_dir / "Q1_R05B2_SOURCE_PAOT_PREDICTOR_LOCK.json"
    write_json(lock_path, lock)
    lock_sha = sha256_file(lock_path)

    for name, meta in lock["artifacts"].items():
        path = build_dir / meta["relative_path"]
        if sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"Artifact changed before commit: {name}")

    build_dir.rename(args.output_dir)

    print()
    print(
        (args.output_dir / "run_log.txt").read_text(
            encoding="utf-8"
        )
    )
    print(
        "Q1-R05B2 LOCK:",
        args.output_dir / "Q1_R05B2_SOURCE_PAOT_PREDICTOR_LOCK.json",
    )
    print("Q1-R05B2 LOCK SHA256:", lock_sha)


def self_test():
    assert len(SOURCE_FEATURE_NAMES) == 19
    assert len(FAMILIES) == 3
    assert len(REPRESENTATIONS) == 2
    assert len(TASKS) == 2
    assert EXPECTED_PREDICTORS == 12
    assert EXPECTED_TRAIN_ROWS_PER_HELD_FAMILY == 8700
    assert EXPECTED_IDENTITY_PER_STATE == 145

    x = np.asarray([-1000.0, 0.0, 1000.0])
    p = sigmoid(x)
    assert p[0] < 1e-100
    assert abs(p[1] - 0.5) < 1e-12
    assert p[2] > 1.0 - 1e-12

    # Synthetic explicit deployment equivalence.
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(20260819)
    X = rng.normal(size=(200, 19))
    y = np.asarray([0, 1] * 100, dtype=np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=20260819,
    )
    model.fit(Xs, y)

    coef = model.coef_.reshape(-1)
    intercept = float(model.intercept_[0])
    p1 = model.predict_proba(Xs)[:, 1]
    p2 = sigmoid(Xs @ coef + intercept)
    assert np.max(np.abs(p1 - p2)) <= 1e-12

    print("FROZEN_CARDINALITY_TEST_PASS")
    print("SIGMOID_STABILITY_TEST_PASS")
    print("EXPLICIT_DEPLOYMENT_EQUIVALENCE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Q1-R05B2: freeze architecture-held-out source-only PAOT "
            "HARM/BENEFIT predictors before external target access."
        )
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.preflight_only:
        preflight()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
