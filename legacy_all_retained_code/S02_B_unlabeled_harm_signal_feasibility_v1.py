#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
S02-B: Target-GT-free signal feasibility for predicting TENT harm.

Scientific order:
A) derive a fixed no-label feature table from already locked Source-Only/TENT
   logits; SHA256-lock the table.
B) only then read the already frozen D1 DeltaDice table, create harm labels,
   and run the preregistered leave-one-unseen-domain-out logistic probe.

This script never opens target GT masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError as exc:
    raise RuntimeError(
        "scikit-learn is required for S02-B. "
        "Install it in the active environment before running."
    ) from exc


VERSION = "2026-08-17-S02-B-v1"
BUILD = "S02_B_LOCK_UNLABELED_FEATURES_THEN_LODO_HARM_PROBE"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

PROTOCOL = ROOT / "docs" / "S02_A_unlabeled_harm_signal_feasibility_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "60b69a0fdd9888817ca8f5af7c8d416d3a69c3febf36f103c38df424ffde7e9e"

SOURCE_ONLY_DIR = ROOT / "outputs" / "S01_C_source_only_frozen_seeds_evaluation_v1"
SOURCE_ONLY_LOCK = SOURCE_ONLY_DIR / "NO_LABEL_PREDICTION_LOCK.json"
EXPECTED_SOURCE_ONLY_LOCK_SHA256 = "64cdd6c69f7287b6caa3f348eb1c5516be9bd04bc4e1834c842a1782dac1ec60"

D1_DIR = ROOT / "outputs" / "S01_D1_tta_feasibility_v1"
D1_LOCK = D1_DIR / "NO_LABEL_TTA_LOCK.json"
EXPECTED_D1_LOCK_SHA256 = "cafb77c03bbc8d3a2a5bc19ce69231b55daf9c73234f8d12431c99c179b743ae"
D1_DELTA = D1_DIR / "per_image_delta_dice.csv"

OUTPUT_DIR = ROOT / "outputs" / "S02_B_unlabeled_harm_signal_feasibility_v1"

SEEDS = (20260817, 20260818, 20260819)
UNSEEN_DOMAINS = (
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)
EXPECTED_UNSEEN_N = {
    "CVC-ColonDB": 380,
    "CVC-300": 60,
    "ETIS-LaribPolypDB": 196,
}

HARM_THRESHOLD = -0.02
POOLED_AUROC_GATE = 0.70
DOMAIN_AUROC_GATE = 0.65
MIN_PASSING_DOMAINS = 2

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


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sigmoid_np(z):
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def entropy_np_from_prob(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def boundary_density(mask):
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise RuntimeError(f"Expected 2D mask, got {m.shape}")
    h = np.mean(m[:, 1:] != m[:, :-1]) if m.shape[1] > 1 else 0.0
    v = np.mean(m[1:, :] != m[:-1, :]) if m.shape[0] > 1 else 0.0
    return float(0.5 * (h + v))


def summarize_logit(z):
    p = sigmoid_np(z)
    h = entropy_np_from_prob(p)
    m = p >= 0.5
    conf = 2.0 * np.abs(p - 0.5)
    return {
        "p": p,
        "h": h,
        "m": m,
        "entropy_mean": float(h.mean()),
        "entropy_std": float(h.std()),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
        "prob_q10": float(np.quantile(p, 0.10)),
        "prob_q50": float(np.quantile(p, 0.50)),
        "prob_q90": float(np.quantile(p, 0.90)),
        "fg_fraction": float(m.mean()),
        "boundary_density": boundary_density(m),
        "confidence_mean": float(conf.mean()),
    }


def validate_protocol():
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    actual = file_sha256(PROTOCOL)
    if actual.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "S02-A protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual  : {actual}"
        )
    return actual


def resolve_source_only_files():
    if not SOURCE_ONLY_LOCK.exists():
        raise FileNotFoundError(SOURCE_ONLY_LOCK)
    if file_sha256(SOURCE_ONLY_LOCK).lower() != EXPECTED_SOURCE_ONLY_LOCK_SHA256.lower():
        raise RuntimeError("S01-C no-label lock SHA mismatch.")

    lock = json.loads(SOURCE_ONLY_LOCK.read_text(encoding="utf-8"))
    entries = lock.get("locked_files", {})
    resolved = {}

    for seed in SEEDS:
        meta = entries.get(str(seed))
        if meta is None:
            raise RuntimeError(f"S01-C lock missing seed {seed}")
        basename = Path(meta["path"]).name
        path = SOURCE_ONLY_DIR / "locked_no_label_logits" / basename
        if not path.exists():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual.lower() != str(meta["sha256"]).lower():
            raise RuntimeError(f"S01-C locked logit SHA mismatch seed={seed}")
        resolved[seed] = path

    return resolved


def resolve_d1_tent_files():
    if not D1_LOCK.exists():
        raise FileNotFoundError(D1_LOCK)
    if file_sha256(D1_LOCK).lower() != EXPECTED_D1_LOCK_SHA256.lower():
        raise RuntimeError("S01-D1 no-label TTA lock SHA mismatch.")

    lock = json.loads(D1_LOCK.read_text(encoding="utf-8"))
    entries = lock.get("entries", {})
    resolved = {}

    for seed in SEEDS:
        key = f"TENT::{seed}"
        meta = entries.get(key)
        if meta is None:
            raise RuntimeError(f"D1 lock missing {key}")

        logit_name = Path(meta["logits_path"]).name
        diag_name = Path(meta["diagnostics_path"]).name

        logit_path = D1_DIR / "locked_no_label_tta_logits" / logit_name
        diag_path = D1_DIR / "no_label_diagnostics" / diag_name

        if not logit_path.exists():
            raise FileNotFoundError(logit_path)
        if not diag_path.exists():
            raise FileNotFoundError(diag_path)

        if file_sha256(logit_path) != meta["logits_sha256"]:
            raise RuntimeError(f"D1 TENT logit SHA mismatch seed={seed}")
        if file_sha256(diag_path) != meta["diagnostics_sha256"]:
            raise RuntimeError(f"D1 TENT diagnostics SHA mismatch seed={seed}")

        resolved[seed] = {
            "logits": logit_path,
            "diagnostics": diag_path,
        }

    return resolved


def extract_no_label_features(source_files, tent_files):
    rows = []

    for seed in SEEDS:
        src = np.load(source_files[seed], allow_pickle=False)
        tnt = np.load(tent_files[seed]["logits"], allow_pickle=False)

        src_ids = src["sample_ids"].tolist()
        tnt_ids = tnt["sample_ids"].tolist()
        src_roles = src["roles"].tolist()
        tnt_roles = tnt["roles"].tolist()
        src_domains = src["datasets"].tolist()
        tnt_domains = tnt["datasets"].tolist()

        if src_ids != tnt_ids:
            raise RuntimeError(f"Source/TENT sample order mismatch seed={seed}")
        if src_roles != tnt_roles or src_domains != tnt_domains:
            raise RuntimeError(f"Source/TENT metadata mismatch seed={seed}")

        src_logits = src["logits"]
        tnt_logits = tnt["logits"]

        if src_logits.shape != tnt_logits.shape:
            raise RuntimeError(f"Source/TENT logit shape mismatch seed={seed}")

        pbar = tqdm(
            range(len(src_ids)),
            desc=f"S02 no-label features seed {seed}",
            unit="img",
            dynamic_ncols=True,
        )

        for i in pbar:
            s = summarize_logit(src_logits[i].astype(np.float32))
            t = summarize_logit(tnt_logits[i].astype(np.float32))

            row = {
                "seed": seed,
                "sample_id": src_ids[i],
                "role": src_roles[i],
                "dataset": src_domains[i],

                "source_entropy_mean": s["entropy_mean"],
                "source_entropy_std": s["entropy_std"],
                "source_prob_mean": s["prob_mean"],
                "source_prob_std": s["prob_std"],
                "source_prob_q10": s["prob_q10"],
                "source_prob_q50": s["prob_q50"],
                "source_prob_q90": s["prob_q90"],
                "source_fg_fraction": s["fg_fraction"],
                "source_boundary_density": s["boundary_density"],
                "source_confidence_mean": s["confidence_mean"],

                "tent_entropy_mean": t["entropy_mean"],
                "tent_entropy_std": t["entropy_std"],
                "tent_prob_mean": t["prob_mean"],
                "tent_prob_std": t["prob_std"],
                "tent_prob_q10": t["prob_q10"],
                "tent_prob_q50": t["prob_q50"],
                "tent_prob_q90": t["prob_q90"],
                "tent_fg_fraction": t["fg_fraction"],
                "tent_boundary_density": t["boundary_density"],
                "tent_confidence_mean": t["confidence_mean"],

                "entropy_mean_shift": t["entropy_mean"] - s["entropy_mean"],
                "prob_abs_change_mean": float(np.mean(np.abs(t["p"] - s["p"]))),
                "mask_disagreement_fraction": float(np.mean(t["m"] != s["m"])),
                "abs_fg_fraction_shift": abs(t["fg_fraction"] - s["fg_fraction"]),
            }
            rows.append(row)

    return rows


def validate_no_label_feature_rows(rows):
    expected_total = 798 * len(SEEDS)
    if len(rows) != expected_total:
        raise RuntimeError(
            f"Feature-row count mismatch: expected={expected_total}, actual={len(rows)}"
        )

    keys = [(int(r["seed"]), r["sample_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (seed, sample_id) in feature table.")

    for r in rows:
        for f in FEATURE_NAMES:
            value = float(r[f])
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Non-finite feature {f} for seed={r['seed']} sample={r['sample_id']}"
                )


def build_harm_label_map():
    """
    PHASE B only. Reads already frozen D1 label-derived DeltaDice table.
    No target mask file is opened here.
    """
    if not D1_DELTA.exists():
        raise FileNotFoundError(D1_DELTA)

    rows, fields = read_csv(D1_DELTA)
    required = {"method", "seed", "sample_id", "role", "dataset", "delta_dice"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"D1 DeltaDice file missing columns: {missing}")

    labels = {}
    for r in rows:
        if r["method"] != "TENT":
            continue
        seed = int(r["seed"])
        delta = float(r["delta_dice"])
        key = (seed, r["sample_id"])
        labels[key] = {
            "role": r["role"],
            "dataset": r["dataset"],
            "delta_dice": delta,
            "harmful": int(delta <= HARM_THRESHOLD),
        }

    expected = 798 * len(SEEDS)
    if len(labels) != expected:
        raise RuntimeError(
            f"TENT harm-label count mismatch: expected={expected}, actual={len(labels)}"
        )
    return labels


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=20260817,
            ),
        ),
    ])


def matrix_from_rows(rows):
    x = np.asarray(
        [[float(r[f]) for f in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )
    return x


def run_lodo(labeled_rows):
    unseen = [r for r in labeled_rows if r["role"] == "unseen_locked"]

    expected = sum(EXPECTED_UNSEEN_N.values()) * len(SEEDS)
    if len(unseen) != expected:
        raise RuntimeError(
            f"Primary unseen image-seed count mismatch: expected={expected}, "
            f"actual={len(unseen)}"
        )

    fold_rows = []
    heldout_predictions = []

    for heldout in UNSEEN_DOMAINS:
        train = [r for r in unseen if r["dataset"] != heldout]
        test = [r for r in unseen if r["dataset"] == heldout]

        expected_test = EXPECTED_UNSEEN_N[heldout] * len(SEEDS)
        if len(test) != expected_test:
            raise RuntimeError(
                f"Held-out count mismatch {heldout}: "
                f"expected={expected_test}, actual={len(test)}"
            )

        x_train = matrix_from_rows(train)
        y_train = np.asarray([int(r["harmful"]) for r in train], dtype=np.int64)
        x_test = matrix_from_rows(test)
        y_test = np.asarray([int(r["harmful"]) for r in test], dtype=np.int64)

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(f"Training fold lacks both classes: heldout={heldout}")
        if len(np.unique(y_test)) != 2:
            raise RuntimeError(f"Held-out fold lacks both classes: heldout={heldout}")

        model = make_model()
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]

        auroc = float(roc_auc_score(y_test, score))
        auprc = float(average_precision_score(y_test, score))

        fold_rows.append({
            "heldout_domain": heldout,
            "n": len(test),
            "harmful_n": int(y_test.sum()),
            "harmful_prevalence": f"{float(y_test.mean()):.10f}",
            "auroc": f"{auroc:.10f}",
            "auprc": f"{auprc:.10f}",
            "passes_domain_gate": bool(auroc >= DOMAIN_AUROC_GATE),
        })

        for r, y, s in zip(test, y_test.tolist(), score.tolist()):
            heldout_predictions.append({
                "seed": r["seed"],
                "sample_id": r["sample_id"],
                "dataset": r["dataset"],
                "harmful": int(y),
                "lodo_harm_score": f"{float(s):.10f}",
            })

    y_all = np.asarray(
        [int(r["harmful"]) for r in heldout_predictions],
        dtype=np.int64,
    )
    s_all = np.asarray(
        [float(r["lodo_harm_score"]) for r in heldout_predictions],
        dtype=np.float64,
    )

    pooled_auroc = float(roc_auc_score(y_all, s_all))
    pooled_auprc = float(average_precision_score(y_all, s_all))
    passing_domains = sum(bool(r["passes_domain_gate"]) for r in fold_rows)

    go = (
        pooled_auroc >= POOLED_AUROC_GATE
        and passing_domains >= MIN_PASSING_DOMAINS
    )

    decision = (
        "GO_TO_S03_SOURCE_SIDE_SAFETTA_DEVELOPMENT"
        if go
        else "STOP_COMPLEX_SAFETTA_GATE_SIGNAL_NOT_ROBUST"
    )

    return (
        fold_rows,
        heldout_predictions,
        pooled_auroc,
        pooled_auprc,
        passing_domains,
        decision,
    )


def run(args):
    # ---------------------------
    # Immutable preflight.
    # ---------------------------
    protocol_sha = validate_protocol()
    source_files = resolve_source_only_files()
    tent_files = resolve_d1_tent_files()

    if args.output_dir.exists():
        raise FileExistsError(
            f"S02-B final output already exists: {args.output_dir}\n"
            "Refusing to overwrite the first frozen S02-B result."
        )

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists():
        if not args.technical_rerun:
            raise FileExistsError(
                f"Partial S02-B build exists: {build_dir}\n"
                "Use --technical-rerun only after a purely technical failure."
            )
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=False)

    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Protocol SHA256: {protocol_sha}")
    print(f"Source-only lock SHA256: {EXPECTED_SOURCE_ONLY_LOCK_SHA256}")
    print(f"D1 TTA lock SHA256: {EXPECTED_D1_LOCK_SHA256}")
    print("Phase A target GT masks opened=NO")
    print("Phase A DeltaDice labels read=NO")
    print()

    # ---------------------------
    # PHASE A — no-label features.
    # ---------------------------
    feature_rows = extract_no_label_features(
        source_files=source_files,
        tent_files=tent_files,
    )
    validate_no_label_feature_rows(feature_rows)

    feature_csv = build_dir / "NO_LABEL_FEATURES_LOCKED.csv"
    feature_fields = ["seed", "sample_id", "role", "dataset"] + FEATURE_NAMES
    write_csv(feature_csv, feature_rows, feature_fields)
    feature_sha = file_sha256(feature_csv)

    feature_lock = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_only_lock_sha256": EXPECTED_SOURCE_ONLY_LOCK_SHA256,
        "d1_tta_lock_sha256": EXPECTED_D1_LOCK_SHA256,
        "feature_table": str(feature_csv),
        "feature_table_sha256": feature_sha,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "row_count": len(feature_rows),
        "target_masks_opened_before_feature_lock": False,
        "delta_dice_labels_read_before_feature_lock": False,
    }

    feature_lock_path = build_dir / "NO_LABEL_FEATURE_LOCK.json"
    feature_lock_path.write_text(
        json.dumps(feature_lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_lock_sha = file_sha256(feature_lock_path)

    print("===== S02-B NO-LABEL FEATURE LOCK COMPLETE =====")
    print(f"Feature rows: {len(feature_rows)}")
    print(f"Feature count: {len(FEATURE_NAMES)}")
    print(f"Feature CSV SHA256: {feature_sha}")
    print(f"Feature lock SHA256: {feature_lock_sha}")
    print("Target masks opened before feature lock: NO")
    print("DeltaDice labels read before feature lock: NO")
    print()

    # Re-verify immediately before retrospective label merge.
    if file_sha256(feature_csv) != feature_sha:
        raise RuntimeError("No-label feature table changed before label merge.")
    if file_sha256(feature_lock_path) != feature_lock_sha:
        raise RuntimeError("No-label feature lock changed before label merge.")

    # ---------------------------
    # PHASE B — frozen harm labels.
    # ---------------------------
    label_map = build_harm_label_map()

    labeled_rows = []
    for row in feature_rows:
        key = (int(row["seed"]), row["sample_id"])
        label = label_map.get(key)
        if label is None:
            raise RuntimeError(f"Missing frozen harm label: {key}")
        if label["role"] != row["role"] or label["dataset"] != row["dataset"]:
            raise RuntimeError(f"Feature/label metadata mismatch: {key}")

        merged = dict(row)
        merged["delta_dice"] = label["delta_dice"]
        merged["harmful"] = label["harmful"]
        labeled_rows.append(merged)

    write_csv(
        build_dir / "retrospective_labeled_feature_table.csv",
        labeled_rows,
        feature_fields + ["delta_dice", "harmful"],
    )

    (
        fold_rows,
        predictions,
        pooled_auroc,
        pooled_auprc,
        passing_domains,
        decision,
    ) = run_lodo(labeled_rows)

    write_csv(
        build_dir / "lodo_domain_summary.csv",
        fold_rows,
        [
            "heldout_domain", "n", "harmful_n", "harmful_prevalence",
            "auroc", "auprc", "passes_domain_gate",
        ],
    )
    write_csv(
        build_dir / "lodo_heldout_predictions.csv",
        predictions,
        [
            "seed", "sample_id", "dataset", "harmful", "lodo_harm_score",
        ],
    )

    result = {
        "script_version": VERSION,
        "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "no_label_feature_lock_sha256": feature_lock_sha,
        "feature_count": len(FEATURE_NAMES),
        "classifier": {
            "type": "StandardScaler + LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 20260817,
        },
        "harm_threshold": HARM_THRESHOLD,
        "pooled_lodo_auroc": pooled_auroc,
        "pooled_lodo_auprc": pooled_auprc,
        "pooled_auroc_gate": POOLED_AUROC_GATE,
        "domain_auroc_gate": DOMAIN_AUROC_GATE,
        "passing_domains": passing_domains,
        "required_passing_domains": MIN_PASSING_DOMAINS,
        "decision": decision,
    }
    (build_dir / "decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = [
        "===== S02-B UNLABELED HARM-SIGNAL FEASIBILITY =====",
        f"Script version: {VERSION}",
        f"Build: {BUILD}",
        "",
        "No-label feature locking:",
        f"  feature_count={len(FEATURE_NAMES)}",
        f"  feature_rows={len(feature_rows)}",
        f"  feature_csv_sha256={feature_sha}",
        f"  feature_lock_sha256={feature_lock_sha}",
        "  target masks opened before feature lock=NO",
        "  DeltaDice labels read before feature lock=NO",
        "",
        "Primary probe:",
        "  method=TENT",
        "  classifier=StandardScaler + LogisticRegression(C=1.0, balanced)",
        "  evaluation=leave-one-unseen-domain-out",
        "",
        "Held-out unseen domains:",
    ]

    for r in fold_rows:
        summary.append(
            f"  {r['heldout_domain']:22s} "
            f"N={r['n']:4d} "
            f"harmPrev={float(r['harmful_prevalence']):.3f} "
            f"AUROC={float(r['auroc']):.6f} "
            f"AUPRC={float(r['auprc']):.6f} "
            f"Pass={r['passes_domain_gate']}"
        )

    summary += [
        "",
        f"Pooled LODO AUROC={pooled_auroc:.6f}",
        f"Pooled LODO AUPRC={pooled_auprc:.6f}",
        f"Domains AUROC>={DOMAIN_AUROC_GATE:.2f}: "
        f"{passing_domains}/3",
        "",
        "Frozen gate:",
        f"  pooled AUROC >= {POOLED_AUROC_GATE:.2f}",
        f"  and >= {MIN_PASSING_DOMAINS}/3 domains AUROC >= {DOMAIN_AUROC_GATE:.2f}",
        "",
        f"Decision: {decision}",
    ]

    (build_dir / "summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)

    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    # Feature math.
    z0 = np.zeros((8, 8), dtype=np.float32)
    s = summarize_logit(z0)
    assert abs(s["prob_mean"] - 0.5) < 1e-6
    assert abs(s["entropy_mean"] - math.log(2.0)) < 1e-5
    assert abs(s["confidence_mean"]) < 1e-6

    checker = (np.indices((8, 8)).sum(axis=0) % 2).astype(bool)
    bd = boundary_density(checker)
    assert 0.99 <= bd <= 1.0

    # Frozen feature count and no identity features.
    assert len(FEATURE_NAMES) == 24
    assert "dataset" not in FEATURE_NAMES
    assert "seed" not in FEATURE_NAMES
    assert "delta_dice" not in FEATURE_NAMES

    # Synthetic three-domain LODO probe.
    rng = np.random.RandomState(7)
    toy = []
    domains = ["CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
    for d_idx, domain in enumerate(domains):
        for i in range(60):
            y = i % 2
            row = {
                "seed": SEEDS[i % 3],
                "sample_id": f"{domain}_{i}",
                "role": "unseen_locked",
                "dataset": domain,
                "harmful": y,
            }
            for j, name in enumerate(FEATURE_NAMES):
                row[name] = (
                    (1.5 if j == 0 else 0.05) * y
                    + rng.normal(0, 0.25)
                    + 0.01 * d_idx
                )
            toy.append(row)

    # Direct miniature LODO independent of project count assertions.
    held = []
    for domain in domains:
        tr = [r for r in toy if r["dataset"] != domain]
        te = [r for r in toy if r["dataset"] == domain]
        model = make_model()
        model.fit(
            matrix_from_rows(tr),
            np.asarray([r["harmful"] for r in tr]),
        )
        score = model.predict_proba(matrix_from_rows(te))[:, 1]
        auc = roc_auc_score(
            np.asarray([r["harmful"] for r in te]),
            score,
        )
        held.append(float(auc))
    assert min(held) > 0.9

    assert HARM_THRESHOLD == -0.02
    assert POOLED_AUROC_GATE == 0.70
    assert DOMAIN_AUROC_GATE == 0.65
    assert MIN_PASSING_DOMAINS == 2

    print("FEATURE_MATH_TEST_PASS")
    print("BOUNDARY_DENSITY_TEST_PASS")
    print("FROZEN_FEATURE_SET_TEST_PASS")
    print("LODO_LOGISTIC_TEST_PASS")
    print("FROZEN_GATE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "S02-B: lock target-GT-free Source/TENT features, then run the "
            "frozen leave-one-unseen-domain-out harm-prediction feasibility probe."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--technical-rerun",
        action="store_true",
        help="Only for a technical failure before a final S02-B result is committed.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
