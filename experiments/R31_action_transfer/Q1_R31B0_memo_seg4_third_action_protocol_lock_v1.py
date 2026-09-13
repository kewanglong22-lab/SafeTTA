#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R31B0
Third-Action MEMO-SEG4-1STEP Protocol + Source Case Split Lock

Purpose
-------
Freeze the genuinely third TTA action BEFORE any R31B third-action outcomes
or leave-one-action-out results are generated.

R31A already established strong two-action feasibility:
  decision = GO_STRONG_TO_R31B_THIRD_ACTION
  final lock SHA256 =
    a03d707e8f3c4054feba867a252b270f58a2838b141ce11f8cd70774c9cf3b62

R31B third action
-----------------
Name: MEMO-SEG4-1STEP

This is a segmentation-compatible instantiation of the published MEMO
principle, not a claim of bit-exact reproduction of the ImageNet code.

Published MEMO principles preserved:
  * one test point at a time;
  * multiple augmentations of that point;
  * minimize entropy of the MARGINAL prediction across augmentations;
  * adapt all model parameters;
  * one gradient update;
  * final prediction on the original input.

Segmentation-specific fixed instantiation:
  * four exactly reversible spatial transforms:
      identity, horizontal flip, vertical flip, horizontal+vertical flip;
  * each transformed prediction is inverse-transformed back to the original
    352x352 coordinate frame BEFORE the marginal probability is formed;
  * binary pixelwise marginal entropy is averaged over all pixels;
  * episodic reset to the original SOURCE checkpoint before every physical
    image, matching the existing SafeTTA transactional action semantics.

Source-only action-design split
-------------------------------
The 1000 NeoPolyp development images are deterministically split BEFORE any
MEMO outcome generation:

  200 physical cases : MEMO action-design subset
  800 physical cases : R31B leave-one-action-out evaluation subset

The split is based only on SHA256("R31B0|" + sample_id); no image content,
GT, Dice, HARM, or TTA result is used.

Only the 200 design cases may select the MEMO optimizer learning rate.
The 800 evaluation cases are forbidden for MEMO hyperparameter selection.

Prespecified optimizer selection
---------------------------------
Optimizer: Adam, all model parameters, weight_decay=0.
One update only.

Learning-rate candidates:
  1e-5, 3e-5, 1e-4, 3e-4

Selection uses ONLY the 200 action-design cases across all 9 frozen model
states:
  primary   : highest mean deployed Dice of MEMO-SEG4-1STEP
  tie-break1: lower HARM rate (DeltaDice <= -0.02)
  tie-break2: lower learning rate

After selection the chosen LR is frozen and may not be changed using the
800-case R31B evaluation panel or any external cohort.

R31B evaluation requirement
----------------------------
On the frozen 800-case evaluation subset, construct three actions:
  TENT1
  PL-CONF90
  MEMO-SEG4-1STEP

Then perform grouped physical-case leave-one-action-out:
  train TENT+PL   -> test MEMO
  train TENT+MEMO -> test PL
  train PL+MEMO   -> test TENT

The held-out action and held-out physical images must both be absent from
predictor fitting.

External information boundary
-----------------------------
EndoTect: FORBIDDEN
PolypGen: FORBIDDEN
SUN-SEG: FORBIDDEN
PROMISE12: FORBIDDEN
Any external outcome: FORBIDDEN

This script performs NO model inference and reads NO GT pixels. It only:
  * verifies the R31A strong-GO lock;
  * freezes the source case split;
  * writes the R31B third-action protocol lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


VERSION = "2026-09-12-R31B0-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")

R31A_DIR = ROOT / "R31A_action_conditional_harm_predictor_v1"
R31A_FINAL = R31A_DIR / "R31A_FINAL_LOCK.json"
EXPECTED_R31A_FINAL_SHA = (
    "a03d707e8f3c4054feba867a252b270f58a2838b141ce11f8cd70774c9cf3b62"
)
EXPECTED_R31A_DECISION = "GO_STRONG_TO_R31B_THIRD_ACTION"

SOURCE_MANIFEST = (
    ROOT
    / "outputs"
    / "Q1_R05D1_neopolyp_acquisition_pairing_duplicate_qc_v1_fix2"
    / "frozen_confirmatory_manifest.csv"
)

DEFAULT_OUT = ROOT / "R31B0_memo_seg4_third_action_protocol_lock_v1"

EXPECTED_SOURCE_CASES = 1000
DESIGN_CASES = 200
EVAL_CASES = 800

SPLIT_SALT = "R31B0|"

THIRD_ACTION = "MEMO-SEG4-1STEP"
AUGMENTATIONS = [
    "IDENTITY",
    "HFLIP",
    "VFLIP",
    "HVFLIP",
]
UPDATE_STEPS = 1
PARAMETER_SCOPE = "ALL_MODEL_PARAMETERS"
OPTIMIZER = "Adam"
WEIGHT_DECAY = 0.0
LR_CANDIDATES = [1e-5, 3e-5, 1e-4, 3e-4]

HARM_THRESHOLD = -0.02

FORBIDDEN_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def assert_no_external_path(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_TOKENS):
        raise RuntimeError(f"Forbidden external path in R31B0: {path}")


def verify_r31a():
    if not R31A_FINAL.is_file():
        raise FileNotFoundError(R31A_FINAL)

    got = sha256_file(R31A_FINAL)
    if got != EXPECTED_R31A_FINAL_SHA:
        raise RuntimeError(
            "R31A final lock SHA changed.\n"
            f"expected={EXPECTED_R31A_FINAL_SHA}\n"
            f"observed={got}"
        )

    payload = json.loads(R31A_FINAL.read_text(encoding="utf-8"))
    decision = str(payload.get("decision", ""))

    if decision != EXPECTED_R31A_DECISION:
        raise RuntimeError(
            f"R31A decision={decision!r}; "
            f"expected={EXPECTED_R31A_DECISION!r}"
        )

    if bool(payload.get("external_data_access", True)):
        raise RuntimeError("R31A lock indicates external-data access.")

    return payload, got


def read_source_ids():
    assert_no_external_path(SOURCE_MANIFEST)

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)

    rows = []
    with SOURCE_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "sample_id",
            "image_path",
            "gt_path",
            "eligible_confirmatory",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"Source manifest missing columns={sorted(missing)}"
            )

        for row in reader:
            if int(row["eligible_confirmatory"]) != 1:
                continue

            # IMPORTANT: paths are recorded but never opened by R31B0.
            image_path = Path(row["image_path"])
            gt_path = Path(row["gt_path"])
            assert_no_external_path(image_path)
            assert_no_external_path(gt_path)

            rows.append({
                "sample_id": str(row["sample_id"]),
                "image_path": str(image_path),
                "gt_path": str(gt_path),
            })

    if len(rows) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(
            f"Eligible SOURCE cases={len(rows)}, "
            f"expected={EXPECTED_SOURCE_CASES}"
        )

    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != EXPECTED_SOURCE_CASES:
        raise RuntimeError("SOURCE sample_id is not unique.")

    return rows


def deterministic_split(rows):
    scored = []
    for row in rows:
        sid = row["sample_id"]
        key = hashlib.sha256(
            (SPLIT_SALT + sid).encode("utf-8")
        ).hexdigest()
        scored.append((key, sid, row))

    scored.sort(key=lambda x: (x[0], x[1]))

    design = [x[2] for x in scored[:DESIGN_CASES]]
    evaluation = [x[2] for x in scored[DESIGN_CASES:]]

    if len(design) != DESIGN_CASES:
        raise RuntimeError("Design split size changed.")
    if len(evaluation) != EVAL_CASES:
        raise RuntimeError("Evaluation split size changed.")

    dset = {x["sample_id"] for x in design}
    eset = {x["sample_id"] for x in evaluation}

    if dset & eset:
        raise RuntimeError("Design/evaluation sample overlap.")
    if len(dset | eset) != EXPECTED_SOURCE_CASES:
        raise RuntimeError("Split does not cover all SOURCE cases.")

    return design, evaluation


def write_split(path: Path, design, evaluation):
    lookup = {}
    for r in design:
        lookup[r["sample_id"]] = "MEMO_ACTION_DESIGN"
    for r in evaluation:
        lookup[r["sample_id"]] = "R31B_LOAO_EVALUATION"

    rows = sorted(
        design + evaluation,
        key=lambda r: r["sample_id"],
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "r31b_partition",
                "image_path",
                "gt_path",
                "split_sha256_key",
            ],
        )
        w.writeheader()

        for r in rows:
            sid = r["sample_id"]
            w.writerow({
                "sample_id": sid,
                "r31b_partition": lookup[sid],
                "image_path": r["image_path"],
                "gt_path": r["gt_path"],
                "split_sha256_key": hashlib.sha256(
                    (SPLIT_SALT + sid).encode("utf-8")
                ).hexdigest(),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = ap.parse_args()

    print("=" * 120)
    print("SafeTTA R31B0 Third-Action Protocol Lock")
    print("Version:", VERSION)
    print("Third action                :", THIRD_ACTION)
    print("Model inference             : NO")
    print("GT pixel read/decode        : NO / NO")
    print("External cohort access      : NO")
    print("R31A required decision      :", EXPECTED_R31A_DECISION)
    print("=" * 120)

    r31a, r31a_sha = verify_r31a()
    source_rows = read_source_ids()
    design, evaluation = deterministic_split(source_rows)

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; do not overwrite a protocol lock: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    split_path = out / "R31B0_SOURCE_CASE_PARTITION.csv"
    write_split(split_path, design, evaluation)

    protocol = {
        "status": "LOCKED_BEFORE_ANY_THIRD_ACTION_OUTCOME_GENERATION",
        "version": VERSION,
        "upstream": {
            "R31A_final_lock": str(R31A_FINAL),
            "R31A_final_lock_sha256": r31a_sha,
            "R31A_decision": r31a["decision"],
        },
        "information_boundary": {
            "model_inference_in_R31B0": False,
            "gt_pixel_read_in_R31B0": False,
            "gt_decode_in_R31B0": False,
            "endotect_access": False,
            "polypgen_access": False,
            "sunseg_access": False,
            "promise12_access": False,
            "external_outcome_access": False,
        },
        "source_population": {
            "dataset": "NeoPolyp",
            "physical_cases": EXPECTED_SOURCE_CASES,
            "frozen_manifest": str(SOURCE_MANIFEST),
            "frozen_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "partition_rule": (
                'rank ascending by SHA256("R31B0|" + sample_id)'
            ),
            "partition_csv": str(split_path),
            "partition_csv_sha256": sha256_file(split_path),
            "memo_action_design_cases": DESIGN_CASES,
            "loao_evaluation_cases": EVAL_CASES,
            "case_overlap": 0,
        },
        "third_action": {
            "name": THIRD_ACTION,
            "method_parent": (
                "MEMO: marginal entropy minimization with one test point"
            ),
            "scope_note": (
                "Segmentation-compatible MEMO instantiation; not claimed as "
                "bit-exact reproduction of the original ImageNet code."
            ),
            "episodic": True,
            "reset_rule": (
                "restore exact frozen SOURCE checkpoint before every "
                "physical image"
            ),
            "augmentations": AUGMENTATIONS,
            "augmentation_count": len(AUGMENTATIONS),
            "alignment_rule": (
                "inverse-transform every augmented probability map back to "
                "original 352x352 coordinates before marginal averaging"
            ),
            "marginal_probability": (
                "pixelwise mean foreground probability across the four "
                "aligned augmented predictions"
            ),
            "loss": (
                "mean binary entropy of pixelwise marginal probability"
            ),
            "parameter_scope": PARAMETER_SCOPE,
            "update_steps": UPDATE_STEPS,
            "final_prediction": (
                "forward on the original unaugmented image after the single "
                "adaptation update"
            ),
            "optimizer_selection": {
                "optimizer": OPTIMIZER,
                "weight_decay": WEIGHT_DECAY,
                "learning_rate_candidates": LR_CANDIDATES,
                "selection_population": "MEMO_ACTION_DESIGN only",
                "primary": "highest mean deployed Dice across all 9 states",
                "tie_break_1": (
                    f"lower HARM rate, HARM defined as DeltaDice <= "
                    f"{HARM_THRESHOLD}"
                ),
                "tie_break_2": "lower learning rate",
                "evaluation_partition_use_for_selection": False,
            },
        },
        "r31b_evaluation": {
            "population": "R31B_LOAO_EVALUATION only",
            "physical_cases": EVAL_CASES,
            "actions": [
                "TENT1",
                "PL-CONF90",
                THIRD_ACTION,
            ],
            "predictor_feature_contract": (
                "q_source66 + delta_semantic64 + "
                "ActionID + FamilyActionID"
            ),
            "leave_one_action_out": [
                {
                    "train_actions": ["TENT1", "PL-CONF90"],
                    "test_action": THIRD_ACTION,
                },
                {
                    "train_actions": ["TENT1", THIRD_ACTION],
                    "test_action": "PL-CONF90",
                },
                {
                    "train_actions": ["PL-CONF90", THIRD_ACTION],
                    "test_action": "TENT1",
                },
            ],
            "case_split_rule": (
                "group by physical sample_id before predictor fitting"
            ),
            "heldout_physical_case_overlap": 0,
            "external_data_allowed": False,
        },
        "development_transparency": {
            "R31A_two_action_results_seen_before_R31B0": True,
            "third_action_outcomes_seen_before_R31B0": False,
            "R31B_LOAO_results_seen_before_R31B0": False,
            "memo_lr_selected_after_R31B0": True,
            "memo_lr_may_use_only_200_design_cases": True,
        },
    }

    protocol_path = out / "R31B0_MEMO_SEG4_PROTOCOL_LOCK.json"
    atomic_write_json(protocol_path, protocol)

    final = {
        "status": "PASS_R31B0_THIRD_ACTION_PROTOCOL_LOCK_COMPLETE",
        "version": VERSION,
        "third_action": THIRD_ACTION,
        "R31A_final_lock_sha256": r31a_sha,
        "source_partition_sha256": sha256_file(split_path),
        "protocol_sha256": sha256_file(protocol_path),
        "design_cases": DESIGN_CASES,
        "evaluation_cases": EVAL_CASES,
        "GT_pixels_read": False,
        "external_data_access": False,
        "next": "R31B1_MEMO_ACTION_DESIGN_AND_FREEZE",
    }

    final_path = out / "R31B0_FINAL_LOCK.json"
    atomic_write_json(final_path, final)

    print("\nR31B0 LOCK")
    print("  R31A strong-GO          : PASS", r31a_sha)
    print("  Source cases            :", EXPECTED_SOURCE_CASES)
    print("  MEMO action-design      :", DESIGN_CASES)
    print("  R31B LOAO evaluation    :", EVAL_CASES)
    print("  Third action            :", THIRD_ACTION)
    print("  Augmentations           :", AUGMENTATIONS)
    print("  Update steps            :", UPDATE_STEPS)
    print("  Parameter scope         :", PARAMETER_SCOPE)
    print("  LR candidates           :", LR_CANDIDATES)
    print("  GT pixels read/decode   : NO / NO")
    print("  External access         : NO")

    print("\nFINAL STATUS : PASS_R31B0_THIRD_ACTION_PROTOCOL_LOCK_COMPLETE")
    print("Protocol SHA256 :", sha256_file(protocol_path))
    print("Final lock SHA256:", sha256_file(final_path))
    print("Output           :", out)
    print("NEXT             : R31B1_MEMO_ACTION_DESIGN_AND_FREEZE")
    print("=" * 120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
