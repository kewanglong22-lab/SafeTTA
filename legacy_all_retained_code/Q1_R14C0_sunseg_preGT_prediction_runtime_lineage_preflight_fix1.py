#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix1.py

Purpose
-------
Pre-GT runtime lineage/preflight for the SUN-SEG confirmatory experiment.

This stage verifies:
- the R14B4B Fix3 final confirmatory manifest is exactly the locked 980-frame /
  49-case cohort;
- the final manifest SHA is unchanged from R14B3 (no exclusions);
- the historical SOURCE/TENT1 prediction script is present;
- the historical PL_CONF90 prediction script is present;
- both scripts contain the expected frozen adaptation semantics;
- exactly 9 source-only checkpoints are discoverable under the known project
  output root using family/seed/name constraints;
- checkpoint SHA256 values are printed and frozen for the next prediction stage.

This stage performs NO:
- SUN RGB decoding;
- GT decoding;
- model loading/inference;
- TTA execution;
- safety scoring;
- target outcome access;
- target tuning.

It exists to prevent guessing model/checkpoint/runtime lineage in R14C1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

FINAL_MANIFEST = (
    OUTPUTS
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
)

FINAL_LOCK = (
    OUTPUTS
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_SUNSEG_CONTAMINATION_AUDIT_LOCK.json"
)

EXPECTED_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)
EXPECTED_ROWS = 980
EXPECTED_CASES = 49

EXPECTED_R05D3_SCRIPT_SHA = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

R05D3_CANDIDATES = (
    "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py",
    "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1.py",
)

R13B_CANDIDATES = (
    "Q1_R13B_PL_CONF90_full_prediction_lock_fix2.py",
    "Q1_R13B_pl_conf90_full_prediction_lock_fix2.py",
    "Q1_R13B_PL_CONF90_prediction_lock_fix2.py",
)

FAMILIES = {
    "DeepLabV3-R50": {
        "seeds": ("20260817", "20260818", "20260819"),
        "checkpoint_name": "best_source_val_dice.pt",
        "path_tokens": ("deeplab",),
    },
    "PraNet": {
        "seeds": ("20260817", "20260818", "20260819"),
        "checkpoint_name": "best_source_val_dice.pt",
        "path_tokens": ("pranet",),
    },
    "SegFormer-B0": {
        "seeds": ("20260820", "20260821", "20260822"),
        "checkpoint_name": "best_model_state.pt",
        "path_tokens": ("segformer",),
    },
}

DECISION = "SUNSEG_PREGT_RUNTIME_LINEAGE_LOCKED_READY_FOR_PREDICTION_EXECUTION"


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)


def find_unique_script(candidates, required_token=None):
    found = []

    for name in candidates:
        p = CODE / name
        if p.is_file():
            found.append(p)

    if required_token:
        for p in CODE.glob("*.py"):
            try:
                txt = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                txt = p.read_text(encoding="utf-8-sig")
            if required_token in txt:
                found.append(p)

    found = sorted(set(found), key=lambda p: str(p).lower())
    if not found:
        raise RuntimeError(
            f"No historical script found. candidates={candidates} "
            f"required_token={required_token}"
        )

    # Prefer exact candidate names over broad token hits.
    exact = [p for p in found if p.name in candidates]
    if exact:
        found = exact

    if len(found) != 1:
        raise RuntimeError(
            "Historical script discovery is ambiguous:\n"
            + "\n".join(str(p) for p in found)
        )
    return found[0]


def assert_tokens(path: Path, groups, label):
    text = path.read_text(encoding="utf-8")
    results = []

    for group_name, alternatives in groups:
        matched = [t for t in alternatives if t in text]
        ok = bool(matched)
        results.append({
            "semantic_group": group_name,
            "matched_tokens": "|".join(matched),
            "pass": int(ok),
        })
        print(
            f"{label} semantic {group_name}: "
            f"{'PASS' if ok else 'FAIL'} | matched={matched}"
        )
        if not ok:
            raise RuntimeError(
                f"{label} missing semantic evidence for {group_name}. "
                f"Expected one of: {alternatives}"
            )

    return results


def discover_checkpoint(family, seed, spec):
    candidates = []
    target_name = spec["checkpoint_name"]

    for p in OUTPUTS.rglob(target_name):
        low = str(p).lower()

        if seed not in low:
            continue
        if not all(tok.lower() in low for tok in spec["path_tokens"]):
            continue

        candidates.append(p)

    candidates = sorted(
        set(candidates),
        key=lambda p: str(p).lower(),
    )

    if len(candidates) != 1:
        msg = (
            f"{family} seed={seed}: expected exactly one checkpoint "
            f"{target_name}, found {len(candidates)}"
        )
        if candidates:
            msg += "\n" + "\n".join(str(p) for p in candidates)
        raise RuntimeError(msg)

    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-manifest", type=Path, default=FINAL_MANIFEST)
    ap.add_argument("--final-lock", type=Path, default=FINAL_LOCK)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUTPUTS
            / "Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix1_v1"
        ),
    )
    args = ap.parse_args()

    print("===== Q1 R14C0 SUN-SEG PRE-GT PREDICTION RUNTIME LINEAGE PREFLIGHT =====")
    print("SUN_RGB_DECODE=NO")
    print("GT_PIXEL_DECODE=NO")
    print("MODEL_LOADING=NO")
    print("MODEL_INFERENCE=NO")
    print("TTA_EXECUTION=NO")
    print("SAFETY_SCORE_ACCESS=NO")
    print("TARGET_OUTCOME_ACCESS=NO")
    print("TARGET_TUNING=NO")

    for p in (args.final_manifest, args.final_lock):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest_sha = sha256_file(args.final_manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(
            f"Final SUN manifest SHA mismatch: {manifest_sha}"
        )

    cols, rows = read_csv(args.final_manifest)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Final SUN rows {len(rows)} != {EXPECTED_ROWS}"
        )

    if "cluster_id" not in cols:
        raise RuntimeError("Final SUN manifest missing cluster_id.")

    cases = {r["cluster_id"] for r in rows}
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"Final SUN physical cases {len(cases)} != {EXPECTED_CASES}"
        )

    lock = json.loads(
        args.final_lock.read_text(encoding="utf-8")
    )
    if lock.get("status") != "PASS":
        raise RuntimeError("R14B4B Fix3 lock status is not PASS.")
    if not lock.get("final_confirmatory_cohort_locked", False):
        raise RuntimeError(
            "R14B4B Fix3 final confirmatory cohort is not locked."
        )
    if (
        lock.get("final_manifest_sha256")
        != EXPECTED_MANIFEST_SHA
    ):
        raise RuntimeError(
            "R14B4B Fix3 lock final-manifest SHA mismatch."
        )

    print("\n===== FINAL SUN COHORT GATE =====")
    print("manifest SHA256:", manifest_sha)
    print("rows:", len(rows))
    print("physical cases:", len(cases))
    print("contamination exclusions:", 0)
    print("PASS")

    r05d3 = find_unique_script(
        R05D3_CANDIDATES,
        required_token="A1_TENT_1STEP",
    )
    r13b = find_unique_script(
        R13B_CANDIDATES,
        required_token="PL_CONF90_1STEP",
    )

    print("\n===== HISTORICAL ACTION SCRIPT LINEAGE =====")
    print("R05D3 SOURCE/TENT1 script:", r05d3)
    print("R05D3 SHA256:", sha256_file(r05d3))
    print("R13B PL_CONF90 script:", r13b)
    print("R13B SHA256:", sha256_file(r13b))

    r05d3_sha = sha256_file(r05d3)
    if r05d3.name == R05D3_CANDIDATES[0]:
        if r05d3_sha != EXPECTED_R05D3_SCRIPT_SHA:
            raise RuntimeError(
                f"Authoritative R05D3 fix1 SHA mismatch: {r05d3_sha}"
            )

    tent_semantics = assert_tokens(
        r05d3,
        [
            (
                "action_identity",
                (
                    "A1_TENT_1STEP",
                    "TENT_1STEP",
                ),
            ),
            (
                "one_step",
                (
                    "steps=1",
                    "steps = 1",
                    "adapt_steps=1",
                    "adapt_steps = 1",
                ),
            ),
            (
                "learning_rate_1e3",
                (
                    "1e-3",
                    "0.001",
                ),
            ),
            (
                "episodic_reset",
                (
                    "episodic",
                    "reset",
                ),
            ),
            (
                "source_masks_packed",
                (
                    "source_masks_packed",
                ),
            ),
            (
                "a1_masks_packed",
                (
                    "a1_masks_packed",
                ),
            ),
        ],
        "R05D3",
    )

    pl_semantics = assert_tokens(
        r13b,
        [
            (
                "action_identity",
                (
                    "PL_CONF90_1STEP",
                    "PL_CONF90",
                ),
            ),
            (
                "confidence_090",
                (
                    "0.90",
                    "0.9",
                    "CONF90",
                ),
            ),
            (
                "one_step",
                (
                    "steps=1",
                    "steps = 1",
                    "adapt_steps=1",
                    "adapt_steps = 1",
                    "one_step",
                ),
            ),
            (
                "episodic_reset",
                (
                    "episodic",
                    "reset",
                ),
            ),
        ],
        "R13B",
    )

    print("\n===== FROZEN CHECKPOINT DISCOVERY =====")
    ckpt_rows = []

    for family, spec in FAMILIES.items():
        for seed in spec["seeds"]:
            p = discover_checkpoint(
                family,
                seed,
                spec,
            )
            digest = sha256_file(p)
            size = p.stat().st_size

            print(
                f"{family} seed={seed}\n"
                f"  path={p}\n"
                f"  SHA256={digest}\n"
                f"  bytes={size}"
            )

            ckpt_rows.append({
                "model_family": family,
                "training_seed": seed,
                "checkpoint_path": str(p),
                "checkpoint_sha256": digest,
                "checkpoint_bytes": size,
            })

    if len(ckpt_rows) != 9:
        raise RuntimeError(
            f"Checkpoint count {len(ckpt_rows)} != 9"
        )

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    ckpt_csv = (
        args.output_dir
        / "R14C0_FROZEN_9_STATE_CHECKPOINT_LINEAGE.csv"
    )
    with ckpt_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_family",
                "training_seed",
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_bytes",
            ],
        )
        w.writeheader()
        w.writerows(ckpt_rows)

    semantic_csv = (
        args.output_dir
        / "R14C0_ACTION_SEMANTIC_TOKEN_AUDIT.csv"
    )
    sem_rows = (
        [
            {
                "script": "R05D3_SOURCE_TENT1",
                **r,
            }
            for r in tent_semantics
        ]
        + [
            {
                "script": "R13B_PL_CONF90",
                **r,
            }
            for r in pl_semantics
        ]
    )

    with semantic_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "script",
                "semantic_group",
                "matched_tokens",
                "pass",
            ],
        )
        w.writeheader()
        w.writerows(sem_rows)

    runtime_lock = {
        "status": "PASS",
        "decision": DECISION,
        "sun_confirmatory_cohort": {
            "manifest_path": str(args.final_manifest),
            "manifest_sha256": manifest_sha,
            "rows": len(rows),
            "physical_cases": len(cases),
        },
        "historical_action_scripts": {
            "source_tent1": {
                "path": str(r05d3),
                "sha256": r05d3_sha,
            },
            "pl_conf90": {
                "path": str(r13b),
                "sha256": sha256_file(r13b),
            },
        },
        "checkpoint_count": len(ckpt_rows),
        "checkpoint_lineage": ckpt_rows,
        "action_semantic_audit": sem_rows,
        "information_boundary": {
            "sun_rgb_decoded": False,
            "gt_pixels_decoded": False,
            "model_loaded": False,
            "model_inference": False,
            "tta_executed": False,
            "safety_scores_accessed": False,
            "target_outcomes_accessed": False,
            "target_tuning": False,
        },
        "next_stage": (
            "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK_PRE_GT"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14C0_SUNSEG_PREGT_RUNTIME_LINEAGE_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(
            runtime_lock,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nDecision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


if __name__ == "__main__":
    main()
