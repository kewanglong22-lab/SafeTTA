#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix3.py

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

EXPECTED_R13B_FIX2_SCRIPT_SHA = (
    "6f7ce6a3b15abf1205dc77703320ba823b79d392dc4f43beb51e0a7b94a2521d"
)

R05D3_CANDIDATES = (
    "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py",
    "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1.py",
)

R13B_CANDIDATES = (
    "Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix2.py",
    "Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix1.py",
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
    """
    Deterministic historical-script resolution.

    Exact ordered filenames are authoritative. The first existing candidate
    wins (Fix2 before Fix1). Broad token search is used only if none of the
    exact filenames exist, and that fallback must be unique.
    """
    for name in candidates:
        p = CODE / name
        if not p.is_file():
            continue

        if required_token is not None:
            txt = p.read_text(encoding="utf-8")
            if required_token not in txt:
                raise RuntimeError(
                    f"Exact historical candidate exists but lacks required "
                    f"token {required_token}: {p}"
                )

        print(f"historical script exact-name resolution: {p.name}")
        return p

    if required_token is None:
        raise RuntimeError(
            f"No exact historical script found. candidates={candidates}"
        )

    token_hits = []
    for p in CODE.glob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="utf-8-sig")

        if required_token in txt:
            token_hits.append(p)

    token_hits = sorted(
        set(token_hits),
        key=lambda p: str(p).lower(),
    )

    if len(token_hits) != 1:
        raise RuntimeError(
            "Exact historical filename was not found and token fallback "
            f"is ambiguous for {required_token}:\n"
            + "\n".join(str(p) for p in token_hits)
        )

    print(
        "historical script token-fallback resolution: "
        f"{token_hits[0].name}"
    )
    return token_hits[0]


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



def get_function_node(tree, name):
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == name
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one function {name}, found {len(hits)}"
        )
    return hits[0]


def count_method_calls(func_node, method_name, object_name=None):
    count = 0
    for n in ast.walk(func_node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not isinstance(f, ast.Attribute) or f.attr != method_name:
            continue
        if object_name is not None:
            if not isinstance(f.value, ast.Name) or f.value.id != object_name:
                continue
        count += 1
    return count


def count_adam_calls(func_node):
    count = 0
    for n in ast.walk(func_node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        # torch.optim.Adam(...)
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "Adam"
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "optim"
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "torch"
        ):
            count += 1
    return count


def audit_r13b_pl_semantics(path: Path):
    """
    Structural audit of the actual frozen R13B Fix2 implementation.

    The one-step semantic is NOT required to appear as a literal `steps=1`
    string in the runtime loop. R13B inherits/fixes the action through the
    R13A lock, explicitly validates sem['steps']==1, and each adaptation
    routine contains exactly one optimizer step call.
    """
    digest = sha256_file(path)
    if digest != EXPECTED_R13B_FIX2_SCRIPT_SHA:
        raise RuntimeError(
            f"R13B Fix2 SHA mismatch: expected="
            f"{EXPECTED_R13B_FIX2_SCRIPT_SHA}, got={digest}"
        )

    source = path.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", "", source)
    tree = ast.parse(source)

    validate_node = get_function_node(tree, "validate_lineage")
    binary_node = get_function_node(tree, "run_binary_state")
    seg_node = get_function_node(tree, "run_segformer_state")

    # Frozen R13A-lock semantic gates embedded in R13B.
    semantic_checks = {
        "action_identity": (
            'ACTION="A4_PL_CONF90_1STEP"' in normalized
        ),
        "confidence_090_via_r13a_lock": (
            'sem.get("confidence_threshold",math.nan)' in normalized
            and "!=0.90" in normalized
        ),
        "optimizer_adam_via_r13a_lock": (
            'str(sem.get("optimizer"))!="Adam"' in normalized
        ),
        "lr_1e3_via_r13a_lock": (
            'float(sem.get("lr",math.nan))!=1e-3' in normalized
        ),
        "weight_decay_zero_via_r13a_lock": (
            'float(sem.get("weight_decay",math.nan))!=0.0' in normalized
        ),
        "one_step_via_r13a_lock": (
            'int(sem.get("steps",-1))!=1' in normalized
        ),
        "episodic_reset_via_r13a_lock": (
            'sem.get("episodic_source_reset",False)' in normalized
        ),
        "same_case_post_update_via_r13a_lock": (
            'sem.get("same_case_post_update_prediction",False)' in normalized
        ),
    }

    # Runtime implementation: exactly one opt.step() in each family routine.
    binary_opt_steps = count_method_calls(
        binary_node, "step", object_name="opt"
    )
    seg_opt_steps = count_method_calls(
        seg_node, "step", object_name="opt"
    )
    binary_adam = count_adam_calls(binary_node)
    seg_adam = count_adam_calls(seg_node)

    semantic_checks["binary_exactly_one_optimizer_step_call"] = (
        binary_opt_steps == 1
    )
    semantic_checks["segformer_exactly_one_optimizer_step_call"] = (
        seg_opt_steps == 1
    )
    semantic_checks["binary_adam_instantiation"] = (
        binary_adam == 1
    )
    semantic_checks["segformer_adam_instantiation"] = (
        seg_adam == 1
    )

    # Full episodic parameter+buffer restore must be present repeatedly in
    # both runtime functions. Three restores correspond to:
    # teacher reset, pre-adaptation reset, post-case cleanup.
    def body_text(node):
        lines = source.splitlines()
        return "\n".join(lines[node.lineno - 1:node.end_lineno])

    binary_body = body_text(binary_node)
    seg_body = body_text(seg_node)

    semantic_checks["binary_full_parameter_reset"] = (
        binary_body.count("r13a.restore(params,source_values)") >= 3
    )
    semantic_checks["binary_full_buffer_reset"] = (
        binary_body.count("r13a.restore_buffers(source_buffers)") >= 3
    )
    semantic_checks["segformer_full_parameter_reset"] = (
        seg_body.count("r13a.restore(params,source_values)") >= 3
    )
    semantic_checks["segformer_full_buffer_reset"] = (
        seg_body.count("r13a.restore_buffers(source_buffers)") >= 3
    )

    # Loss/objective lineage by family.
    semantic_checks["binary_masked_pseudolabel_loss"] = (
        "r13a.binary_masked_loss(" in binary_body
        and "r13a.binary_pseudolabel(" in binary_body
    )
    semantic_checks["segformer_masked_pseudolabel_loss"] = (
        "r13a.categorical_masked_loss(" in seg_body
        and "r13a.categorical_pseudolabel(" in seg_body
    )

    rows = []
    for name, ok in semantic_checks.items():
        print(
            f"R13B structural semantic {name}: "
            f"{'PASS' if ok else 'FAIL'}"
        )
        rows.append({
            "semantic_group": name,
            "matched_tokens": "AST_OR_EXACT_SOURCE_EVIDENCE",
            "pass": int(bool(ok)),
        })

    print(
        "R13B runtime optimizer-step counts: "
        f"binary={binary_opt_steps}, segformer={seg_opt_steps}"
    )
    print(
        "R13B runtime Adam counts: "
        f"binary={binary_adam}, segformer={seg_adam}"
    )

    failed = [k for k, v in semantic_checks.items() if not v]
    if failed:
        raise RuntimeError(
            "R13B structural semantic audit failed: "
            + ", ".join(failed)
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-manifest", type=Path, default=FINAL_MANIFEST)
    ap.add_argument("--final-lock", type=Path, default=FINAL_LOCK)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUTPUTS
            / "Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix3_v1"
        ),
    )
    args = ap.parse_args()

    print("===== Q1 R14C0 FIX3 SUN-SEG PRE-GT PREDICTION RUNTIME LINEAGE PREFLIGHT =====")
    print("HISTORICAL_SCRIPT_RESOLUTION=ORDERED_EXACT_FILENAME_FIRST")
    print("SCIENTIFIC_PROTOCOL_CHANGE_FROM_FIX2=NO")
    print("PL_ONE_STEP_AUDIT=AST_PLUS_R13A_LOCK_SEMANTICS")
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

    pl_semantics = audit_r13b_pl_semantics(r13b)

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
        "pl_one_step_audit_method": "R13A lock steps==1 + AST exact one opt.step per runtime family",
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
