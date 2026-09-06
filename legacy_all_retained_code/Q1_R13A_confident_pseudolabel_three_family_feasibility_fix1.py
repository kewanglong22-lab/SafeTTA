#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R13A_confident_pseudolabel_three_family_feasibility_fix1.py

R13A — New, separately pre-registered cross-TTA candidate:
A4_PL_CONF90_1STEP

Purpose:
- test technical feasibility and unlabeled non-degeneracy only;
- NO GT / Dice / DeltaDice / HARM / safety-score evaluation;
- NO PolypGen access;
- NO hyperparameter sweep.

To isolate the adaptation objective from parameterization, R13A reuses the
exact R05D3 three-family TENT trainable-parameter panels and one-step Adam
optimizer settings. The only algorithmic change is the loss:
source-prediction hard pseudo-label self-training on pixels with confidence
>= 0.90.

Pseudo-label confidence threshold 0.90 is inherited from the historical
pre-existing pseudo-label/TestFit feasibility protocol, but the new three-family
action is frozen here BEFORE its outcomes are seen.

Action output is the same-case post-update segmentation, with episodic SOURCE
reset before the next physical image.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE, OUT = ROOT/"code", ROOT/"outputs"

R05D3 = CODE/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
R05D3_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
R05D3_OUT = OUT/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"

R12D_LOCK = (
    OUT/"Q1_R12D_sar_transfer_degeneracy_stop_lock_fix1_v1"
    /"R12D_SAR_TRANSFER_DEGENERACY_STOP_LOCK.json"
)
R12D_SHA = "2fd7ce015a7598490470ff29d3c2dd1a085a578af2ed73d34bee0bc9ca7eb04f"
R12D_DECISION = "SAR_CROSS_TTA_SAFETY_TRANSFER_NOT_EVALUABLE_ZERO_HARM_DEGENERACY"

OUTPUT_DIR = OUT/"Q1_R13A_confident_pseudolabel_three_family_feasibility_fix1_v1"

ACTION = "A4_PL_CONF90_1STEP"
PROBE_INDICES = (0, 142, 285, 428, 570, 713, 856, 999)

CASES = 1000
STATES = 9
PACKED_BYTES = 15488

CONFIDENCE_THRESHOLD = 0.90
LR = 1e-3
WEIGHT_DECAY = 0.0
STEPS = 1
EPISODIC_RESET = True

PASS_DECISION = "PL_CONF90_THREE_FAMILY_TECHNICAL_NONDEGENERACY_PASS"
STOP_DECISION = "PL_CONF90_THREE_FAMILY_TECHNICAL_NONDEGENERACY_STOP"


def sha256_file(path: Path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, values):
    import torch
    with torch.no_grad():
        for p, v in zip(params, values):
            p.copy_(v)


def max_param_delta(params, values):
    vals = [
        float((p.detach()-v).abs().max().cpu())
        for p, v in zip(params, values)
    ]
    return max(vals) if vals else 0.0


def validate_lineage():
    for p in (R05D3, R12D_LOCK):
        if not p.exists():
            raise FileNotFoundError(p)

    if sha256_file(R05D3) != R05D3_SHA:
        raise RuntimeError("R05D3 SHA mismatch.")

    if sha256_file(R12D_LOCK) != R12D_SHA:
        raise RuntimeError("R12D STOP lock SHA mismatch.")

    r12d = json.loads(R12D_LOCK.read_text(encoding="utf-8"))
    if r12d.get("decision") != R12D_DECISION:
        raise RuntimeError("Unexpected R12D decision.")
    if bool(
        r12d.get("scientific_interpretation", {})
        .get("sar_branch_retuning_authorized", True)
    ):
        raise RuntimeError("R12D did not formally close SAR retuning.")

    return {
        "r05d3_script_sha256": sha256_file(R05D3),
        "r12d_stop_lock_sha256": sha256_file(R12D_LOCK),
    }


def source_pack(r05, state):
    d = R05D3_OUT/"state_predictions"
    pred, idx, lock = r05.state_paths(d, state)
    for p in (pred, idx, lock):
        if not p.exists():
            raise FileNotFoundError(p)
    side = json.loads(lock.read_text(encoding="utf-8"))
    if sha256_file(pred) != side["prediction_npz_sha256"]:
        raise RuntimeError("R05D3 source prediction SHA mismatch.")
    with np.load(pred, allow_pickle=False) as z:
        # SOURCE only. A1 values are intentionally never indexed.
        arr = np.asarray(z["source_masks_packed"], dtype=np.uint8).copy()
    if arr.shape != (CASES, PACKED_BYTES):
        raise RuntimeError(f"SOURCE pack shape={arr.shape}")
    return arr


def binary_pseudolabel(teacher_logit):
    import torch
    p = torch.sigmoid(teacher_logit.detach())
    hard = (p >= 0.5).to(dtype=teacher_logit.dtype)
    conf = torch.maximum(p, 1.0-p)
    mask = conf >= CONFIDENCE_THRESHOLD
    return hard, mask


def categorical_pseudolabel(teacher_logits):
    import torch
    p = torch.softmax(teacher_logits.detach(), dim=1)
    conf, hard = torch.max(p, dim=1)
    mask = conf >= CONFIDENCE_THRESHOLD
    return hard, mask


def binary_masked_loss(student_logit, hard, mask):
    import torch.nn.functional as F
    per = F.binary_cross_entropy_with_logits(
        student_logit,
        hard,
        reduction="none",
    )
    if int(mask.sum().detach().cpu()) == 0:
        return None
    return per[mask].mean()


def categorical_masked_loss(student_logits, hard, mask):
    import torch.nn.functional as F
    per = F.cross_entropy(
        student_logits,
        hard,
        reduction="none",
    )
    if int(mask.sum().detach().cpu()) == 0:
        return None
    return per[mask].mean()


def common_row(state, target, idx, parity, confident, total, loss, delta, changed, updated):
    return {
        "probe_index": idx,
        "sample_id": target["sample_id"],
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "source_parity": int(parity),
        "confident_pixels": int(confident),
        "total_pixels": int(total),
        "confident_fraction": float(confident/total),
        "pl_loss": float(loss) if loss is not None else math.nan,
        "parameter_max_abs_delta": float(delta),
        "updated": int(updated),
        "source_to_pl_changed_pixels": int(changed),
    }


def run_binary_family(r05, state, targets, device, spack):
    import torch

    family = state["model_family"]
    seed = int(state["training_seed"])
    set_seed(seed)

    if family == "PraNet":
        h = r05.import_module(r05.PRANET_HELPER, f"r13a_pranet_{seed}")
        tr = h.import_training_helper()
        model = h.load_model(tr, seed, device)
        params = h.configure_tent(model)

        def to_tensor(native):
            return h.image_to_model_tensor(tr, native).to(device)

        def source_mode():
            model.eval()

        def adapt_mode():
            model.train()

        def logits(x):
            return h.final_logit(model, x)

    elif family == "DeepLabV3-R50":
        h = r05.import_module(r05.DEEPLAB_HELPER, f"r13a_deeplab_{seed}")
        tr = h.import_training_helper()
        h.seed_everything(20260817)
        model = h.load_model(tr, seed, device)
        params, _, _, unsafe, _, drops = h.configure_singleton_safe_tent(model)

        def to_tensor(native):
            return h.image_to_model_tensor(tr, native).to(device)

        def source_mode():
            model.eval()

        def adapt_mode():
            h.set_singleton_safe_tent_mode(model, unsafe, drops)

        def logits(x):
            return tr.deeplab_logits(model, x)

    else:
        raise RuntimeError(f"Unexpected binary family={family}")

    source_values = snapshot(params)
    rows = []

    for idx in tqdm(
        PROBE_INDICES,
        desc=f"R13A {family} {seed}",
        unit="case",
        dynamic_ncols=True,
    ):
        target = targets[idx]
        with Image.open(target["image_path"]) as im:
            x = to_tensor(im.convert("RGB"))

        # Frozen teacher/source forward + SOURCE parity.
        restore(params, source_values)
        source_mode()
        with torch.no_grad():
            z_teacher = logits(x).detach()
        source_mask = r05.logit_to_mask(
            z_teacher[0,0].float().cpu().numpy()
        )
        ref = r05.unpack_mask(spack[idx])
        parity = bool(np.array_equal(source_mask, ref))
        if not parity:
            raise RuntimeError(
                f"SOURCE parity mismatch: {state['model_state_id']} idx={idx}"
            )

        hard, conf_mask = binary_pseudolabel(z_teacher)
        confident = int(conf_mask.sum().cpu())
        total = int(conf_mask.numel())

        # New objective; same TENT parameter panel / optimizer settings.
        restore(params, source_values)
        adapt_mode()
        optimizer = torch.optim.Adam(
            params,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        z_student = logits(x)
        loss = binary_masked_loss(
            z_student,
            hard,
            conf_mask,
        )

        updated = False
        loss_value = None
        if loss is not None:
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pseudo-label loss.")
            loss_value = float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
            updated = True

        source_mode()
        with torch.no_grad():
            z_final = logits(x).detach()
        final_mask = r05.logit_to_mask(
            z_final[0,0].float().cpu().numpy()
        )

        rows.append(
            common_row(
                state,
                target,
                idx,
                parity,
                confident,
                total,
                loss_value,
                max_param_delta(params, source_values),
                int(np.count_nonzero(source_mask != final_mask)),
                updated,
            )
        )

        restore(params, source_values)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_segformer(r05, r03, state, targets, device, spack, context=None):
    if context is None:
        context = r03.build_segformer_context(device)

    torch, nn, F, SF, cfg, mean, std = context
    seed = int(state["training_seed"])
    r05.set_runtime_seed(seed)

    model = r03.load_segformer_state(seed, device, SF, cfg, torch)
    params, _, bns, bntrack, drops, _ = r03.configure_segformer_tent(model, nn)
    source_values = snapshot(params)
    rows = []

    for idx in tqdm(
        PROBE_INDICES,
        desc=f"R13A SegFormer {seed}",
        unit="case",
        dynamic_ncols=True,
    ):
        target = targets[idx]
        with Image.open(target["image_path"]) as im:
            x = r03.segformer_tensor(
                im.convert("RGB"),
                mean,
                std,
                torch,
            ).to(device)

        restore(params, source_values)
        r03.segformer_source_mode(model, bns, bntrack, drops)
        with torch.no_grad():
            teacher_logits, z_teacher = r03.segformer_logits_and_z(model, x, F)

        source_mask = r05.logit_to_mask(
            z_teacher[0].float().cpu().numpy()
        )
        ref = r05.unpack_mask(spack[idx])
        parity = bool(np.array_equal(source_mask, ref))
        if not parity:
            raise RuntimeError(
                f"SOURCE parity mismatch: {state['model_state_id']} idx={idx}"
            )

        hard, conf_mask = categorical_pseudolabel(teacher_logits)
        confident = int(conf_mask.sum().cpu())
        total = int(conf_mask.numel())

        restore(params, source_values)
        r03.segformer_tent_mode(model, bns, drops)
        optimizer = torch.optim.Adam(
            params,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        student_logits, _ = r03.segformer_logits_and_z(model, x, F)
        loss = categorical_masked_loss(
            student_logits,
            hard,
            conf_mask,
        )

        updated = False
        loss_value = None
        if loss is not None:
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SegFormer pseudo-label loss.")
            loss_value = float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
            updated = True

        r03.segformer_source_mode(model, bns, bntrack, drops)
        with torch.no_grad():
            _, z_final = r03.segformer_logits_and_z(model, x, F)
        final_mask = r05.logit_to_mask(
            z_final[0].float().cpu().numpy()
        )

        rows.append(
            common_row(
                state,
                target,
                idx,
                parity,
                confident,
                total,
                loss_value,
                max_param_delta(params, source_values),
                int(np.count_nonzero(source_mask != final_mask)),
                updated,
            )
        )

        restore(params, source_values)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, context


def self_test():
    import torch

    z = torch.tensor([[[[-3.0, 0.0, 3.0]]]])
    hard, mask = binary_pseudolabel(z)
    assert hard.shape == z.shape
    assert mask.shape == z.shape
    assert int(mask.sum()) == 2

    logits = torch.tensor([[
        [[3.0, 0.0]],
        [[0.0, 3.0]],
    ]])
    hard2, mask2 = categorical_pseudolabel(logits)
    assert tuple(hard2.shape) == (1,1,2)
    assert int(mask2.sum()) == 2

    assert CONFIDENCE_THRESHOLD == 0.90
    assert LR == 1e-3
    assert WEIGHT_DECAY == 0.0
    assert len(PROBE_INDICES) == 8

    print("BINARY_PSEUDOLABEL_TEST_PASS")
    print("CATEGORICAL_PSEUDOLABEL_TEST_PASS")
    print("HYPERPARAMETER_FREEZE_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):
    print("===== R13A CONFIDENT PSEUDOLABEL THREE-FAMILY FEASIBILITY FIX1 =====")
    print("action=", ACTION)
    print("new distinct TTA branch=YES")
    print("GT/Dice/DeltaDice/HARM=NO")
    print("safety score read=NO")
    print("PolypGen access=NO")
    print("hyperparameter sweep=NO")
    print("confidence threshold=", CONFIDENCE_THRESHOLD)
    print("optimizer=Adam")
    print("lr=", LR)
    print("weight decay=", WEIGHT_DECAY)
    print("steps=", STEPS)
    print("episodic reset=YES")
    print("same-case post-update prediction=YES")
    print("technical non-degeneracy criterion=at least 1 changed-mask probe per state")
    print()

    provenance = validate_lineage()
    print("R12D SAR-stop exact lock gate=PASS")

    r05 = import_module(R05D3, "q1_r13a_r05d3")
    _, _, _, manifest_path, _ = r05.validate_upstream()
    targets, d2helper = r05.load_frozen_targets(manifest_path)
    panel, r03 = r05.build_frozen_panel(d2helper)

    if len(targets) != CASES or len(panel) != STATES:
        raise RuntimeError("Frozen target/state cardinality changed.")

    import torch
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable. Use --cpu only for debugging.")
    device = torch.device("cpu" if args.cpu else "cuda")
    print("device=", device)

    rows = []
    seg_context = None

    for state in panel:
        spack = source_pack(r05, state)

        if state["model_family"] in ("PraNet", "DeepLabV3-R50"):
            rows += run_binary_family(
                r05,
                state,
                targets,
                device,
                spack,
            )
        elif state["model_family"] == "SegFormer-B0":
            rr, seg_context = run_segformer(
                r05,
                r03,
                state,
                targets,
                device,
                spack,
                context=seg_context,
            )
            rows += rr
        else:
            raise RuntimeError(f"Unexpected family={state['model_family']}")

    df = pd.DataFrame(rows)
    if len(df) != STATES * len(PROBE_INDICES):
        raise RuntimeError(f"Probe rows={len(df)}")

    state_summary = (
        df.groupby(
            ["model_family","model_state_id","training_seed"],
            as_index=False,
        )
        .agg(
            probe_cases=("updated","size"),
            source_parity_mismatches=("source_parity",lambda x:int((x==0).sum())),
            updated_cases=("updated","sum"),
            confident_fraction_mean=("confident_fraction","mean"),
            confident_fraction_min=("confident_fraction","min"),
            parameter_delta_max=("parameter_max_abs_delta","max"),
            changed_case_count=("source_to_pl_changed_pixels",lambda x:int((x>0).sum())),
            changed_pixels_mean=("source_to_pl_changed_pixels","mean"),
            changed_pixels_max=("source_to_pl_changed_pixels","max"),
        )
    )

    source_ok = bool((df["source_parity"] == 1).all())
    update_ok = bool(
        (state_summary["updated_cases"] >= 1).all()
        and (state_summary["parameter_delta_max"] > 0).all()
    )
    nondegenerate = bool(
        (state_summary["changed_case_count"] >= 1).all()
    )
    confident_ok = bool(
        (state_summary["confident_fraction_mean"] > 0).all()
    )

    passed = source_ok and update_ok and nondegenerate and confident_ok
    decision = PASS_DECISION if passed else STOP_DECISION

    print("\n===== STATE SUMMARY =====")
    print(state_summary.to_string(index=False))

    family_summary = (
        df.groupby("model_family", as_index=False)
        .agg(
            probe_rows=("updated","size"),
            updated_rows=("updated","sum"),
            confident_fraction_mean=("confident_fraction","mean"),
            changed_case_count=("source_to_pl_changed_pixels",lambda x:int((x>0).sum())),
            changed_pixels_mean=("source_to_pl_changed_pixels","mean"),
            parameter_delta_max=("parameter_max_abs_delta","max"),
        )
    )
    print("\n===== FAMILY SUMMARY =====")
    print(family_summary.to_string(index=False))

    print("\n===== FINAL DECISION =====")
    print("SOURCE parity all pass=", source_ok)
    print("every state updated=", update_ok)
    print("every state has changed-mask probe=", nondegenerate)
    print("every state has confident pixels=", confident_ok)
    print("Decision=", decision)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    p1 = args.output_dir/"R13A_probe_detail.csv"
    p2 = args.output_dir/"R13A_state_summary.csv"
    p3 = args.output_dir/"R13A_family_summary.csv"

    df.to_csv(p1, index=False)
    state_summary.to_csv(p2, index=False)
    family_summary.to_csv(p3, index=False)

    lock = {
        "status": "COMPLETE",
        "decision": decision,
        "action": ACTION,
        "probe_indices": list(PROBE_INDICES),
        "action_semantics": {
            "teacher": "frozen SOURCE-mode prediction on same image",
            "pseudo_label": "hard source prediction",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "binary_confidence": "max(sigmoid(z),1-sigmoid(z))",
            "categorical_confidence": "max softmax probability",
            "loss": "masked hard-pseudolabel cross-entropy",
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "steps": STEPS,
            "trainable_parameter_panel": "exact R05D3 TENT panel by family",
            "episodic_source_reset": EPISODIC_RESET,
            "same_case_post_update_prediction": True,
        },
        "checks": {
            "source_parity_all_pass": source_ok,
            "every_state_updated": update_ok,
            "every_state_has_changed_mask_probe": nondegenerate,
            "every_state_has_confident_pixels": confident_ok,
        },
        "information_boundary": {
            "target_rgb_used": True,
            "target_gt_used": False,
            "dice_computed": False,
            "delta_dice_computed": False,
            "harm_label_computed": False,
            "safety_scores_read": False,
            "polypgen_access": False,
            "hyperparameter_sweep": False,
            "target_outcome_based_selection": False,
        },
        "provenance": provenance,
        "artifacts": {
            p1.name: sha256_file(p1),
            p2.name: sha256_file(p2),
            p3.name: sha256_file(p3),
        },
        "next_stage": (
            "R13B_FULL_NEOPOLYP_PL_CONF90_PREDICTION_LOCK"
            if passed else
            "STOP_PL_CONF90_BRANCH_NO_RETUNING"
        ),
    }

    lp = args.output_dir/"R13A_PL_CONF90_FEASIBILITY_LOCK.json"
    lp.write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding="utf-8")

    print("R13A LOCK:", lp)
    print("R13A LOCK SHA256:", sha256_file(lp))
    print("PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    run(args)


if __name__ == "__main__":
    main()
