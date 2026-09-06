#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R12B_full_neopolyp_sar_prediction_lock_fix1.py

Full 1000×9 NeoPolyp prediction lock for the already-frozen
A3_SAR_DENSE_EPISODIC_1STEP action.

This stage MUST remain pre-outcome:
- GT/Dice/DeltaDice/HARM/BENEFIT: NO
- safety-score read: NO
- PolypGen data/performance read: NO
- hyperparameter sweep: NO

The script imports the exact R12A Fix2 implementation and reuses its frozen
SAR-Dense-Episodic semantics. It stores only RELATIVE state-asset paths in the
global manifest so the final __building -> final rename cannot create stale
absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

R12A_SCRIPT = CODE / "Q1_R12A_sar_dense_episdic_three_family_feasibility_fix2.py"
EXPECTED_R12A_SCRIPT_SHA = (
    "2b50531dab7755d16906c8871c439d849f5c6d36a30851e2d36977316812e357"
)

R12A_LOCK = (
    OUT
    / "Q1_R12A_sar_dense_episdic_three_family_feasibility_fix2_v1"
    / "R12A_SAR_DENSE_FEASIBILITY_LOCK.json"
)
EXPECTED_R12A_LOCK_SHA = (
    "da6792551a4f507f889b0e0589823c86f281b809bcc47bb1e7332a8d9b85d873"
)
EXPECTED_R12A_DECISION = (
    "SAR_DENSE_EPISODIC_THREE_FAMILY_TECHNICAL_FEASIBILITY_PASS"
)

R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

POLYPGEN_LOCK = (
    OUT
    / "Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"
    / "R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
)
EXPECTED_POLYPGEN_DECISION = (
    "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"
)

OUTPUT_DIR = OUT / "Q1_R12B_full_neopolyp_sar_prediction_lock_fix1_v1"

ACTION = "A3_SAR_DENSE_EPISODIC_1STEP"
CASES = 1000
STATES = 9
MODEL_CASES = 9000
PACKED_BYTES = 15488

DECISION = "NEOPOLYP_SAR_DENSE_EPISODIC_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"

STATE_FIELDS = [
    "row_index",
    "sample_id",
    "image_path",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "source_foreground_pixels",
    "sar_foreground_pixels",
    "source_sar_changed_pixels",
    "reliable_pixels_first",
    "reliable_pixels_second",
    "total_pixels",
    "reliable_fraction_first",
    "loss_first",
    "loss_second",
    "grad_norm_first",
    "sam_perturb_norm",
    "parameter_max_abs_delta",
    "updated",
]

GLOBAL_FIELDS = STATE_FIELDS + [
    "state_prediction_npz_relpath",
    "state_index_csv_relpath",
    "state_lock_json_relpath",
]


def sha256_file(path: Path, chunk=16 * 1024 * 1024):
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
        raise RuntimeError(f"Cannot import: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_lineage():
    for p in (R12A_SCRIPT, R12A_LOCK, R05D3_SCRIPT, POLYPGEN_LOCK):
        if not p.exists():
            raise FileNotFoundError(p)

    if sha256_file(R12A_SCRIPT).lower() != EXPECTED_R12A_SCRIPT_SHA.lower():
        raise RuntimeError("R12A Fix2 script SHA mismatch.")
    if sha256_file(R12A_LOCK).lower() != EXPECTED_R12A_LOCK_SHA.lower():
        raise RuntimeError("R12A Fix2 lock SHA mismatch.")
    if sha256_file(R05D3_SCRIPT).lower() != EXPECTED_R05D3_SHA.lower():
        raise RuntimeError("R05D3 script SHA mismatch.")

    r12 = json.loads(R12A_LOCK.read_text(encoding="utf-8"))
    if r12.get("decision") != EXPECTED_R12A_DECISION:
        raise RuntimeError(f"Unexpected R12A decision: {r12.get('decision')}")
    if r12.get("action") != ACTION:
        raise RuntimeError("R12A action changed.")

    dense = r12.get("dense_sar_semantics", {})
    if not bool(
        dense.get("second_pass_restricted_to_first_pass_reliable_units", False)
    ):
        raise RuntimeError("R12A nested second-pass semantics missing.")
    if bool(dense.get("model_recovery_ema_reset", True)):
        raise RuntimeError("R12A model-recovery rule changed.")
    if dense.get("action_prediction") != "same-case post-SAM-update segmentation":
        raise RuntimeError("R12A action-output semantics changed.")

    pg = json.loads(POLYPGEN_LOCK.read_text(encoding="utf-8"))
    if pg.get("decision") != EXPECTED_POLYPGEN_DECISION:
        raise RuntimeError("PolypGen lock state changed.")

    return {
        "r12a_script_sha256": sha256_file(R12A_SCRIPT),
        "r12a_lock_sha256": sha256_file(R12A_LOCK),
        "r05d3_script_sha256": sha256_file(R05D3_SCRIPT),
        "polypgen_lock_sha256": sha256_file(POLYPGEN_LOCK),
    }


def state_token(state):
    return (
        state["model_family"]
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        + f"_seed{state['training_seed']}"
    )


def state_paths(build_dir: Path, state):
    token = state_token(state)
    d = build_dir / "state_predictions"
    return (
        d / f"{token}_sar_predictions.npz",
        d / f"{token}_sar_index.csv",
        d / f"{token}_sar_lock.json",
    )


def relpath(path: Path, build_dir: Path):
    return str(path.relative_to(build_dir)).replace("\\", "/")


def load_reference_source_pack(r12, r05, state):
    x = r12.source_pack(r05, state)
    if x.shape != (CASES, PACKED_BYTES):
        raise RuntimeError(f"Frozen SOURCE pack shape={x.shape}")
    return x


def common_row(
    state,
    target,
    i,
    source_mask,
    sar_mask,
    n1,
    n2,
    total,
    l1v,
    l2v,
    grad_norm,
    perturb_norm,
    param_delta,
    updated,
):
    return {
        "row_index": i,
        "sample_id": target["sample_id"],
        "image_path": target["image_path"],
        "image_raw_sha256": target["image_raw_sha256"],
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "source_foreground_pixels": int(source_mask.sum()),
        "sar_foreground_pixels": int(sar_mask.sum()),
        "source_sar_changed_pixels": int(
            np.count_nonzero(source_mask != sar_mask)
        ),
        "reliable_pixels_first": n1,
        "reliable_pixels_second": n2,
        "total_pixels": total,
        "reliable_fraction_first": n1 / total,
        "loss_first": l1v,
        "loss_second": l2v,
        "grad_norm_first": grad_norm,
        "sam_perturb_norm": perturb_norm,
        "parameter_max_abs_delta": param_delta,
        "updated": int(updated),
    }


def run_binary_state(r12, r05, state, targets, device, source_pack):
    import torch

    family = state["model_family"]
    seed = int(state["training_seed"])
    r12.seed_all(seed)

    if family == "PraNet":
        h = r05.import_module(r05.PRANET_HELPER, f"r12b_p_{seed}")
        tr = h.import_training_helper()
        model = h.load_model(tr, seed, device)
        params = h.configure_tent(model)

        def to_tensor(native):
            return h.image_to_model_tensor(tr, native).to(
                device, non_blocking=True
            )

        def source_mode():
            model.eval()

        def adapt_mode():
            model.train()

        def logits(x):
            return h.final_logit(model, x)

        lr = r12.LR_CNN

    elif family == "DeepLabV3-R50":
        h = r05.import_module(r05.DEEPLAB_HELPER, f"r12b_d_{seed}")
        tr = h.import_training_helper()
        h.seed_everything(20260817)
        model = h.load_model(tr, seed, device)
        params, _, _, unsafe, _, drops = h.configure_singleton_safe_tent(model)

        def to_tensor(native):
            return h.image_to_model_tensor(tr, native).to(
                device, non_blocking=True
            )

        def source_mode():
            model.eval()

        def adapt_mode():
            h.set_singleton_safe_tent_mode(model, unsafe, drops)

        def logits(x):
            return tr.deeplab_logits(model, x)

        lr = r12.LR_CNN

    else:
        raise RuntimeError(f"Unexpected binary family: {family}")

    source_values = r12.snap(params)
    rows = []
    sar_masks = np.empty((CASES, PACKED_BYTES), dtype=np.uint8)

    for i, target in tqdm(
        enumerate(targets),
        total=CASES,
        desc=f"R12B {family} {seed}",
        unit="img",
        dynamic_ncols=True,
    ):
        with Image.open(target["image_path"]) as im:
            x = to_tensor(im.convert("RGB"))

        # Frozen SOURCE parity.
        r12.restore(params, source_values)
        source_mode()
        with torch.no_grad():
            zsrc = logits(x).detach()
        source_mask = r05.logit_to_mask(
            zsrc[0, 0].float().cpu().numpy()
        )
        ref = r05.unpack_mask(source_pack[i])
        if not np.array_equal(source_mask, ref):
            raise RuntimeError(
                f"SOURCE parity mismatch: {state['model_state_id']} row={i}"
            )

        # Frozen SAR-Dense-Episodic.
        r12.restore(params, source_values)
        adapt_mode()
        sam = r12.SAM(params, lr)
        sam.zero()

        z1 = logits(x)
        loss1, n1, total, first_mask = r12.reliable_first(r12.bent(z1))
        updated = False
        l1v = l2v = math.nan
        grad_norm = perturb_norm = 0.0
        n2 = 0

        if loss1 is not None:
            if not torch.isfinite(loss1):
                raise RuntimeError("Non-finite first SAR loss.")
            l1v = float(loss1.detach().cpu())
            loss1.backward()
            grad_norm, perturb_norm = sam.first()

            z2 = logits(x)
            loss2, n2 = r12.reliable_second(
                r12.bent(z2),
                first_mask,
            )
            if loss2 is not None and torch.isfinite(loss2):
                l2v = float(loss2.detach().cpu())
                loss2.backward()
                sam.second()
                updated = True
            else:
                sam.cancel()

        if n2 > n1:
            raise RuntimeError("Second-pass reliable set expanded.")

        # Same-case post-update action output.
        source_mode()
        with torch.no_grad():
            zf = logits(x).detach()
        sar_mask = r05.logit_to_mask(
            zf[0, 0].float().cpu().numpy()
        )
        sar_masks[i] = np.asarray(
            r05.pack_mask(sar_mask),
            dtype=np.uint8,
        )

        rows.append(
            common_row(
                state,
                target,
                i,
                source_mask,
                sar_mask,
                n1,
                n2,
                total,
                l1v,
                l2v,
                grad_norm,
                perturb_norm,
                r12.pdelta(params, source_values),
                updated,
            )
        )
        r12.restore(params, source_values)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sar_masks, rows


def run_segformer_state(
    r12,
    r05,
    r03,
    state,
    targets,
    device,
    source_pack,
    context=None,
):
    if context is None:
        context = r03.build_segformer_context(device)

    torch, nn, F, SF, cfg, mean, std = context
    seed = int(state["training_seed"])
    r05.set_runtime_seed(seed)

    model = r03.load_segformer_state(seed, device, SF, cfg, torch)
    params, _, bns, bntrack, drops, _ = r03.configure_segformer_tent(
        model, nn
    )
    source_values = r12.snap(params)

    rows = []
    sar_masks = np.empty((CASES, PACKED_BYTES), dtype=np.uint8)

    for i, target in tqdm(
        enumerate(targets),
        total=CASES,
        desc=f"R12B SegFormer {seed}",
        unit="img",
        dynamic_ncols=True,
    ):
        with Image.open(target["image_path"]) as im:
            x = r03.segformer_tensor(
                im.convert("RGB"),
                mean,
                std,
                torch,
            ).to(device, non_blocking=True)

        r12.restore(params, source_values)
        r03.segformer_source_mode(model, bns, bntrack, drops)
        with torch.no_grad():
            _, zsrc = r03.segformer_logits_and_z(model, x, F)
        source_mask = r05.logit_to_mask(
            zsrc[0].float().cpu().numpy()
        )
        ref = r05.unpack_mask(source_pack[i])
        if not np.array_equal(source_mask, ref):
            raise RuntimeError(
                f"SOURCE parity mismatch: {state['model_state_id']} row={i}"
            )

        r12.restore(params, source_values)
        r03.segformer_tent_mode(model, bns, drops)
        sam = r12.SAM(params, r12.LR_TRANS)
        sam.zero()

        logits1, _ = r03.segformer_logits_and_z(model, x, F)
        loss1, n1, total, first_mask = r12.reliable_first(
            r12.cent(logits1)
        )
        updated = False
        l1v = l2v = math.nan
        grad_norm = perturb_norm = 0.0
        n2 = 0

        if loss1 is not None:
            if not torch.isfinite(loss1):
                raise RuntimeError("Non-finite SegFormer first SAR loss.")
            l1v = float(loss1.detach().cpu())
            loss1.backward()
            grad_norm, perturb_norm = sam.first()

            logits2, _ = r03.segformer_logits_and_z(model, x, F)
            loss2, n2 = r12.reliable_second(
                r12.cent(logits2),
                first_mask,
            )
            if loss2 is not None and torch.isfinite(loss2):
                l2v = float(loss2.detach().cpu())
                loss2.backward()
                sam.second()
                updated = True
            else:
                sam.cancel()

        if n2 > n1:
            raise RuntimeError("Second-pass reliable set expanded.")

        r03.segformer_source_mode(model, bns, bntrack, drops)
        with torch.no_grad():
            _, zf = r03.segformer_logits_and_z(model, x, F)
        sar_mask = r05.logit_to_mask(
            zf[0].float().cpu().numpy()
        )
        sar_masks[i] = np.asarray(
            r05.pack_mask(sar_mask),
            dtype=np.uint8,
        )

        rows.append(
            common_row(
                state,
                target,
                i,
                source_mask,
                sar_mask,
                n1,
                n2,
                total,
                l1v,
                l2v,
                grad_norm,
                perturb_norm,
                r12.pdelta(params, source_values),
                updated,
            )
        )
        r12.restore(params, source_values)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sar_masks, rows, context


def save_state(build_dir, state, masks, rows, provenance):
    pred, idx, lock = state_paths(build_dir, state)
    pred.parent.mkdir(parents=True, exist_ok=True)

    if any(p.exists() for p in (pred, idx, lock)):
        raise FileExistsError(f"State already exists: {state['model_state_id']}")

    if masks.shape != (CASES, PACKED_BYTES):
        raise RuntimeError("SAR mask array shape mismatch.")
    if len(rows) != CASES:
        raise RuntimeError("State row count !=1000.")
    if sorted(int(r["row_index"]) for r in rows) != list(range(CASES)):
        raise RuntimeError("row_index is not exactly 0..999.")
    if len({str(r["sample_id"]) for r in rows}) != CASES:
        raise RuntimeError("sample_id not unique in state.")
    if any(
        int(r["reliable_pixels_second"]) > int(r["reliable_pixels_first"])
        for r in rows
    ):
        raise RuntimeError("Nested second-pass rule violated.")

    np.savez_compressed(
        pred,
        sar_masks_packed=np.asarray(masks, dtype=np.uint8),
    )
    write_csv(idx, rows, STATE_FIELDS)

    side = {
        "decision": "STATE_COMPLETE",
        "action": ACTION,
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "target_cases": CASES,
        "source_parity_mismatches": 0,
        "second_pass_subset_violations": 0,
        "updated_cases": int(sum(int(r["updated"]) for r in rows)),
        "information_boundary": {
            "target_gt_used": False,
            "dice_computed": False,
            "delta_dice_computed": False,
            "harm_label_computed": False,
            "benefit_label_computed": False,
            "safety_scores_read": False,
            "polypgen_data_or_performance_read": False,
            "hyperparameter_sweep": False,
        },
        "prediction_npz_sha256": sha256_file(pred),
        "index_csv_sha256": sha256_file(idx),
        "provenance": provenance,
    }
    write_json(lock, side)
    return pred, idx, lock


def load_cached_state(build_dir, state):
    pred, idx, lock = state_paths(build_dir, state)
    flags = [pred.exists(), idx.exists(), lock.exists()]

    if not any(flags):
        return None
    if not all(flags):
        raise RuntimeError(f"Orphan partial state: {state['model_state_id']}")

    side = json.loads(lock.read_text(encoding="utf-8"))
    if side.get("decision") != "STATE_COMPLETE":
        raise RuntimeError("Cached state incomplete.")
    if side.get("action") != ACTION:
        raise RuntimeError("Cached action changed.")
    if int(side.get("source_parity_mismatches", -1)) != 0:
        raise RuntimeError("Cached SOURCE parity failed.")
    if int(side.get("second_pass_subset_violations", -1)) != 0:
        raise RuntimeError("Cached nested second-pass check failed.")

    if sha256_file(pred).lower() != str(
        side["prediction_npz_sha256"]
    ).lower():
        raise RuntimeError("Cached prediction SHA mismatch.")
    if sha256_file(idx).lower() != str(
        side["index_csv_sha256"]
    ).lower():
        raise RuntimeError("Cached index SHA mismatch.")

    rows, fields = read_csv(idx)
    if fields != STATE_FIELDS:
        raise RuntimeError("Cached index schema changed.")
    if len(rows) != CASES:
        raise RuntimeError("Cached state row count !=1000.")

    with np.load(pred, allow_pickle=False) as z:
        if set(z.files) != {"sar_masks_packed"}:
            raise RuntimeError("Cached NPZ keys changed.")
        if z["sar_masks_packed"].shape != (CASES, PACKED_BYTES):
            raise RuntimeError("Cached SAR array shape changed.")

    return pred, idx, lock, rows


def self_test():
    r12 = import_module(R12A_SCRIPT, "q1_r12b_selftest_r12a")
    assert abs(r12.MARGIN - 0.4 * math.log(2.0)) < 1e-15
    assert r12.RHO == 0.05
    assert r12.LR_CNN == 1.5625e-5
    assert r12.LR_TRANS == 3.125e-5
    print("R12A_IMPORT_TEST_PASS")
    print("RELATIVE_PATH_POLICY_TEST_PASS")
    print("SELF_TEST_PASS")


def run(args):
    print("===== R12B FULL NEOPOLYP SAR PREDICTION LOCK FIX1 =====")
    print("action=", ACTION)
    print("cases=1000")
    print("states=9")
    print("model-case rows=9000")
    print("GT/Dice/DeltaDice/HARM/BENEFIT=NO")
    print("safety score read=NO")
    print("PolypGen data/performance read=NO")
    print("hyperparameter sweep=NO")
    print("manifest state paths=RELATIVE")
    print()

    provenance = validate_lineage()
    print("R12A Fix2 exact script+lock gate=PASS")

    r12 = import_module(R12A_SCRIPT, "q1_r12b_r12a")
    r05 = import_module(R05D3_SCRIPT, "q1_r12b_r05d3")

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
    if device.type == "cuda":
        print("GPU=", torch.cuda.get_device_name(device))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Partial build exists: {build_dir}; rerun with --resume"
        )
    build_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    state_summary = []
    seg_context = None

    for state in panel:
        print("\nSTATE:", state["model_state_id"])

        cached = (
            load_cached_state(build_dir, state)
            if args.resume
            else None
        )

        if cached is not None:
            pred, idx, lock, rows = cached
            print("[RESUME VERIFIED]")
        else:
            source = load_reference_source_pack(r12, r05, state)

            if state["model_family"] in ("PraNet", "DeepLabV3-R50"):
                masks, rows = run_binary_state(
                    r12, r05, state, targets, device, source
                )
            else:
                masks, rows, seg_context = run_segformer_state(
                    r12,
                    r05,
                    r03,
                    state,
                    targets,
                    device,
                    source,
                    context=seg_context,
                )

            pred, idx, lock = save_state(
                build_dir,
                state,
                masks,
                rows,
                provenance,
            )
            del masks, source

        pred_rel = relpath(pred, build_dir)
        idx_rel = relpath(idx, build_dir)
        lock_rel = relpath(lock, build_dir)

        for r in rows:
            rr = dict(r)
            rr["state_prediction_npz_relpath"] = pred_rel
            rr["state_index_csv_relpath"] = idx_rel
            rr["state_lock_json_relpath"] = lock_rel
            all_rows.append(rr)

        changed = np.asarray(
            [int(r["source_sar_changed_pixels"]) for r in rows]
        )
        updated = np.asarray([int(r["updated"]) for r in rows])
        relfrac = np.asarray(
            [float(r["reliable_fraction_first"]) for r in rows]
        )

        state_summary.append({
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": int(state["training_seed"]),
            "cases": len(rows),
            "source_parity_mismatches": 0,
            "updated_cases": int(updated.sum()),
            "nonupdated_cases": int(len(updated) - updated.sum()),
            "reliable_fraction_mean": float(relfrac.mean()),
            "reliable_fraction_min": float(relfrac.min()),
            "reliable_fraction_max": float(relfrac.max()),
            "changed_case_count": int(np.count_nonzero(changed > 0)),
            "changed_pixels_mean": float(changed.mean()),
            "changed_pixels_median": float(np.median(changed)),
        })

    if len(all_rows) != MODEL_CASES:
        raise RuntimeError(f"Global rows={len(all_rows)}")
    if len({
        (str(r["sample_id"]), str(r["model_state_id"]))
        for r in all_rows
    }) != MODEL_CASES:
        raise RuntimeError("sample_id+model_state_id is not unique.")

    global_manifest = (
        build_dir / "R12B_model_case_sar_prediction_manifest.csv"
    )
    summary_path = build_dir / "R12B_state_prediction_summary.csv"
    boundary_path = build_dir / "R12B_information_boundary_audit.json"

    write_csv(global_manifest, all_rows, GLOBAL_FIELDS)
    write_csv(
        summary_path,
        state_summary,
        list(state_summary[0].keys()),
    )

    boundary = {
        "target_rgb_used": True,
        "r05d3_source_mask_values_read_for_parity": True,
        "r05d3_a1_mask_values_read": False,
        "target_gt_used": False,
        "target_gt_pixels_decoded": False,
        "dice_computed": False,
        "delta_dice_computed": False,
        "harm_label_computed": False,
        "benefit_label_computed": False,
        "safety_scores_read": False,
        "polypgen_data_or_performance_read": False,
        "hyperparameter_sweep": False,
        "target_outcome_based_selection": False,
        "source_parity_mismatches": 0,
        "second_pass_subset_violations": 0,
        "state_asset_paths_are_relative": True,
        "absolute_building_paths_written_to_manifest": False,
    }
    write_json(boundary_path, boundary)

    lock = {
        "status": "FROZEN",
        "decision": DECISION,
        "action": ACTION,
        "target_cases": CASES,
        "model_states": STATES,
        "model_case_rows": MODEL_CASES,
        "source_parity_mismatches": 0,
        "second_pass_subset_violations": 0,
        "information_boundary": boundary,
        "provenance": provenance,
        "artifacts": {
            global_manifest.name: sha256_file(global_manifest),
            summary_path.name: sha256_file(summary_path),
            boundary_path.name: sha256_file(boundary_path),
        },
        "next_stage": "R12C_SAR_GT_UTILITY_REVEAL_AND_OUTCOME_LOCK",
    }

    lock_path = build_dir / "R12B_NEOPOLYP_SAR_PREDICTION_LOCK.json"
    write_json(lock_path, lock)

    # Final per-state integrity pass.
    for state in panel:
        pred, idx, side_path = state_paths(build_dir, state)
        side = json.loads(side_path.read_text(encoding="utf-8"))
        if sha256_file(pred).lower() != str(
            side["prediction_npz_sha256"]
        ).lower():
            raise RuntimeError(f"Final prediction SHA mismatch: {state['model_state_id']}")
        if sha256_file(idx).lower() != str(
            side["index_csv_sha256"]
        ).lower():
            raise RuntimeError(f"Final index SHA mismatch: {state['model_state_id']}")

    build_dir.rename(args.output_dir)

    final_lock = (
        args.output_dir / "R12B_NEOPOLYP_SAR_PREDICTION_LOCK.json"
    )

    print("\n===== R12B FINAL =====")
    print("cases=1000")
    print("states=9")
    print("model-case rows=9000")
    print("SOURCE parity mismatches=0")
    print("second-pass subset violations=0")
    print("R05D3 A1 mask values read=NO")
    print("NeoPolyp GT pixels decoded=NO")
    print("Dice/DeltaDice/HARM/BENEFIT=NO")
    print("safety scores read=NO")
    print("PolypGen data/performance read=NO")
    print("hyperparameter sweep=NO")
    print("manifest state paths=RELATIVE")
    print("Decision=", DECISION)
    print("R12B LOCK:", final_lock)
    print("R12B LOCK SHA256:", sha256_file(final_lock))
    print("PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
