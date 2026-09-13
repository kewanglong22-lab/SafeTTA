#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R31B1
MEMO-SEG4-1STEP 200-case SOURCE Design + Learning-Rate Freeze

Upstream
--------
R31B0 has already frozen:
  * third action = MEMO-SEG4-1STEP
  * 1000 NeoPolyp SOURCE cases -> 200 action-design / 800 LOAO evaluation
  * augmentations = identity / HFlip / VFlip / HVFlip
  * all model parameters
  * exactly one Adam update
  * LR candidates = [1e-5, 3e-5, 1e-4, 3e-4]
  * selection: highest mean deployed Dice; then lower HARM rate; then lower LR
  * external data forbidden

R31B1 does ONLY the 200-case design experiment and freezes one LR.
The 800-case R31B evaluation partition is NEVER opened for GT or model
inference in this script.

Runtime execution contract (frozen before first outcome)
--------------------------------------------------------
The R31B0 conceptual protocol did not spell out module train/eval behavior.
R31B1 therefore writes and SHA-locks the following execution contract BEFORE
opening any design GT or producing any MEMO outcome:

  * four spatial transforms are applied to the already-preprocessed 352x352
    tensor as one 4-image batch;
  * transforms are exactly invertible and require no interpolation:
      identity, horizontal flip, vertical flip, horizontal+vertical flip;
  * before every physical case x candidate-LR transaction, ALL model
    parameters and ALL registered buffers are restored to the frozen SOURCE
    state;
  * all model parameters are requires_grad=True;
  * MEMO adaptation forward runs under model.train(); therefore normalization
    and dropout modules follow their native training-mode semantics;
  * the four aligned foreground probabilities are inverse-transformed to the
    original coordinate frame and averaged pixelwise;
  * loss = mean binary entropy of this pixelwise marginal probability;
  * optimizer = Adam(all parameters), weight_decay=0, exactly one step;
  * final deployed prediction is produced on the original image under
    model.eval();
  * after each candidate transaction the SOURCE state is restored;
  * a deterministic per-(model_state_id, sample_id) action seed is reused
    across all four LR candidates, so stochastic training-mode behavior does
    not advantage one LR.

This is a segmentation-compatible MEMO instantiation. It does NOT claim
bit-exact reproduction of the original ImageNet MEMO code.

Information boundary
--------------------
NeoPolyp 200-case action-design GT : YES
NeoPolyp 800-case LOAO evaluation GT: NO
EndoTect                           : NO
PolypGen / SUN-SEG / PROMISE12     : NO
Third-action hyperparameter search : ONLY the 4 prelocked LR candidates
Model architecture/checkpoints     : frozen
TENT1/PL-CONF90 outcomes           : not used for LR selection

Run
---
python Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py

Resume
------
python Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


VERSION = "2026-09-12-R31B1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

R31B0_DIR = ROOT / "R31B0_memo_seg4_third_action_protocol_lock_v1"
R31B0_PROTOCOL = R31B0_DIR / "R31B0_MEMO_SEG4_PROTOCOL_LOCK.json"
EXPECTED_R31B0_PROTOCOL_SHA256 = (
    "9489a06e6fa87dcc9f3d5e57bcabb9e2217b51a59ab7f7d875e43fdd1ef5c5d0"
)
R31B0_FINAL = R31B0_DIR / "R31B0_FINAL_LOCK.json"
EXPECTED_R31B0_FINAL_SHA256 = (
    "03e16b4064611a927946b577d8a0740353e9037bd4eca72426d441c709010951"
)
R31B0_PARTITION = R31B0_DIR / "R31B0_SOURCE_CASE_PARTITION.csv"

R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)
R03_SCRIPT = CODE / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py"
EXPECTED_R03_SHA256 = (
    "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"
)
R05D3_CHECKPOINT_MANIFEST = (
    OUTPUTS
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
    / "checkpoint_manifest.csv"
)

DEFAULT_OUT = ROOT / "R31B1_memo_seg4_action_design_and_lr_freeze_v1"

DESIGN_PARTITION = "MEMO_ACTION_DESIGN"
EVAL_PARTITION = "R31B_LOAO_EVALUATION"

EXPECTED_SOURCE_CASES = 1000
EXPECTED_DESIGN_CASES = 200
EXPECTED_EVAL_CASES = 800
EXPECTED_STATES = 9
IMAGE_SIZE = 352

THIRD_ACTION = "MEMO-SEG4-1STEP"
LR_CANDIDATES = (1e-5, 3e-5, 1e-4, 3e-4)
WEIGHT_DECAY = 0.0
UPDATE_STEPS = 1
HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = 0.02

TRANSFORMS = ("IDENTITY", "HFLIP", "VFLIP", "HVFLIP")

EXPECTED_CHECKPOINT_SHAS = {
    "PraNet": {
        20260817: "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
        20260818: "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
        20260819: "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
    },
    "DeepLabV3-R50": {
        20260817: "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
        20260818: "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
        20260819: "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    },
    "SegFormer-B0": {
        20260820: "9dae2ae907b193ea36c2bccc8e376ddbad1699ba27ae0769cf40bcba13e95605",
        20260821: "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d",
        20260822: "ea0b3881373e9f966475a082490fabe4b0acae81179b8596c48a06ee2661a34a",
    },
}

FORBIDDEN_EXTERNAL_TOKENS = (
    "endotect",
    "polypgen",
    "sun-seg",
    "sunseg",
    "promise12",
)

EXECUTION_CONTRACT = {
    "action": THIRD_ACTION,
    "augmentation_space": list(TRANSFORMS),
    "augmentation_application": (
        "spatial flips on already-preprocessed 1xCx352x352 tensor; "
        "concatenated to one 4-image batch"
    ),
    "alignment": (
        "inverse-transform each 352x352 foreground probability map to original "
        "coordinates before averaging"
    ),
    "marginal": "pixelwise arithmetic mean foreground probability",
    "loss": "mean binary entropy of pixelwise marginal foreground probability",
    "adaptation_mode": "model.train()",
    "parameter_scope": "all model.parameters(); requires_grad=True",
    "buffer_semantics": (
        "native training-mode buffer behavior during the one adaptation forward; "
        "all registered buffers restored before every case x LR transaction"
    ),
    "dropout_semantics": "native model.train() behavior during adaptation",
    "optimizer": "torch.optim.Adam",
    "weight_decay": WEIGHT_DECAY,
    "steps": UPDATE_STEPS,
    "final_deployed_mode": "model.eval() on original unaugmented input",
    "episodic_reset": "all parameters and buffers restored to SOURCE",
    "stochastic_fairness": (
        "same deterministic per-(model_state_id,sample_id) seed reused for "
        "all LR candidates"
    ),
    "lr_candidates": list(LR_CANDIDATES),
    "lr_selection": [
        "highest mean deployed Dice across 200 design cases x 9 states",
        "lower HARM rate where HARM = delta_dice <= -0.02",
        "lower learning rate",
    ],
    "design_cases": EXPECTED_DESIGN_CASES,
    "loao_evaluation_cases_forbidden_for_selection": EXPECTED_EVAL_CASES,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\nexpected={expected}\nobserved={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, obj: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def reject_external_path(path: Path):
    low = str(path).lower()
    if any(tok in low for tok in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"External path forbidden in R31B1: {path}")


def canonical_family(v: Any) -> str:
    s = str(v).strip().lower().replace("_", "-")
    if "pranet" in s:
        return "PraNet"
    if "deeplab" in s:
        return "DeepLabV3-R50"
    if "segformer" in s:
        return "SegFormer-B0"
    raise RuntimeError(f"Unknown model family: {v}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def action_seed(model_state_id: str, sample_id: str) -> int:
    s = f"R31B1|{model_state_id}|{sample_id}".encode("utf-8")
    x = int(hashlib.sha256(s).hexdigest()[:8], 16)
    return 1 + (x % 2_000_000_000)


def verify_upstream() -> Dict[str, Any]:
    require_sha(
        R31B0_PROTOCOL,
        EXPECTED_R31B0_PROTOCOL_SHA256,
        "R31B0 protocol",
    )
    require_sha(
        R31B0_FINAL,
        EXPECTED_R31B0_FINAL_SHA256,
        "R31B0 final lock",
    )
    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SHA256, "R05D3 runner")
    require_sha(R03_SCRIPT, EXPECTED_R03_SHA256, "R03 model implementation")

    p = json.loads(R31B0_PROTOCOL.read_text(encoding="utf-8"))
    f = json.loads(R31B0_FINAL.read_text(encoding="utf-8"))

    if p.get("third_action", {}).get("name") != THIRD_ACTION:
        raise RuntimeError("R31B0 third action changed.")
    if list(p["third_action"]["optimizer_selection"]["learning_rate_candidates"]) != list(LR_CANDIDATES):
        raise RuntimeError("R31B0 LR candidates changed.")
    if int(f.get("design_cases", -1)) != EXPECTED_DESIGN_CASES:
        raise RuntimeError("R31B0 design case count changed.")
    if int(f.get("evaluation_cases", -1)) != EXPECTED_EVAL_CASES:
        raise RuntimeError("R31B0 evaluation case count changed.")

    return {"protocol": p, "final": f}


def read_partition() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not R31B0_PARTITION.is_file():
        raise FileNotFoundError(R31B0_PARTITION)

    d = pd.read_csv(R31B0_PARTITION, low_memory=False)
    required = {
        "sample_id",
        "r31b_partition",
        "image_path",
        "gt_path",
        "split_sha256_key",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Partition missing columns={missing}")
    if len(d) != EXPECTED_SOURCE_CASES:
        raise RuntimeError(f"Partition rows={len(d)} expected={EXPECTED_SOURCE_CASES}")
    if d["sample_id"].astype(str).nunique() != EXPECTED_SOURCE_CASES:
        raise RuntimeError("Partition sample_id not unique.")

    design = d[d["r31b_partition"] == DESIGN_PARTITION].copy()
    evaluation = d[d["r31b_partition"] == EVAL_PARTITION].copy()

    if len(design) != EXPECTED_DESIGN_CASES:
        raise RuntimeError(f"Design cases={len(design)}")
    if len(evaluation) != EXPECTED_EVAL_CASES:
        raise RuntimeError(f"Evaluation cases={len(evaluation)}")

    # Metadata-only checks. Evaluation image/GT bytes are NOT opened.
    for x in d["image_path"].astype(str):
        reject_external_path(Path(x))
    for x in d["gt_path"].astype(str):
        reject_external_path(Path(x))

    overlap = set(design["sample_id"].astype(str)) & set(evaluation["sample_id"].astype(str))
    if overlap:
        raise RuntimeError("Design/evaluation case overlap.")

    design = design.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    evaluation = evaluation.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return design, evaluation


def load_panel() -> pd.DataFrame:
    if not R05D3_CHECKPOINT_MANIFEST.is_file():
        raise FileNotFoundError(R05D3_CHECKPOINT_MANIFEST)

    d = pd.read_csv(R05D3_CHECKPOINT_MANIFEST, low_memory=False)
    required = {"model_family", "model_state_id", "training_seed", "checkpoint_sha256"}
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"Checkpoint manifest missing columns={missing}")
    if len(d) != EXPECTED_STATES:
        raise RuntimeError(f"Checkpoint states={len(d)} expected={EXPECTED_STATES}")

    rows = []
    seen = set()
    for _, row in d.iterrows():
        fam = canonical_family(row["model_family"])
        seed = int(row["training_seed"])
        sid = str(row["model_state_id"])
        sha = str(row["checkpoint_sha256"]).strip().lower()

        exp = EXPECTED_CHECKPOINT_SHAS.get(fam, {}).get(seed)
        if exp is None or sha != exp:
            raise RuntimeError(
                f"Frozen checkpoint mismatch family={fam} seed={seed} got={sha} expected={exp}"
            )
        if (fam, seed) in seen:
            raise RuntimeError(f"Duplicate state {fam}/{seed}")
        seen.add((fam, seed))
        rows.append({
            "model_family": fam,
            "model_state_id": sid,
            "training_seed": seed,
            "checkpoint_sha256": sha,
        })

    expected = {
        (fam, seed)
        for fam, by_seed in EXPECTED_CHECKPOINT_SHAS.items()
        for seed in by_seed
    }
    if seen != expected:
        raise RuntimeError(f"Nine-state inventory drift: {seen ^ expected}")

    return pd.DataFrame(rows)


def write_execution_contract(out: Path) -> Tuple[Path, str]:
    path = out / "R31B1_MEMO_EXECUTION_CONTRACT_PRE_OUTCOME.json"
    payload = {
        "status": "LOCKED_BEFORE_DESIGN_GT_OR_MEMO_OUTCOME",
        "version": VERSION,
        "upstream_R31B0_final_sha256": EXPECTED_R31B0_FINAL_SHA256,
        "upstream_R31B0_protocol_sha256": EXPECTED_R31B0_PROTOCOL_SHA256,
        "execution_contract": EXECUTION_CONTRACT,
        "information_boundary_at_lock_time": {
            "design_gt_pixels_read": False,
            "memo_outcomes_generated": False,
            "evaluation_800_gt_pixels_read": False,
            "external_data_access": False,
        },
    }
    atomic_json(path, payload)
    return path, sha256_file(path)


def load_gt352(path: Path) -> np.ndarray:
    reject_external_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        gt = im.convert("L")
        if gt.size != (IMAGE_SIZE, IMAGE_SIZE):
            gt = gt.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.NEAREST,
            )
        a = np.asarray(gt, dtype=np.uint8)
    return (a > 127).astype(np.uint8)


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    if p.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Prediction shape={p.shape}")
    if g.shape != p.shape:
        raise RuntimeError(f"GT shape={g.shape}")
    ps = int(p.sum())
    gs = int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    inter = int(np.logical_and(p, g).sum())
    return float((2.0 * inter) / (ps + gs + 1e-12))


def snapshot_model(model) -> Tuple[List[Any], List[Any]]:
    params = [p.detach().clone() for p in model.parameters()]
    buffers = [b.detach().clone() for b in model.buffers()]
    return params, buffers


def restore_model(model, param_values: Sequence[Any], buffer_values: Sequence[Any]):
    import torch
    with torch.no_grad():
        params = list(model.parameters())
        bufs = list(model.buffers())
        if len(params) != len(param_values) or len(bufs) != len(buffer_values):
            raise RuntimeError("Model parameter/buffer schema changed during MEMO.")
        for p, v in zip(params, param_values):
            p.copy_(v)
        for b, v in zip(bufs, buffer_values):
            b.copy_(v)


def augment_batch(x, torch):
    if x.ndim != 4 or x.shape[0] != 1:
        raise RuntimeError(f"Expected 1xCxHxW tensor, got {tuple(x.shape)}")
    return torch.cat(
        [
            x,
            torch.flip(x, dims=(-1,)),
            torch.flip(x, dims=(-2,)),
            torch.flip(x, dims=(-2, -1)),
        ],
        dim=0,
    )


def inverse_align_probs(prob4, torch):
    if prob4.ndim != 4 or prob4.shape[0] != 4 or prob4.shape[1] != 1:
        raise RuntimeError(f"Expected 4x1xHxW probabilities, got {tuple(prob4.shape)}")
    return torch.cat(
        [
            prob4[0:1],
            torch.flip(prob4[1:2], dims=(-1,)),
            torch.flip(prob4[2:3], dims=(-2,)),
            torch.flip(prob4[3:4], dims=(-2, -1)),
        ],
        dim=0,
    )


def binary_entropy_mean(p, torch):
    eps = 1e-6
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()


def build_runtime_adapter(state: Mapping[str, Any], device, r05, r03):
    import torch

    family = canonical_family(state["model_family"])
    seed = int(state["training_seed"])
    r05.set_runtime_seed(seed)

    if family == "PraNet":
        helper = r05.import_module(
            r05.PRANET_HELPER,
            f"r31b1_pranet_{seed}",
        )
        training = helper.import_training_helper()
        model = helper.load_model(training, seed, device)

        def to_tensor(native):
            return helper.image_to_model_tensor(training, native).to(
                device, non_blocking=True
            )

        def foreground_prob(x):
            z = helper.final_logit(model, x)
            if z.ndim != 4 or z.shape[1] != 1:
                raise RuntimeError(f"PraNet logit shape={tuple(z.shape)}")
            return torch.sigmoid(z)

        def final_mask(x):
            z = helper.final_logit(model, x)
            a = z[0, 0].detach().float().cpu().numpy()
            return np.asarray(r05.logit_to_mask(a), dtype=np.uint8)

        context = {"helper": "PraNet S03/R05 exact helper"}

    elif family == "DeepLabV3-R50":
        helper = r05.import_module(
            r05.DEEPLAB_HELPER,
            f"r31b1_deeplab_{seed}",
        )
        training = helper.import_training_helper()
        helper.seed_everything(20260817)
        model = helper.load_model(training, seed, device)

        def to_tensor(native):
            return helper.image_to_model_tensor(training, native).to(
                device, non_blocking=True
            )

        def foreground_prob(x):
            z = training.deeplab_logits(model, x)
            if z.ndim != 4 or z.shape[1] != 1:
                raise RuntimeError(f"DeepLab logit shape={tuple(z.shape)}")
            return torch.sigmoid(z)

        def final_mask(x):
            z = training.deeplab_logits(model, x)
            a = z[0, 0].detach().float().cpu().numpy()
            return np.asarray(r05.logit_to_mask(a), dtype=np.uint8)

        context = {"helper": "DeepLab S05-C/R05 exact helper"}

    elif family == "SegFormer-B0":
        seg_context = r03.build_segformer_context(device)
        (
            _torch,
            _nn,
            F,
            SegformerForSemanticSegmentation,
            config,
            mean,
            std,
        ) = seg_context

        model = r03.load_segformer_state(
            seed,
            device,
            SegformerForSemanticSegmentation,
            config,
            torch,
        )

        def to_tensor(native):
            return r03.segformer_tensor(native, mean, std, torch).to(
                device, non_blocking=True
            )

        def foreground_prob(x):
            logits, _ = r03.segformer_logits_and_z(model, x, F)
            if logits.ndim != 4 or logits.shape[1] != 2:
                raise RuntimeError(f"SegFormer logits shape={tuple(logits.shape)}")
            return torch.softmax(logits, dim=1)[:, 1:2]

        def final_mask(x):
            _, z = r03.segformer_logits_and_z(model, x, F)
            a = z[0].detach().float().cpu().numpy()
            return np.asarray(r05.logit_to_mask(a), dtype=np.uint8)

        context = {"helper": "R03 SegFormer exact loader/preprocess"}

    else:
        raise RuntimeError(f"Unsupported family={family}")

    model.to(device)

    # R31B0: ALL MODEL PARAMETERS.
    for p in model.parameters():
        p.requires_grad_(True)

    if not list(model.parameters()):
        raise RuntimeError("No model parameters.")

    source_params, source_buffers = snapshot_model(model)

    return {
        "family": family,
        "seed": seed,
        "model": model,
        "to_tensor": to_tensor,
        "foreground_prob": foreground_prob,
        "final_mask": final_mask,
        "source_params": source_params,
        "source_buffers": source_buffers,
        "context": context,
    }


def one_case_all_lrs(
    runtime: Mapping[str, Any],
    state: Mapping[str, Any],
    row: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    import torch

    model = runtime["model"]
    to_tensor = runtime["to_tensor"]
    foreground_prob = runtime["foreground_prob"]
    final_mask = runtime["final_mask"]
    source_params = runtime["source_params"]
    source_buffers = runtime["source_buffers"]

    image_path = Path(str(row["image_path"]))
    gt_path = Path(str(row["gt_path"]))
    reject_external_path(image_path)
    reject_external_path(gt_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as im:
        native = im.convert("RGB")
    x = to_tensor(native)

    # This is the first GT pixel access for this design case.
    gt = load_gt352(gt_path)

    restore_model(model, source_params, source_buffers)
    model.eval()
    with torch.no_grad():
        source_mask = final_mask(x)
    source_dice = dice_binary(source_mask, gt)

    seed_txn = action_seed(str(state["model_state_id"]), str(row["sample_id"]))
    outputs = []

    for lr in LR_CANDIDATES:
        restore_model(model, source_params, source_buffers)
        set_seed(seed_txn)

        model.train()
        params = list(model.parameters())
        for p in params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            params,
            lr=float(lr),
            weight_decay=WEIGHT_DECAY,
        )
        opt.zero_grad(set_to_none=True)

        x4 = augment_batch(x, torch)
        prob4 = foreground_prob(x4)
        aligned = inverse_align_probs(prob4, torch)
        marginal = aligned.mean(dim=0, keepdim=True)
        loss = binary_entropy_mean(marginal, torch)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite MEMO loss state={state['model_state_id']} "
                f"sample={row['sample_id']} lr={lr}"
            )

        loss.backward()

        grad_sq = 0.0
        for p in params:
            if p.grad is not None:
                grad_sq += float(torch.sum(p.grad.detach().float() ** 2).cpu())
        grad_norm = math.sqrt(max(0.0, grad_sq))

        opt.step()
        opt.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            memo_mask = final_mask(x)

        memo_dice = dice_binary(memo_mask, gt)
        delta = float(memo_dice - source_dice)

        outputs.append({
            "sample_id": str(row["sample_id"]),
            "model_family": str(state["model_family"]),
            "model_state_id": str(state["model_state_id"]),
            "training_seed": int(state["training_seed"]),
            "checkpoint_sha256": str(state["checkpoint_sha256"]),
            "action": THIRD_ACTION,
            "learning_rate": float(lr),
            "action_seed": int(seed_txn),
            "source_dice": float(source_dice),
            "memo_dice": float(memo_dice),
            "delta_dice": delta,
            "harm": int(delta <= HARM_THRESHOLD),
            "benefit": int(delta >= BENEFIT_THRESHOLD),
            "neutral": int(HARM_THRESHOLD < delta < BENEFIT_THRESHOLD),
            "memo_loss_preupdate": float(loss.detach().cpu()),
            "grad_l2": float(grad_norm),
            "source_foreground_pixels": int(source_mask.sum()),
            "memo_foreground_pixels": int(memo_mask.sum()),
            "changed_pixels": int(np.count_nonzero(source_mask != memo_mask)),
        })

        del opt, x4, prob4, aligned, marginal, loss, memo_mask

    restore_model(model, source_params, source_buffers)
    del x, source_mask, gt
    return outputs


def worker_run(
    state_index: int,
    output_dir: Path,
    design_csv: Path,
):
    import torch

    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SHA256, "R05D3")
    require_sha(R03_SCRIPT, EXPECTED_R03_SHA256, "R03")

    panel = load_panel()
    if not (0 <= state_index < len(panel)):
        raise RuntimeError(f"Invalid state index={state_index}")

    state = panel.iloc[state_index].to_dict()
    design = pd.read_csv(design_csv, low_memory=False)

    if len(design) != EXPECTED_DESIGN_CASES:
        raise RuntimeError("Worker design case count changed.")
    if set(design["r31b_partition"].unique()) != {DESIGN_PARTITION}:
        raise RuntimeError("Worker received non-design cases.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[WORKER {state_index+1}/{EXPECTED_STATES}] "
        f"{state['model_state_id']} | {state['model_family']} | "
        f"device={device}"
    )

    runtime = build_runtime_adapter(state, device, import_module(
        R05D3_SCRIPT,
        f"r31b1_r05_{state_index}",
    ), import_module(
        R03_SCRIPT,
        f"r31b1_r03_{state_index}",
    ))

    rows: List[Dict[str, Any]] = []
    pbar = tqdm(
        design.to_dict(orient="records"),
        total=EXPECTED_DESIGN_CASES,
        desc=f"R31B1 {state['model_family']} {state['training_seed']}",
        unit="case",
        dynamic_ncols=True,
    )

    for row in pbar:
        result = one_case_all_lrs(runtime, state, row)
        rows.extend(result)
        last = result[-1]
        pbar.set_postfix(
            dDice=f"{last['delta_dice']:+.3f}",
            LR=f"{last['learning_rate']:.0e}",
        )

    expected_rows = EXPECTED_DESIGN_CASES * len(LR_CANDIDATES)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Worker rows={len(rows)} expected={expected_rows}")

    df = pd.DataFrame(rows)
    final_csv = output_dir / f"state_{state_index:02d}_memo_design.csv"
    atomic_csv(df, final_csv)

    sidecar = output_dir / f"state_{state_index:02d}_memo_design.lock.json"
    atomic_json(sidecar, {
        "status": "PASS_R31B1_STATE_MEMO_DESIGN",
        "version": VERSION,
        "state_index": int(state_index),
        "model_state_id": str(state["model_state_id"]),
        "model_family": str(state["model_family"]),
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": str(state["checkpoint_sha256"]),
        "rows": int(len(df)),
        "design_cases": int(df["sample_id"].nunique()),
        "lr_candidates": sorted(df["learning_rate"].unique().tolist()),
        "csv": str(final_csv),
        "csv_sha256": sha256_file(final_csv),
        "evaluation_800_gt_read": False,
        "external_data_access": False,
    })

    # Full CUDA process exits after this worker.
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("STATE DESIGN LOCK PASS:", state["model_state_id"])
    print("CSV SHA256:", sha256_file(final_csv))


def validate_state_artifact(out: Path, idx: int, state: Mapping[str, Any]) -> bool:
    csv_path = out / f"state_{idx:02d}_memo_design.csv"
    lock_path = out / f"state_{idx:02d}_memo_design.lock.json"
    if not csv_path.is_file() or not lock_path.is_file():
        return False
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "PASS_R31B1_STATE_MEMO_DESIGN":
            return False
        if str(lock.get("model_state_id")) != str(state["model_state_id"]):
            return False
        if sha256_file(csv_path) != str(lock.get("csv_sha256")):
            return False
        d = pd.read_csv(csv_path, low_memory=False)
        if len(d) != EXPECTED_DESIGN_CASES * len(LR_CANDIDATES):
            return False
        if d["sample_id"].astype(str).nunique() != EXPECTED_DESIGN_CASES:
            return False
        got_lrs = sorted(float(x) for x in d["learning_rate"].unique())
        if got_lrs != sorted(float(x) for x in LR_CANDIDATES):
            return False
        return True
    except Exception:
        return False


def summarize_and_select(all_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    expected = (
        EXPECTED_DESIGN_CASES
        * EXPECTED_STATES
        * len(LR_CANDIDATES)
    )
    if len(all_rows) != expected:
        raise RuntimeError(f"Aggregate rows={len(all_rows)} expected={expected}")

    if all_rows["sample_id"].astype(str).nunique() != EXPECTED_DESIGN_CASES:
        raise RuntimeError("Aggregate design case count changed.")
    if all_rows["model_state_id"].astype(str).nunique() != EXPECTED_STATES:
        raise RuntimeError("Aggregate state count changed.")

    summary_rows = []
    for lr, g in all_rows.groupby("learning_rate", sort=True):
        summary_rows.append({
            "learning_rate": float(lr),
            "model_cases": int(len(g)),
            "physical_cases": int(g["sample_id"].astype(str).nunique()),
            "states": int(g["model_state_id"].astype(str).nunique()),
            "mean_source_dice": float(g["source_dice"].mean()),
            "mean_memo_dice": float(g["memo_dice"].mean()),
            "mean_delta_dice": float(g["delta_dice"].mean()),
            "median_delta_dice": float(g["delta_dice"].median()),
            "harm_rate": float(g["harm"].mean()),
            "benefit_rate": float(g["benefit"].mean()),
            "neutral_rate": float(g["neutral"].mean()),
            "mean_changed_pixels": float(g["changed_pixels"].mean()),
            "mean_memo_loss_preupdate": float(g["memo_loss_preupdate"].mean()),
        })
    summary = pd.DataFrame(summary_rows)

    # Exact preregistered selection:
    # 1) highest mean deployed Dice
    # 2) lower HARM rate
    # 3) lower LR
    ranked = summary.sort_values(
        ["mean_memo_dice", "harm_rate", "learning_rate"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected_lr = float(ranked.iloc[0]["learning_rate"])

    family_rows = []
    for (lr, fam), g in all_rows.groupby(
        ["learning_rate", "model_family"],
        sort=True,
    ):
        family_rows.append({
            "learning_rate": float(lr),
            "model_family": str(fam),
            "model_cases": int(len(g)),
            "mean_source_dice": float(g["source_dice"].mean()),
            "mean_memo_dice": float(g["memo_dice"].mean()),
            "mean_delta_dice": float(g["delta_dice"].mean()),
            "harm_rate": float(g["harm"].mean()),
            "benefit_rate": float(g["benefit"].mean()),
        })
    family = pd.DataFrame(family_rows)

    selected_row = ranked.iloc[0].to_dict()
    decision = {
        "selected_learning_rate": selected_lr,
        "selection_rule": [
            "highest mean deployed Dice across all 200 design cases x 9 states",
            "lower HARM rate",
            "lower learning rate",
        ],
        "selected_summary": {
            k: (
                int(v)
                if isinstance(v, (np.integer,))
                else float(v)
                if isinstance(v, (np.floating,))
                else v
            )
            for k, v in selected_row.items()
        },
        "candidate_ranking": ranked.to_dict(orient="records"),
        "freeze_effect": (
            "The selected LR is now immutable for R31B evaluation. "
            "The 800 evaluation cases and all external cohorts may not change it."
        ),
    }
    return summary, family, decision


def make_inventory(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            rows.append({
                "relative_path": str(p.relative_to(out)),
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--_worker-state", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_design-csv", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    out = args.output_dir.resolve()

    if args._worker_state is not None:
        if args._design_csv is None:
            raise RuntimeError("Worker requires --_design-csv.")
        worker_run(args._worker_state, out, args._design_csv.resolve())
        return 0

    print("=" * 124)
    print("SafeTTA R31B1 MEMO-SEG4-1STEP Action Design + LR Freeze")
    print("Version:", VERSION)
    print("NeoPolyp design GT access      : YES (200 cases only)")
    print("NeoPolyp 800-case eval GT      : NO")
    print("EndoTect / external access     : NO")
    print("Third action                   :", THIRD_ACTION)
    print("LR candidates                  :", list(LR_CANDIDATES))
    print("Model states                   :", EXPECTED_STATES)
    print("Fresh CUDA process per state   : YES")
    print("Resume                         :", bool(args.resume))
    print("=" * 124)

    verify_upstream()
    design, evaluation = read_partition()
    panel = load_panel()

    if out.exists() and not args.resume:
        raise RuntimeError(
            f"Output exists; use --resume only if this exact R31B1 run should continue: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    contract_path = out / "R31B1_MEMO_EXECUTION_CONTRACT_PRE_OUTCOME.json"
    if args.resume and contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("execution_contract") != EXECUTION_CONTRACT:
            raise RuntimeError("Existing R31B1 execution contract differs.")
        contract_sha = sha256_file(contract_path)
    else:
        contract_path, contract_sha = write_execution_contract(out)

    # Freeze the exact 200-case runtime manifest before GT pixels are opened.
    design_runtime = out / "R31B1_MEMO_DESIGN_CASES_LOCK.csv"
    if not design_runtime.is_file():
        atomic_csv(design, design_runtime)
    else:
        prior = pd.read_csv(design_runtime, low_memory=False)
        if not prior.equals(design):
            raise RuntimeError("Design runtime manifest changed.")

    print("\nPRE-OUTCOME EXECUTION CONTRACT: PASS")
    print("  Contract SHA256      :", contract_sha)
    print("  Design cases         :", len(design))
    print("  Evaluation cases     :", len(evaluation), "(metadata only; GT forbidden)")
    print("  Adaptation mode      : model.train()")
    print("  Final prediction     : model.eval()")
    print("  All params + buffers : restored before each case x LR")
    print("  GT pixels read so far: NO")

    # From this point design GT is allowed, but only inside workers.
    for idx, state in panel.iterrows():
        if args.resume and validate_state_artifact(out, idx, state):
            print(
                f"[RESUME] state {idx+1}/{EXPECTED_STATES} "
                f"{state['model_state_id']}: PASS, skipping"
            )
            continue

        print(
            f"[CUDA ISOLATION] launching state {idx+1}/{EXPECTED_STATES}: "
            f"{state['model_state_id']}"
        )

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(out),
            "--_worker-state",
            str(idx),
            "--_design-csv",
            str(design_runtime),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R31B1 worker failed state={idx} "
                f"{state['model_state_id']} returncode={proc.returncode}"
            )
        if not validate_state_artifact(out, idx, state):
            raise RuntimeError(
                f"Worker returned success but state artifact validation failed: "
                f"{state['model_state_id']}"
            )

    parts = []
    for idx, state in panel.iterrows():
        if not validate_state_artifact(out, idx, state):
            raise RuntimeError(f"Missing/invalid completed state={idx}")
        parts.append(
            pd.read_csv(out / f"state_{idx:02d}_memo_design.csv", low_memory=False)
        )

    all_rows = pd.concat(parts, ignore_index=True)

    results_path = out / "R31B1_MEMO_DESIGN_ALL_RESULTS.csv"
    atomic_csv(all_rows, results_path)

    summary, family_summary, decision = summarize_and_select(all_rows)

    summary_path = out / "R31B1_MEMO_LR_SUMMARY.csv"
    family_path = out / "R31B1_MEMO_LR_FAMILY_SUMMARY.csv"
    selected_path = out / "R31B1_SELECTED_MEMO_LR_LOCK.json"

    atomic_csv(summary, summary_path)
    atomic_csv(family_summary, family_path)

    selected_payload = {
        "status": "FROZEN_R31B_MEMO_LR_AFTER_SOURCE_DESIGN_ONLY",
        "version": VERSION,
        "third_action": THIRD_ACTION,
        "R31B0_protocol_sha256": EXPECTED_R31B0_PROTOCOL_SHA256,
        "R31B0_final_sha256": EXPECTED_R31B0_FINAL_SHA256,
        "execution_contract_sha256": contract_sha,
        "design_case_manifest_sha256": sha256_file(design_runtime),
        "design_results_sha256": sha256_file(results_path),
        "lr_summary_sha256": sha256_file(summary_path),
        "family_summary_sha256": sha256_file(family_path),
        "design_cases": EXPECTED_DESIGN_CASES,
        "states": EXPECTED_STATES,
        "model_cases_per_lr": EXPECTED_DESIGN_CASES * EXPECTED_STATES,
        "evaluation_800_gt_access": False,
        "external_data_access": False,
        **decision,
        "next": "R31B2_800CASE_THIRD_ACTION_PREDICTION_AND_SEMANTIC_LOCK",
    }
    atomic_json(selected_path, selected_payload)

    inventory = make_inventory(out)
    inv_path = out / "R31B1_SHA256_INVENTORY.csv"
    atomic_csv(inventory, inv_path)

    final = {
        "status": "PASS_R31B1_MEMO_ACTION_DESIGN_AND_LR_FREEZE_COMPLETE",
        "version": VERSION,
        "third_action": THIRD_ACTION,
        "selected_learning_rate": decision["selected_learning_rate"],
        "R31B0_final_sha256": EXPECTED_R31B0_FINAL_SHA256,
        "execution_contract_sha256": contract_sha,
        "selected_lr_lock": str(selected_path),
        "selected_lr_lock_sha256": sha256_file(selected_path),
        "design_results_sha256": sha256_file(results_path),
        "design_cases": EXPECTED_DESIGN_CASES,
        "evaluation_cases_used_for_selection": 0,
        "evaluation_800_gt_access": False,
        "external_data_access": False,
        "next": "R31B2_800CASE_THIRD_ACTION_PREDICTION_AND_SEMANTIC_LOCK",
    }

    final_path = out / "R31B1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R31B1 MEMO LR DESIGN RESULTS")
    print("=" * 124)
    print(
        summary[
            [
                "learning_rate",
                "mean_source_dice",
                "mean_memo_dice",
                "mean_delta_dice",
                "harm_rate",
                "benefit_rate",
                "neutral_rate",
                "mean_changed_pixels",
            ]
        ].to_string(index=False)
    )

    print("\nSELECTED MEMO LR:", decision["selected_learning_rate"])
    print("Selection used 800-case evaluation subset: NO")
    print("EndoTect/external access               : NO")

    print("\nFINAL STATUS : PASS_R31B1_MEMO_ACTION_DESIGN_AND_LR_FREEZE_COMPLETE")
    print("Selected LR lock SHA256 :", sha256_file(selected_path))
    print("Final lock SHA256       :", sha256_file(final_path))
    print("Output                  :", out)
    print("NEXT                    : R31B2_800CASE_THIRD_ACTION_PREDICTION_AND_SEMANTIC_LOCK")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
