#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1.py

SafeTTA R17A — pre-GT score lock for the two published reliability signals
that can be reproduced faithfully on the supported PolypGen family:

  1) TEGDA-ADIC (binary-task-compatible instantiation of the official
     class-specific ADIC formulation)
  2) MC-dropout predictive uncertainty from the same 10 stochastic passes

SUPPORTED FAMILY ONLY
---------------------
DeepLabV3-R50 seeds:
  20260817, 20260818, 20260819

PraNet is deliberately EXCLUDED:
  exact frozen-model audit found 0 native torch.nn.Dropout modules.
No dropout is injected into PraNet.

OFFICIAL TEGDA IMPLEMENTATION ANCHOR
------------------------------------
Repository:
  https://github.com/HiLab-git/TEGDA
File:
  code/sota/adic2d.py

Relevant official behavior:
  - n_iter=10
  - existing nn.Dropout modules are switched to train mode
  - no dropout p rewrite is performed inside evaluate_dropout
  - stochastic probabilities are generated before adaptation
  - hard dropout predictions are compared with the deterministic current
    prediction using Dice
  - confidence weighting is computed from dropout-prediction entropy

Binary-task-compatible ADIC used here
-------------------------------------
The official 2D code computes a class-specific ADIC for each foreground class.
PolypGen is binary, so the single foreground class is used directly:

  dropout hard masks:
      h_m = 1[p_m >= 0.5]

  agreement:
      ADI = mean_m Dice(h_m, h_det)

  official class-specific confidence weight:
      r = mean_m h_m
      u = max_m h_m
      e = -(r log(r+1e-6) + (1-r) log(1-r+1e-6))
      e_norm = (e-e_min+1e-6)/(e_max-e_min+2e-6)
      w = (sum((1-e_norm)u)+1e-6)/(sum(u)+1e-6)

  ADIC_quality = w * ADI
  ADIC_harm_risk = -ADIC_quality

This is the direct single-foreground-class reduction of the official
class-specific formula; no target labels or target performance are used.

MC-dropout baseline
-------------------
Using the same 10 stochastic probabilities:
  pbar = mean_m p_m
  MC predictive entropy =
      mean_pixels[-pbar log pbar -(1-pbar)log(1-pbar)]

Higher MC entropy = higher HARM risk.

STRICT INFORMATION BOUNDARY
---------------------------
Training: NO
TTA: NO
Target GT loading: NO
HARM/outcome loading: NO
Threshold tuning: NO
Target recalibration: NO
Target score reversal: NO
Model modification: only official-compatible Dropout train/eval mode toggling
New dropout injection: NO

HISTORICAL SOURCE-STATE PARITY GATE
-----------------------------------
Before a score row is accepted, the deterministic SOURCE hard prediction from
the exact frozen model/input path must match the canonical frozen PolypGen
DeepLab SOURCE hard mask bit-for-bit.

If ANY parity mismatch occurs:
  STOP;
  do not write the score lock.

IMPLEMENTATION NOTE
-------------------
The exact DeepLab model is instantiated by intercepting the authoritative
R05D3 runner at the exact frozen checkpoint load. The exact per-target input
preprocessing prefix is extracted from the same SHA-locked run_deeplab_state
function via AST, rather than re-implementing the transform by hand.

The score table is serialized and SHA256-locked before any later GT/HARM join.

FIX1
----
The v1 runner stopped before inference because the extracted per-target
preprocessing prefix referenced the historical local variable `helper`, whose
exact assignment occurs before the target loop. fix1 resolves and prepends
that prerequisite assignment (and any of its exact local dependencies) from
the same SHA-locked run_deeplab_state AST. No preprocessing is guessed.

FIX2
----
The exact pre-loop setup also references `state`, which is not a local
dependency to reconstruct: it is an original run_deeplab_state function
argument. fix2 explicitly supplies the already SHA-locked current state dict
to the extracted historical input contract. No new value is inferred.
"""

from __future__ import annotations

import argparse
import builtins
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERSION = "2026-09-09-Q1-R17A-ADIC-MC-DEEPLAB3-PREGT-SCORE-LOCK-v1-fix2"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2")

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R10_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")
R10_SHA = "4cae02c30e82781c6e7d3b74b6a312974789213a9cc59a44ca99bbfb94405708"

R10_DIR_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
)
MANIFEST_REL = R10_DIR_REL / "frozen_polypgen_rgb_manifest_no_gt.csv"
CHECKPOINT_MANIFEST_REL = R10_DIR_REL / "checkpoint_manifest.csv"
STATE_DIR_REL = R10_DIR_REL / "state_predictions"

SEEDS = (20260817, 20260818, 20260819)
FAMILY = "DeepLabV3-R50"

EXPECTED_CASES = 1532
EXPECTED_STATES = 3
EXPECTED_ROWS = EXPECTED_CASES * EXPECTED_STATES
MASK_H = 352
MASK_W = 352
MASK_PIXELS = MASK_H * MASK_W
PACKED_BYTES = (MASK_PIXELS + 7) // 8

N_DROPOUT = 10
BASE_RNG_SEED = 20260909

# We intentionally do not fetch anything from target GT/HARM/outcomes here.
BANNED_MANIFEST_COLUMNS = {
    "gt", "gt_path", "gt_sha256", "ground_truth", "groundtruth",
    "dice", "delta_dice", "harm", "harm_label", "outcome", "benefit",
}


class _CapturedExactModel(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA mismatch\npath={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def robust_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Tuple[dict, str]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint payload is not dict: {type(payload)}")
    for key in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "best_model_state",
        "weights",
    ]:
        v = payload.get(key)
        if isinstance(v, dict) and v:
            return v, key

    if payload and all(isinstance(k, str) for k in payload):
        tensor_count = sum(torch.is_tensor(v) for v in payload.values())
        if tensor_count >= max(1, len(payload) // 2):
            return payload, "RAW_STATE_DICT"

    raise RuntimeError("Unable to identify checkpoint state_dict.")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([
            root / p,
            root / str(raw).replace("/", "\\"),
        ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(raw)


def validate_no_gt_manifest(df: pd.DataFrame) -> None:
    cols = {str(c).strip().lower() for c in df.columns}
    bad = sorted(cols & BANNED_MANIFEST_COLUMNS)
    if bad:
        raise RuntimeError(f"Pre-GT manifest exposes banned target columns: {bad}")

    required = {
        "sample_id",
        "original_polypgen_sample_id",
        "center",
        "image_path",
        "image_raw_sha256",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Pre-GT manifest missing columns: {missing}")

    if len(df) != EXPECTED_CASES:
        raise RuntimeError(f"Manifest rows={len(df)}, expected={EXPECTED_CASES}")
    if df["sample_id"].astype(str).nunique() != EXPECTED_CASES:
        raise RuntimeError("Manifest sample_id is not unique.")


def load_deeplab_states(root: Path) -> List[dict]:
    p = root / CHECKPOINT_MANIFEST_REL
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)

    required = {
        "model_family",
        "training_seed",
        "model_state_id",
        "checkpoint",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Checkpoint manifest missing columns: {missing}")

    rows = []
    for r in df.itertuples(index=False):
        if str(r.model_family) != FAMILY:
            continue
        seed = int(r.training_seed)
        if seed not in SEEDS:
            continue
        ckpt = resolve_path(root, str(r.checkpoint))
        actual = validate_sha(
            ckpt,
            str(r.checkpoint_sha256),
            f"{FAMILY} seed={seed} checkpoint",
        )
        rows.append({
            "model_family": FAMILY,
            "training_seed": seed,
            "model_state_id": str(r.model_state_id),
            "checkpoint": str(ckpt),
            "checkpoint_sha256": actual,
        })

    rows.sort(key=lambda x: x["training_seed"])
    if [r["training_seed"] for r in rows] != list(SEEDS):
        raise RuntimeError(
            f"DeepLab states mismatch: {[r['training_seed'] for r in rows]}"
        )
    return rows


def exact_state_artifact_paths(root: Path, seed: int) -> dict:
    state_dir = root / STATE_DIR_REL
    stem = f"deeplabv3_r50_seed{seed}"
    return {
        "npz": state_dir / f"{stem}_predictions.npz",
        "index": state_dir / f"{stem}_index.csv",
        "lock": state_dir / f"{stem}_state_lock.json",
    }


def load_frozen_source_lock(
    root: Path,
    state: dict,
    manifest: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    seed = int(state["training_seed"])
    paths = exact_state_artifact_paths(root, seed)

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)

    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))

    if str(lock.get("checkpoint_sha256", "")).lower() != state["checkpoint_sha256"]:
        raise RuntimeError(f"State lock checkpoint SHA mismatch seed={seed}")

    if lock.get("npz_sha256"):
        validate_sha(paths["npz"], lock["npz_sha256"], f"DeepLab NPZ seed={seed}")
    if lock.get("index_sha256"):
        validate_sha(paths["index"], lock["index_sha256"], f"DeepLab index seed={seed}")

    idx = pd.read_csv(paths["index"], low_memory=False)
    if len(idx) != EXPECTED_CASES:
        raise RuntimeError(f"DeepLab index rows={len(idx)}, seed={seed}")

    required = {
        "row_index",
        "sample_id",
        "image_path",
        "image_raw_sha256",
        "model_family",
        "model_state_id",
        "training_seed",
        "checkpoint_sha256",
        "source_foreground_pixels",
    }
    missing = sorted(required - set(idx.columns))
    if missing:
        raise RuntimeError(f"DeepLab index missing columns seed={seed}: {missing}")

    if list(idx["row_index"].astype(int)) != list(range(EXPECTED_CASES)):
        raise RuntimeError(f"DeepLab index row order invalid seed={seed}")

    if not np.array_equal(
        idx["sample_id"].astype(str).to_numpy(),
        manifest["sample_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError(f"DeepLab index sample order mismatch seed={seed}")

    if not np.array_equal(
        idx["image_raw_sha256"].astype(str).str.lower().to_numpy(),
        manifest["image_raw_sha256"].astype(str).str.lower().to_numpy(),
    ):
        raise RuntimeError(f"DeepLab index image SHA order mismatch seed={seed}")

    with np.load(paths["npz"], allow_pickle=False) as z:
        if "source_masks_packed" not in z.files:
            raise RuntimeError(f"source_masks_packed missing seed={seed}")
        source_pack = np.asarray(z["source_masks_packed"], dtype=np.uint8)

    if source_pack.shape != (EXPECTED_CASES, PACKED_BYTES):
        raise RuntimeError(
            f"Frozen source mask shape={source_pack.shape}, "
            f"expected=({EXPECTED_CASES},{PACKED_BYTES})"
        )

    return source_pack, idx, lock


def capture_exact_loaded_deeplab(
    r05,
    r03,
    state: dict,
) -> nn.Module:
    runner = getattr(r05, "run_deeplab_state", None)
    if runner is None:
        raise RuntimeError("Authoritative R05D3 lacks run_deeplab_state")

    payload = robust_torch_load(Path(state["checkpoint"]))
    state_dict, _ = extract_state_dict(payload)
    expected_keys = set(str(k) for k in state_dict.keys())

    captured = {}
    original = nn.Module.load_state_dict

    def wrapped(self, sd, *args, **kwargs):
        result = original(self, sd, *args, **kwargs)
        try:
            keys = set(str(k) for k in sd.keys())
        except Exception:
            keys = set()

        if keys == expected_keys:
            captured["model"] = self
            captured["missing"] = list(getattr(result, "missing_keys", []))
            captured["unexpected"] = list(getattr(result, "unexpected_keys", []))
            raise _CapturedExactModel()
        return result

    nn.Module.load_state_dict = wrapped
    try:
        sig = inspect.signature(runner)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "state":
                kwargs[name] = state
            elif name == "targets":
                kwargs[name] = []
            elif name == "device":
                kwargs[name] = torch.device("cpu")
            elif name == "r03":
                kwargs[name] = r03
            elif name == "seg_context":
                kwargs[name] = None
            elif param.default is inspect._empty:
                raise RuntimeError(
                    f"Unsupported required run_deeplab_state argument: {name}"
                )
        try:
            runner(**kwargs)
        except _CapturedExactModel:
            pass
        except Exception as e:
            if "model" not in captured:
                raise RuntimeError(
                    "Exact DeepLab model capture failed before checkpoint load: "
                    f"{type(e).__name__}: {e}"
                ) from e
    finally:
        nn.Module.load_state_dict = original

    if "model" not in captured:
        raise RuntimeError("Exact DeepLab checkpoint load was not captured.")
    if captured["missing"] or captured["unexpected"]:
        raise RuntimeError(
            f"DeepLab strict-load mismatch missing={captured['missing']} "
            f"unexpected={captured['unexpected']}"
        )

    model = captured["model"]
    del payload, state_dict
    return model


def compile_exact_input_prefix(r05):
    """
    Extract the exact SHA-locked DeepLab input-construction path from
    run_deeplab_state.

    fix1 difference:
      v1 copied only the per-target loop prefix through `x = ...`. The
      historical prefix depends on local setup variables (observed: `helper`)
      created before the loop, so executing only the loop prefix raised
      NameError before any inference.

      fix1 resolves those local dependencies from the SAME historical
      function AST and prepends only the exact prerequisite assignments.
      Nothing is guessed or reimplemented.
    """
    runner = r05.run_deeplab_state
    fn_src = textwrap.dedent(inspect.getsource(runner))
    tree = ast.parse(fn_src)
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Could not parse run_deeplab_state AST.")
    fn = tree.body[0]

    candidate_for = None
    x_index = None

    for node in ast.walk(fn):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        for j, stmt in enumerate(node.body):
            target_names = []
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target_names.append(stmt.target.id)
            if "x" in target_names:
                candidate_for = node
                x_index = j
                break
        if candidate_for is not None:
            break

    if candidate_for is None or x_index is None:
        raise RuntimeError(
            "Could not locate x assignment in run_deeplab_state target loop."
        )

    loop_prefix = candidate_for.body[: x_index + 1]

    def assigned_names(stmt):
        out = set()

        def add_target(t):
            if isinstance(t, ast.Name):
                out.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    add_target(e)

        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    add_target(t)
            elif isinstance(node, ast.AnnAssign):
                add_target(node.target)
            elif isinstance(node, ast.NamedExpr):
                add_target(node.target)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                add_target(node.target)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    out.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    out.add(alias.asname or alias.name)
        return out

    def loaded_names(stmts):
        out = set()
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    out.add(node.id)
        return out

    # Locate top-level statements before the target loop. The DeepLab runner's
    # helper/module setup is expected here.
    top_before = []
    found_top_loop = False
    for stmt in fn.body:
        if stmt is candidate_for:
            found_top_loop = True
            break
        top_before.append(stmt)

    if not found_top_loop:
        # The target loop may be nested; find the top-level statement that
        # contains it and use statements preceding that container.
        top_before = []
        container_found = False
        for stmt in fn.body:
            if any(node is candidate_for for node in ast.walk(stmt)):
                container_found = True
                break
            top_before.append(stmt)
        if not container_found:
            raise RuntimeError(
                "Could not locate target loop container in run_deeplab_state."
            )

    global_names = set(r05.__dict__.keys())
    provided_names = {"r03", "state", "target", "i", "device"}
    builtin_names = set(dir(builtins))

    # Resolve only local prerequisites that the loop prefix loads but does not
    # itself assign and that are not module globals/provided names/builtins.
    selected_setup = []
    selected_ids = set()

    def unresolved_for(stmts):
        assigned = set()
        for s in stmts:
            assigned |= assigned_names(s)
        loaded = loaded_names(stmts)
        return (
            loaded
            - assigned
            - global_names
            - provided_names
            - builtin_names
        )

    unresolved = unresolved_for(loop_prefix)

    # Iteratively select exact earlier top-level assignments that bind missing
    # names; then resolve any dependencies introduced by those assignments.
    for _ in range(20):
        if not unresolved:
            break
        progress = False

        for idx, stmt in enumerate(top_before):
            if idx in selected_ids:
                continue
            binds = assigned_names(stmt)
            if binds & unresolved:
                selected_ids.add(idx)
                selected_setup.append((idx, stmt))
                progress = True

        selected_setup.sort(key=lambda x: x[0])
        combined = [s for _, s in selected_setup] + loop_prefix
        unresolved = unresolved_for(combined)

        if not progress and unresolved:
            raise RuntimeError(
                "Could not resolve exact historical DeepLab preprocessing "
                f"local dependencies from run_deeplab_state: {sorted(unresolved)}"
            )
    else:
        raise RuntimeError("Dependency resolution exceeded iteration limit.")

    setup = [s for _, s in selected_setup]
    combined = setup + loop_prefix

    setup_text = "\\n".join(
        ast.unparse(s) if hasattr(ast, "unparse") else type(s).__name__
        for s in setup
    )
    prefix_text = "\\n".join(
        ast.unparse(s) if hasattr(ast, "unparse") else type(s).__name__
        for s in loop_prefix
    )
    provenance_text = (
        "===== EXACT PRE-LOOP DEPENDENCY SETUP =====\\n"
        + (setup_text if setup_text else "<NONE>")
        + "\\n\\n===== EXACT PER-TARGET PREFIX THROUGH x =====\\n"
        + prefix_text
    )

    # Safety gate: prerequisites and input prefix must remain strictly
    # pre-inference/pre-outcome. We permit the substring "target" because it
    # denotes the image record itself, but block actual outcome operations.
    low = provenance_text.lower()
    banned = [
        "optimizer.step",
        ".backward(",
        "delta_dice",
        "harm_label",
        "source_dice",
        "action_dice",
        "tent_loss",
    ]
    hit = [b for b in banned if b in low]
    if hit:
        raise RuntimeError(
            f"Unsafe DeepLab input-construction extraction contains {hit}\\n"
            f"{provenance_text}"
        )

    module = ast.Module(body=combined, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(
        module,
        filename="<exact_R05_run_deeplab_state_input_contract>",
        mode="exec",
    )

    return code, provenance_text


def build_exact_input(
    prefix_code,
    r05,
    r03,
    state: dict,
    target: dict,
    i: int,
    device: torch.device,
) -> torch.Tensor:
    env = dict(r05.__dict__)
    env.update({
        "r03": r03,
        "state": state,
        "target": target,
        "i": i,
        "device": device,
    })
    exec(prefix_code, env, env)
    if "x" not in env or not torch.is_tensor(env["x"]):
        raise RuntimeError(
            f"Historical preprocessing prefix did not create tensor x at row={i}"
        )
    return env["x"]


def extract_logits_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, dict):
        if "out" in output and torch.is_tensor(output["out"]):
            logits = output["out"]
        else:
            tensors = [v for v in output.values() if torch.is_tensor(v)]
            if len(tensors) != 1:
                raise RuntimeError(
                    f"Ambiguous DeepLab dict output keys={list(output.keys())}"
                )
            logits = tensors[0]
    elif isinstance(output, (tuple, list)):
        tensors = [v for v in output if torch.is_tensor(v)]
        if len(tensors) != 1:
            raise RuntimeError(
                f"Ambiguous DeepLab tuple/list output length={len(output)}"
            )
        logits = tensors[0]
    else:
        raise RuntimeError(f"Unsupported DeepLab output type: {type(output)}")

    if logits.ndim != 4 or logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected DeepLab logits shape={tuple(logits.shape)}")

    if tuple(logits.shape[-2:]) != (MASK_H, MASK_W):
        logits = F.interpolate(
            logits,
            size=(MASK_H, MASK_W),
            mode="bilinear",
            align_corners=False,
        )
    return logits


def binary_probability_and_hard(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = int(logits.shape[1])
    if c == 1:
        p = torch.sigmoid(logits[:, 0])
        hard = (logits[:, 0] >= 0.0).to(torch.uint8)
    elif c == 2:
        prob = torch.softmax(logits, dim=1)
        p = prob[:, 1]
        hard = torch.argmax(logits, dim=1).to(torch.uint8)
    else:
        raise RuntimeError(
            f"DeepLab output has {c} channels; binary PolypGen expected 1 or 2."
        )
    return p, hard


def set_source_deterministic_mode(model: nn.Module) -> List[dict]:
    model.eval()
    drops = []
    for name, m in model.named_modules():
        if type(m) is nn.Dropout:
            drops.append({"name": name, "p": float(m.p)})
            m.eval()
    return drops


def set_official_dropout_mode(model: nn.Module) -> List[dict]:
    # Match official TEGDA behavior: only exact nn.Dropout modules are toggled.
    model.eval()
    drops = []
    for name, m in model.named_modules():
        if type(m) is nn.Dropout:
            m.train()
            drops.append({"name": name, "p": float(m.p)})
    return drops


def pack_mask(mask_hw: np.ndarray, bitorder: str) -> np.ndarray:
    m = np.asarray(mask_hw, dtype=np.uint8)
    if m.shape != (MASK_H, MASK_W):
        raise RuntimeError(f"Mask shape={m.shape}")
    return np.packbits(m.reshape(-1), bitorder=bitorder)


def binary_dice_batch(drop_hard: np.ndarray, curr_hard: np.ndarray) -> np.ndarray:
    """
    Official get_batch_dice foreground-class reduction:
      (2*intersection+1e-5)/(sum_s+sum_g+1e-5)
    drop_hard: [M,H,W], curr_hard: [H,W]
    """
    s = (np.asarray(drop_hard) > 0).astype(np.uint8)
    g = (np.asarray(curr_hard) > 0).astype(np.uint8)[None, :, :]
    prod = s * g
    s0 = prod.sum(axis=(1, 2))
    s1 = s.sum(axis=(1, 2))
    s2 = g.sum(axis=(1, 2))
    return (2.0 * s0 + 1e-5) / (s1 + s2 + 1e-5)


def official_binary_class_confidence_weight(drop_hard: np.ndarray) -> float:
    """
    Direct class-1 reduction of official TEGDA calculate_ent_weight.
    """
    pred_sub = (np.asarray(drop_hard) == 1).astype(np.uint8)
    soft_pred = np.mean(pred_sub, axis=0)
    uni_pred = np.max(pred_sub, axis=0)
    ent_pred = -(
        soft_pred * np.log(soft_pred + 1e-6)
        + (1.0 - soft_pred) * np.log(1.0 - soft_pred + 1e-6)
    )
    ent_norm = (
        (ent_pred - ent_pred.min() + 1e-6)
        / (ent_pred.max() - ent_pred.min() + 2e-6)
    )
    weight = (
        (np.sum((1.0 - ent_norm) * uni_pred) + 1e-6)
        / (np.sum(uni_pred) + 1e-6)
    )
    return float(weight)


def compute_adic_mc(
    model: nn.Module,
    x: torch.Tensor,
    deterministic_p: torch.Tensor,
    deterministic_hard: torch.Tensor,
) -> dict:
    drops = set_official_dropout_mode(model)
    if len(drops) != 1:
        raise RuntimeError(
            f"Faithful DeepLab ADIC expected exactly 1 native nn.Dropout, got {drops}"
        )
    if not math.isclose(float(drops[0]["p"]), 0.5, rel_tol=0, abs_tol=0):
        raise RuntimeError(
            f"Frozen DeepLab dropout p changed: {drops[0]['p']} expected=0.5"
        )

    stochastic_probs = []
    stochastic_hard = []

    with torch.no_grad():
        for _ in range(N_DROPOUT):
            logits = extract_logits_tensor(model(x))
            p, hard = binary_probability_and_hard(logits)
            stochastic_probs.append(p[0].detach().float().cpu().numpy())
            stochastic_hard.append(hard[0].detach().cpu().numpy().astype(np.uint8))

    prob = np.stack(stochastic_probs, axis=0).astype(np.float32)  # [M,H,W]
    hard = np.stack(stochastic_hard, axis=0).astype(np.uint8)

    curr = deterministic_hard[0].detach().cpu().numpy().astype(np.uint8)

    dice_each = binary_dice_batch(hard, curr)
    adi = float(np.mean(dice_each))
    confidence_weight = official_binary_class_confidence_weight(hard)
    adic_quality = float(confidence_weight * adi)

    pbar = np.mean(prob.astype(np.float64), axis=0)
    predictive_entropy_map = -(
        pbar * np.log(pbar + 1e-10)
        + (1.0 - pbar) * np.log(1.0 - pbar + 1e-10)
    )
    mc_predictive_entropy = float(np.mean(predictive_entropy_map))
    mc_probability_variance = float(np.mean(np.var(prob.astype(np.float64), axis=0)))

    # Restore fully deterministic SOURCE eval mode after each sample.
    set_source_deterministic_mode(model)

    return {
        "adic_quality": adic_quality,
        "adic_harm_risk": -adic_quality,
        "adi_mean_foreground_dice": adi,
        "adic_confidence_weight": confidence_weight,
        "mc_predictive_entropy_risk": mc_predictive_entropy,
        "mc_probability_variance": mc_probability_variance,
        "dropout_dice_min": float(np.min(dice_each)),
        "dropout_dice_max": float(np.max(dice_each)),
    }


def verify_images_once(root: Path, manifest: pd.DataFrame) -> List[Path]:
    paths = []
    iterator = range(len(manifest))
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(manifest),
            desc="Verify frozen PolypGen RGB SHA256",
            unit="img",
            dynamic_ncols=True,
        )
    for i in iterator:
        row = manifest.iloc[i]
        p = resolve_path(root, str(row["image_path"]))
        actual = sha256_file(p)
        expected = str(row["image_raw_sha256"]).lower()
        if actual.lower() != expected:
            raise RuntimeError(
                f"Image SHA mismatch row={i} sample={row['sample_id']}\n"
                f"expected={expected}\nactual={actual}\npath={p}"
            )
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A ADIC + MC DROPOUT DEEPLAB3 PRE-GT SCORE LOCK =====")
    print("Version:", VERSION)
    print("Dataset: PolypGen")
    print("Family: DeepLabV3-R50")
    print("States: 3")
    print("Dropout passes per image:", N_DROPOUT)
    print("Training: NO")
    print("TTA: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Threshold tuning: NO")
    print("PraNet included: NO (no native nn.Dropout)")
    print()

    r05_path = root / R05_REL
    r03_path = root / R03_REL
    r10_path = root / R10_REL

    print("[1/7] Validate authoritative code lineage...")
    validate_sha(r05_path, R05_SHA, "R05D3")
    validate_sha(r03_path, R03_SHA, "R03")
    validate_sha(r10_path, R10_SHA, "R10L3A")

    r05 = import_from_path(r05_path, "q1_r05d3_r17a_adic_score")
    r03 = import_from_path(r03_path, "q1_r03_r17a_adic_score")

    print("[2/7] Load frozen no-GT manifest and 3 DeepLab states...")
    manifest_path = root / MANIFEST_REL
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    validate_no_gt_manifest(manifest)
    states = load_deeplab_states(root)

    print("[3/7] Verify frozen target image bytes...")
    image_paths = verify_images_once(root, manifest)

    print("[4/7] Extract exact R05D3 target preprocessing prefix...")
    prefix_code, prefix_text = compile_exact_input_prefix(r05)

    # Save the exact extracted prefix as provenance before score generation.
    prefix_path = out_dir / "R17A_ADIC_EXACT_R05_DEEPLAB_INPUT_PREFIX.txt"
    prefix_path.write_text(prefix_text + "\n", encoding="utf-8")

    score_rows = []
    state_meta = []

    print("[5/7] Run deterministic SOURCE parity + ADIC/MC scoring...")
    for state_ord, state in enumerate(states):
        seed = int(state["training_seed"])
        print(f"\n--- DeepLab seed {seed} ---")

        frozen_pack, idx, state_lock = load_frozen_source_lock(
            root, state, manifest
        )

        bitorder = str(state_lock.get("bitorder", "big"))
        if bitorder not in {"big", "little"}:
            raise RuntimeError(f"Unsupported bitorder={bitorder}")

        set_all_seeds(seed)
        model = capture_exact_loaded_deeplab(r05, r03, state)
        model = model.to(device)

        drops = set_source_deterministic_mode(model)
        if len(drops) != 1:
            raise RuntimeError(
                f"DeepLab seed={seed}: expected exactly 1 native nn.Dropout, got {drops}"
            )
        if float(drops[0]["p"]) != 0.5:
            raise RuntimeError(
                f"DeepLab seed={seed}: native dropout p={drops[0]['p']} expected=0.5"
            )

        parity_matches = 0

        iterator = range(EXPECTED_CASES)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=EXPECTED_CASES,
                desc=f"ADIC+MC DeepLab seed {seed}",
                unit="img",
                dynamic_ncols=True,
            )

        for i in iterator:
            mr = manifest.iloc[i]

            # Use a deterministic state-specific global stream; no target labels
            # or outcomes influence RNG.
            if i == 0:
                stochastic_seed = BASE_RNG_SEED + state_ord * 100000
                set_all_seeds(stochastic_seed)

            target = {
                "sample_id": str(mr["sample_id"]),
                "original_polypgen_sample_id": str(mr["original_polypgen_sample_id"]),
                "center": str(mr["center"]),
                "image_path": str(image_paths[i]),
                "image_raw_sha256": str(mr["image_raw_sha256"]).lower(),
            }

            x = build_exact_input(
                prefix_code=prefix_code,
                r05=r05,
                r03=r03,
                state=state,
                target=target,
                i=i,
                device=device,
            )

            set_source_deterministic_mode(model)
            with torch.no_grad():
                det_logits = extract_logits_tensor(model(x))
                det_p, det_hard = binary_probability_and_hard(det_logits)

            current_mask = det_hard[0].detach().cpu().numpy().astype(np.uint8)
            current_pack = pack_mask(current_mask, bitorder)
            frozen = frozen_pack[i]

            if not np.array_equal(current_pack, frozen):
                mismatch_pixels = int(np.count_nonzero(
                    np.unpackbits(current_pack, bitorder=bitorder)[:MASK_PIXELS]
                    != np.unpackbits(frozen, bitorder=bitorder)[:MASK_PIXELS]
                ))
                current_fg = int(current_mask.sum())
                frozen_fg = int(idx.iloc[i]["source_foreground_pixels"])
                raise RuntimeError(
                    "DETERMINISTIC_SOURCE_PARITY_FAIL\n"
                    f"seed={seed} row={i} sample={mr['sample_id']}\n"
                    f"mismatch_pixels={mismatch_pixels}\n"
                    f"current_fg={current_fg} frozen_fg={frozen_fg}\n"
                    "STOP: no ADIC/MC score lock will be written."
                )

            parity_matches += 1

            metrics = compute_adic_mc(
                model=model,
                x=x,
                deterministic_p=det_p,
                deterministic_hard=det_hard,
            )

            score_rows.append({
                "sample_id": str(mr["sample_id"]),
                "original_polypgen_sample_id": str(mr["original_polypgen_sample_id"]),
                "center": str(mr["center"]),
                "image_path": str(mr["image_path"]),
                "image_raw_sha256": str(mr["image_raw_sha256"]).lower(),
                "model_family": FAMILY,
                "model_state_id": str(state["model_state_id"]),
                "training_seed": seed,
                "checkpoint_sha256": str(state["checkpoint_sha256"]),
                "n_dropout": N_DROPOUT,
                "native_dropout_p": 0.5,
                **metrics,
            })

            del x, det_logits, det_p, det_hard

        if parity_matches != EXPECTED_CASES:
            raise RuntimeError(
                f"DeepLab seed={seed}: parity matches={parity_matches}, "
                f"expected={EXPECTED_CASES}"
            )

        state_meta.append({
            "training_seed": seed,
            "model_state_id": state["model_state_id"],
            "checkpoint_sha256": state["checkpoint_sha256"],
            "deterministic_source_parity_matches": parity_matches,
            "deterministic_source_parity_mismatches": 0,
            "native_nn_dropout_count": 1,
            "native_dropout_p": 0.5,
            "stochastic_stream_seed": BASE_RNG_SEED + state_ord * 100000,
            "frozen_prediction_npz_sha256": sha256_file(
                exact_state_artifact_paths(root, seed)["npz"]
            ),
            "frozen_prediction_index_sha256": sha256_file(
                exact_state_artifact_paths(root, seed)["index"]
            ),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[6/7] Validate complete 3-state score panel...")
    scores = pd.DataFrame(score_rows)

    if len(scores) != EXPECTED_ROWS:
        raise RuntimeError(f"Score rows={len(scores)}, expected={EXPECTED_ROWS}")
    if scores["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("Unique sample count mismatch.")
    if scores["training_seed"].nunique() != EXPECTED_STATES:
        raise RuntimeError("DeepLab state count mismatch.")
    if scores.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate sample/model-state scores.")

    numeric_cols = [
        "adic_quality",
        "adic_harm_risk",
        "adi_mean_foreground_dice",
        "adic_confidence_weight",
        "mc_predictive_entropy_risk",
        "mc_probability_variance",
        "dropout_dice_min",
        "dropout_dice_max",
    ]
    for c in numeric_cols:
        vals = scores[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Non-finite score column: {c}")

    # Internal orientation identity only; no target outcomes are inspected.
    if not np.array_equal(
        scores["adic_harm_risk"].to_numpy(dtype=float),
        -scores["adic_quality"].to_numpy(dtype=float),
    ):
        raise RuntimeError("ADIC risk orientation identity failed.")

    print("[7/7] Serialize and SHA-lock PRE-GT scores...")
    score_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK.csv"
    scores.to_csv(score_path, index=False)

    summary = (
        scores.groupby(["model_family", "training_seed"], as_index=False)
        .agg(
            rows=("sample_id", "size"),
            adic_quality_mean=("adic_quality", "mean"),
            adic_quality_std=("adic_quality", "std"),
            mc_entropy_mean=("mc_predictive_entropy_risk", "mean"),
            mc_entropy_std=("mc_predictive_entropy_risk", "std"),
        )
    )
    summary_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    lock = {
        "status": "PASS",
        "decision": "PASS_R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK",
        "version": VERSION,
        "dataset": "PolypGen",
        "scope": {
            "model_family": FAMILY,
            "states": EXPECTED_STATES,
            "rows": EXPECTED_ROWS,
            "physical_samples": EXPECTED_CASES,
            "pranet_included": False,
            "pranet_exclusion_reason": "no native torch.nn.Dropout",
            "segformer_included": False,
        },
        "information_boundary": {
            "training": False,
            "tta": False,
            "target_gt_loading": False,
            "harm_outcome_loading": False,
            "threshold_tuning": False,
            "target_recalibration": False,
            "target_score_reversal": False,
        },
        "official_tegda_anchor": {
            "repo": "https://github.com/HiLab-git/TEGDA",
            "file": "code/sota/adic2d.py",
            "n_iter": N_DROPOUT,
            "native_module_type": "torch.nn.Dropout",
            "dropout_p_rewritten": False,
            "new_dropout_injected": False,
            "binary_instantiation": (
                "single foreground class reduction of official class-specific "
                "ADIC confidence-weighted dropout-vs-current Dice"
            ),
        },
        "rng": {
            "base_seed": BASE_RNG_SEED,
            "strategy": (
                "state-specific deterministic global torch RNG stream; "
                "fixed sorted state and manifest traversal"
            ),
        },
        "authoritative_code": {
            "r05": str(R05_REL).replace("\\", "/"),
            "r05_sha256": R05_SHA,
            "r03": str(R03_REL).replace("\\", "/"),
            "r03_sha256": R03_SHA,
            "r10": str(R10_REL).replace("\\", "/"),
            "r10_sha256": R10_SHA,
            "extracted_input_prefix": prefix_path.name,
            "extracted_input_prefix_sha256": sha256_file(prefix_path),
        },
        "states": state_meta,
        "score_table": score_path.name,
        "score_table_sha256": sha256_file(score_path),
        "summary_table": summary_path.name,
        "summary_table_sha256": sha256_file(summary_path),
        "next": (
            "JOIN_ONLY_FROZEN_DEEPLAB3_HARM_OUTCOMES_AND_COMPARE_"
            "ADIC_MC_VS_SAFETTA"
        ),
    }

    lock_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "===== R17A POLYPGEN DEEPLAB3 ADIC + MC PRE-GT SCORE LOCK =====",
        f"Version: {VERSION}",
        "Dataset: PolypGen",
        "Family: DeepLabV3-R50",
        f"Rows: {len(scores)}",
        f"Physical samples: {scores['sample_id'].nunique()}",
        "States: 3/3",
        f"Dropout passes: {N_DROPOUT}",
        "Native nn.Dropout count/state: 1",
        "Native dropout p: 0.5",
        f"Deterministic SOURCE parity matches: {EXPECTED_ROWS}",
        "Deterministic SOURCE parity mismatches: 0",
        f"Score table SHA256: {sha256_file(score_path)}",
        "",
        "Information boundary:",
        "  Training: NO",
        "  TTA: NO",
        "  Target GT loading: NO",
        "  HARM/outcome loading: NO",
        "  Threshold tuning: NO",
        "",
        "Scores:",
        "  ADIC quality: higher = better current prediction quality",
        "  ADIC harm risk: -ADIC quality; higher = higher HARM risk",
        "  MC predictive entropy: higher = higher HARM risk",
        "",
        "GATE=PASS_R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK",
        "NEXT=JOIN_ONLY_FROZEN_DEEPLAB3_HARM_OUTCOMES_AND_COMPARE_ADIC_MC_VS_SAFETTA",
    ]

    report_path = out_dir / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    for line in report_lines[-18:]:
        print(line)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
