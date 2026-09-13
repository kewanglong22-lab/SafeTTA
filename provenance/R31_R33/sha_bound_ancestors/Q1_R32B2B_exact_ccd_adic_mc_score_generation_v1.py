#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32B2B
Exact NeoPolyp CCD / ADIC / MC-dropout SOURCE Reliability Score Generation

Scientific status
-----------------
POST-R31B3 retrospective matched published-baseline comparison.

IMPORTANT chronology:
- R31B3 HARM outcomes are already known historically.
- This script nevertheless has an outcome-blind execution boundary:
  it does NOT read R31B3 HARM/GT/outcome tables.
- Baseline algorithms, orientation and implementation semantics are inherited
  unchanged from the frozen R17A implementations bound in R32B2A.
- Scores are SHA-locked before R32B3 joins them to future-action HARM outcomes.

Why score all 1000 NeoPolyp SOURCE cases?
-----------------------------------------
R32B1's final matched comparison uses 800 cases x 3 DeepLab states.
CCD contains stochastic pixel subsampling driven by a fixed torch.Generator.
To avoid making that RNG stream depend on the later 800-case subset, R32B2B
first scores the complete frozen NeoPolyp SOURCE cohort:
    1000 cases x 3 DeepLab states = 3000 SOURCE model-cases
in exact R05D3 row order, and only then extracts:
    800 cases x 3 DeepLab states = 2400 common SOURCE model-cases.

Exact inherited baseline semantics
----------------------------------
SicTTA-CCD:
  - reuse the exact retained ccd_from_binary_logits implementation
  - RNG seed inherited from R17A full9 fix2: 20260908
  - one continuous CPU torch.Generator stream
  - higher CCD = higher HARM-risk orientation (frozen; no reversal)

TEGDA-ADIC / MC-dropout:
  - exact R17A fix2 compute_adic_mc()
  - N_DROPOUT = 10
  - only exact native nn.Dropout modules are enabled
  - DeepLab must have exactly one native nn.Dropout with p=0.5
  - ADIC quality = confidence_weight * mean dropout-vs-current Dice
  - ADIC HARM risk = -ADIC quality
  - MC risk = mean predictive entropy of the MC mean probability
  - no injected dropout, no modified p

SOURCE state identity:
  - exact DeepLab checkpoints inherited from retained R17A/R05D3
  - exact R05D3 DeepLab preprocessing
  - each deterministic hard mask must exactly match the already-frozen
    R05D3 SOURCE packed mask before its scores are accepted

No target calibration, threshold tuning, feature selection, score reversal,
new TTA action inference, HARM evaluation, GT decoding or external data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch


VERSION = "2026-09-12-R32B2B-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------
# Exact R32B2A / R32B1 bindings
# ---------------------------------------------------------------------

R32B2A_SCRIPT = CODE / "Q1_R32B2A_exact_baseline_execution_context_bundle_v1.py"
EXPECTED_R32B2A_SCRIPT_SHA256 = (
    "a9d62ed169c42d4de555c1806dd78250923e5b33676281962d593109b7fb8d80"
)

R32B2A_DIR = ROOT / "R32B2A_exact_baseline_execution_context_bundle_v1"
R32B2A_FINAL = R32B2A_DIR / "R32B2A_FINAL_LOCK.json"
EXPECTED_R32B2A_FINAL_SHA256 = (
    "7cbabecdfada42eddda34265086b4fb0c24e258489719924c8850deb41802082"
)
R32B2A_BUNDLE = R32B2A_DIR / "R32B2A_EXACT_EXECUTION_CONTEXT_BUNDLE.txt"
EXPECTED_R32B2A_BUNDLE_SHA256 = (
    "49b87566ad72caa5b670f5f83004de8799e3b26d4ff5ce04347ddfdb78e1a022"
)
R32B2A_MAPPING = R32B2A_DIR / "R32B2A_NEOPOLYP_800CASE_DEEPLAB_MAPPING.csv"

R32B1_DIR = ROOT / "R32B1_faithful_common_baseline_panel_lock_v1"
R32B1_FINAL = R32B1_DIR / "R32B1_FINAL_LOCK.json"
EXPECTED_R32B1_FINAL_SHA256 = (
    "ce305d32f77b2c72e5ca6589043aa6644542e35ee74fa7f41e68db45d4cd31e4"
)

# ---------------------------------------------------------------------
# Exact retained implementation scripts
# ---------------------------------------------------------------------

CCD_FULL9_SCRIPT = CODE / "Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1_fix2.py"
EXPECTED_CCD_FULL9_SHA256 = (
    "64f70b40fffbd72d2828cbfa910b05f322ba637caf4a73f42f048ba6da492da3"
)

ADIC_MC_SCRIPT = CODE / "Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2.py"
EXPECTED_ADIC_MC_SHA256 = (
    "009aaadc7b65b39b28b054cbfb5f9f8bf80fd7581e00df33eca96434505599c4"
)

R05D3_SCRIPT = CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA256 = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)

R05D3_STATE_DIR = (
    OUTPUTS
    / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"
    / "state_predictions"
)

# ---------------------------------------------------------------------
# Frozen scope
# ---------------------------------------------------------------------

FAMILY = "DeepLabV3-R50"
SEEDS = (20260817, 20260818, 20260819)
STATE_IDS = tuple(f"{FAMILY}::{s}" for s in SEEDS)

ALL_CASES = 1000
COMMON_CASES = 800
N_STATES = 3

EXPECTED_ALL_ROWS = ALL_CASES * N_STATES
EXPECTED_COMMON_ROWS = COMMON_CASES * N_STATES

EXPECTED_CCD_RNG_SEED = 20260908
EXPECTED_N_DROPOUT = 10
EXPECTED_NATIVE_DROPOUT_P = 0.5

DEFAULT_OUT = ROOT / "R32B2B_exact_ccd_adic_mc_score_generation_v1"

SCORE_COLUMNS = [
    "ccd_risk",
    "adic_quality",
    "adic_harm_risk",
    "adi_mean_foreground_dice",
    "adic_confidence_weight",
    "mc_predictive_entropy_risk",
    "mc_probability_variance",
    "dropout_dice_min",
    "dropout_dice_max",
]

FORBIDDEN_NAME_TOKENS = (
    "harm",
    "gt",
    "ground_truth",
    "dice_delta",
    "delta_dice",
    "r31b3_three_action",
)


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

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
            f"{label} SHA mismatch\n"
            f"expected={expected}\nobserved={got}\npath={path}"
        )
    return got


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def resolve_path(raw: str) -> Path:
    p = Path(str(raw))
    if p.is_absolute() and p.is_file():
        return p
    for c in [
        ROOT / p,
        ROOT / str(raw).replace("/", "\\"),
    ]:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Cannot resolve file: {raw}")


def no_forbidden_outcome_columns(df: pd.DataFrame, label: str) -> None:
    bad = []
    for c in df.columns:
        low = str(c).lower()
        if any(tok in low for tok in FORBIDDEN_NAME_TOKENS):
            bad.append(str(c))
    if bad:
        raise RuntimeError(
            f"{label} violates outcome-blind boundary; forbidden columns={bad}"
        )


# ---------------------------------------------------------------------
# Upstream verification
# ---------------------------------------------------------------------

def verify_upstream() -> Dict[str, Any]:
    require_sha(R32B2A_SCRIPT, EXPECTED_R32B2A_SCRIPT_SHA256, "R32B2A script")
    require_sha(R32B2A_FINAL, EXPECTED_R32B2A_FINAL_SHA256, "R32B2A final")
    require_sha(R32B2A_BUNDLE, EXPECTED_R32B2A_BUNDLE_SHA256, "R32B2A bundle")
    require_sha(R32B1_FINAL, EXPECTED_R32B1_FINAL_SHA256, "R32B1 final")

    require_sha(CCD_FULL9_SCRIPT, EXPECTED_CCD_FULL9_SHA256, "CCD fix2")
    require_sha(ADIC_MC_SCRIPT, EXPECTED_ADIC_MC_SHA256, "ADIC/MC fix2")
    require_sha(R05D3_SCRIPT, EXPECTED_R05D3_SHA256, "R05D3 fix1")

    final = json.loads(R32B2A_FINAL.read_text(encoding="utf-8"))
    if final.get("status") != (
        "PASS_R32B2A_EXACT_EXECUTION_CONTEXT_BINDING_COMPLETE"
    ):
        raise RuntimeError("R32B2A final status changed.")
    if final.get("decision") != "READY_FOR_R32B2B_EXACT_SCORE_RUNNER":
        raise RuntimeError("R32B2A was not READY for R32B2B.")
    if bool(final.get("baseline_inference", True)):
        raise RuntimeError("R32B2A inference boundary changed.")
    if bool(final.get("external_data_access", True)):
        raise RuntimeError("R32B2A external-access boundary changed.")

    mapping_sha = str(final.get("mapping_sha256", ""))
    if not R32B2A_MAPPING.is_file():
        raise FileNotFoundError(R32B2A_MAPPING)
    if mapping_sha and sha256_file(R32B2A_MAPPING) != mapping_sha:
        raise RuntimeError("R32B2A mapping SHA changed.")

    return final


# ---------------------------------------------------------------------
# Exact implementation introspection
# ---------------------------------------------------------------------

def ast_function_source(path: Path, function_name: str) -> Optional[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(path))
    lines = src.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                start = int(node.lineno)
                end = int(getattr(node, "end_lineno", node.lineno))
                return "\n".join(lines[start - 1:end])
    return None


def normalized_function_ast(path: Path, function_name: str) -> Optional[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return ast.dump(node, include_attributes=False)
    return None


def referenced_python_literals(path: Path) -> List[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    vals = re.findall(r"""["']([^"']+\.py)["']""", src)
    return vals


def discover_exact_ccd_helper() -> Tuple[Path, List[Path], str]:
    """
    First prefer a Python file literally referenced by the exact CCD full9
    fix2 script and defining ccd_from_binary_logits.

    If that does not resolve uniquely, scan CODE and require all discovered
    definitions to have one identical normalized AST semantics.
    """
    referenced = []
    for raw in referenced_python_literals(CCD_FULL9_SCRIPT):
        candidates = [
            Path(raw),
            CODE / raw,
            ROOT / raw,
        ]
        for p in candidates:
            if p.is_file() and ast_function_source(
                p, "ccd_from_binary_logits"
            ) is not None:
                referenced.append(p.resolve())

    uniq_ref = []
    seen = set()
    for p in referenced:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq_ref.append(p)

    if len(uniq_ref) == 1:
        p = uniq_ref[0]
        norm = normalized_function_ast(p, "ccd_from_binary_logits")
        assert norm is not None
        return p, uniq_ref, hashlib.sha256(norm.encode()).hexdigest()

    found = []
    for p in CODE.glob("*.py"):
        try:
            norm = normalized_function_ast(p, "ccd_from_binary_logits")
        except Exception:
            continue
        if norm is not None:
            found.append((p.resolve(), norm))

    if not found:
        raise RuntimeError(
            "Cannot locate retained ccd_from_binary_logits implementation."
        )

    groups: Dict[str, List[Path]] = {}
    for p, norm in found:
        sem_sha = hashlib.sha256(norm.encode()).hexdigest()
        groups.setdefault(sem_sha, []).append(p)

    if len(groups) != 1:
        detail = {
            k: [str(p) for p in v]
            for k, v in groups.items()
        }
        raise RuntimeError(
            "Multiple non-identical retained CCD helper semantics found; "
            f"refusing to guess: {detail}"
        )

    sem_sha, paths = next(iter(groups.items()))

    # Deterministic preference: R17A + CCD in filename, then lexicographic.
    paths = sorted(
        paths,
        key=lambda p: (
            0 if ("r17a" in p.name.lower() and "ccd" in p.name.lower()) else 1,
            str(p).lower(),
        ),
    )
    return paths[0], paths, sem_sha


def extract_stochastic_seed_binding(path: Path) -> Dict[str, Any]:
    """
    Recover the exact old R17A seed rule from source rather than inventing it.

    Finds:
      stochastic_seed = <expr>
    and a call whose argument contains stochastic_seed, preferably
      set_all_seeds(<expr_using_stochastic_seed>)
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(path))
    lines = src.splitlines()

    assign_expr = None
    assign_line = None
    call_exprs = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [
                t.id for t in node.targets
                if isinstance(t, ast.Name)
            ]
            if "stochastic_seed" in names:
                assign_expr = ast.unparse(node.value)
                assign_line = int(node.lineno)

        if isinstance(node, ast.Call):
            has_stochastic_name = any(
                isinstance(x, ast.Name) and x.id == "stochastic_seed"
                for x in ast.walk(node)
            )
            if not has_stochastic_name:
                continue

            try:
                fn = ast.unparse(node.func)
            except Exception:
                fn = ""

            arg_exprs = []
            for a in node.args:
                try:
                    arg_exprs.append(ast.unparse(a))
                except Exception:
                    pass

            call_exprs.append({
                "function": fn,
                "args": arg_exprs,
                "lineno": int(getattr(node, "lineno", -1)),
            })

    if assign_expr is None:
        raise RuntimeError(
            "Exact ADIC/MC script no longer contains stochastic_seed assignment."
        )

    preferred = [
        x for x in call_exprs
        if x["function"].endswith("set_all_seeds")
    ]
    selected = preferred[0] if len(preferred) == 1 else None

    if selected is None:
        # Fall back only if there is exactly one call using stochastic_seed.
        if len(call_exprs) == 1:
            selected = call_exprs[0]
        else:
            raise RuntimeError(
                "Cannot uniquely recover stochastic seed call from exact "
                f"ADIC/MC source. assignment={assign_expr}, calls={call_exprs}"
            )

    if len(selected["args"]) < 1:
        raise RuntimeError("Recovered stochastic seed call has no argument.")

    seed_arg_expr = selected["args"][0]

    start = max(1, int(assign_line) - 5)
    end = min(len(lines), int(selected["lineno"]) + 5)
    context = "\n".join(lines[start - 1:end])

    return {
        "assignment_expr": assign_expr,
        "assignment_line": assign_line,
        "seed_call_function": selected["function"],
        "seed_call_arg_expr": seed_arg_expr,
        "seed_call_line": selected["lineno"],
        "source_context": context,
    }


def eval_seed_rule(
    binding: Dict[str, Any],
    *,
    base_rng_seed: int,
    state_ord: int,
    i: int,
) -> int:
    env = {
        "BASE_RNG_SEED": int(base_rng_seed),
        "state_ord": int(state_ord),
        "i": int(i),
    }

    stochastic_seed = eval(
        str(binding["assignment_expr"]),
        {"__builtins__": {}},
        dict(env),
    )
    env["stochastic_seed"] = int(stochastic_seed)

    final_seed = eval(
        str(binding["seed_call_arg_expr"]),
        {"__builtins__": {}},
        dict(env),
    )
    return int(final_seed)


# ---------------------------------------------------------------------
# NeoPolyp all-1000 SOURCE asset discovery
# ---------------------------------------------------------------------

def load_common_ids() -> set[str]:
    d = pd.read_csv(
        R32B2A_MAPPING,
        dtype={"sample_id": str},
        low_memory=False,
    )
    no_forbidden_outcome_columns(d, "R32B2A mapping")

    ids = set(d["sample_id"].astype(str))
    if len(ids) != COMMON_CASES:
        raise RuntimeError(
            f"R32B2A common IDs={len(ids)} expected={COMMON_CASES}"
        )
    return ids


def state_index_path(seed: int) -> Path:
    p = R05D3_STATE_DIR / f"deeplabv3_r50_seed{seed}_index.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def discover_state_npz(seed: int) -> Path:
    index_p = state_index_path(seed)

    direct = index_p.with_name(
        index_p.name.replace("_index.csv", ".npz")
    )
    candidates = []
    if direct.is_file():
        candidates.append(direct)

    for p in R05D3_STATE_DIR.glob(f"*seed{seed}*.npz"):
        if p not in candidates:
            candidates.append(p)

    valid = []
    for p in candidates:
        try:
            with np.load(p, allow_pickle=False) as z:
                if {
                    "source_masks_packed",
                    "a1_masks_packed",
                }.issubset(set(z.files)):
                    src = np.asarray(z["source_masks_packed"])
                    if src.shape[0] == ALL_CASES:
                        valid.append(p)
        except Exception:
            pass

    if len(valid) != 1:
        raise RuntimeError(
            f"seed={seed}: expected exactly one 1000-case R05D3 state NPZ, "
            f"found={[str(p) for p in valid]}"
        )
    return valid[0]


def load_all_state_indexes() -> Dict[int, pd.DataFrame]:
    out = {}
    base_ids = None

    for seed in SEEDS:
        p = state_index_path(seed)
        d = pd.read_csv(
            p,
            dtype={
                "sample_id": str,
                "model_state_id": str,
                "model_family": str,
            },
            low_memory=False,
        )
        no_forbidden_outcome_columns(d, f"R05D3 index seed={seed}")

        required = {
            "sample_id",
            "image_path",
            "image_raw_sha256",
            "model_family",
            "model_state_id",
            "training_seed",
            "checkpoint_sha256",
        }
        missing = sorted(required.difference(d.columns))
        if missing:
            raise RuntimeError(
                f"seed={seed}: R05D3 index missing={missing}"
            )

        if len(d) != ALL_CASES:
            raise RuntimeError(
                f"seed={seed}: index rows={len(d)} expected={ALL_CASES}"
            )
        if d["sample_id"].nunique() != ALL_CASES:
            raise RuntimeError(f"seed={seed}: duplicate/missing sample IDs.")
        if set(d["model_family"].astype(str)) != {FAMILY}:
            raise RuntimeError(f"seed={seed}: family drift.")
        if set(d["model_state_id"].astype(str)) != {f"{FAMILY}::{seed}"}:
            raise RuntimeError(f"seed={seed}: state ID drift.")
        if set(pd.to_numeric(d["training_seed"]).astype(int)) != {seed}:
            raise RuntimeError(f"seed={seed}: training_seed drift.")

        if "row_index" in d.columns:
            ri = pd.to_numeric(d["row_index"]).astype(int).to_numpy()
            if not np.array_equal(np.sort(ri), np.arange(ALL_CASES)):
                raise RuntimeError(
                    f"seed={seed}: row_index is not exact 0..999."
                )
            d = d.assign(_row_index=ri).sort_values(
                "_row_index", kind="mergesort"
            ).drop(columns=["_row_index"]).reset_index(drop=True)
        else:
            # Exact CSV row order is the authoritative R05D3 packed-mask order.
            d = d.reset_index(drop=True)

        ids = d["sample_id"].astype(str).tolist()
        if base_ids is None:
            base_ids = ids
        elif ids != base_ids:
            raise RuntimeError(
                "R05D3 DeepLab state indexes do not share exact case order."
            )

        out[seed] = d

    return out


def verify_images_once(indexes: Dict[int, pd.DataFrame]) -> List[Path]:
    ref = indexes[SEEDS[0]].copy()

    # Ensure image path/hash identity is identical across states.
    for seed in SEEDS[1:]:
        d = indexes[seed]
        for col in ["sample_id", "image_path", "image_raw_sha256"]:
            if not np.array_equal(
                ref[col].astype(str).to_numpy(),
                d[col].astype(str).to_numpy(),
            ):
                raise RuntimeError(
                    f"Image mapping differs across DeepLab states: {col}"
                )

    paths = []
    for r in tqdm(
        ref.itertuples(index=False),
        total=len(ref),
        desc="R32B2B verify NeoPolyp RGB SHA256",
        unit="img",
        dynamic_ncols=True,
    ):
        p = resolve_path(str(r.image_path))
        got = sha256_file(p)
        exp = str(r.image_raw_sha256).lower()
        if got.lower() != exp:
            raise RuntimeError(
                f"NeoPolyp RGB SHA mismatch sample={r.sample_id}\n"
                f"expected={exp}\nobserved={got}\npath={p}"
            )
        paths.append(p)

    return paths


def load_frozen_source_packs() -> Dict[int, Tuple[Path, np.ndarray]]:
    out = {}
    for seed in SEEDS:
        p = discover_state_npz(seed)
        with np.load(p, allow_pickle=False) as z:
            src = np.asarray(z["source_masks_packed"], dtype=np.uint8)
        if src.shape[0] != ALL_CASES:
            raise RuntimeError(
                f"seed={seed}: SOURCE pack rows={src.shape[0]}"
            )
        out[seed] = (p, src)
    return out


# ---------------------------------------------------------------------
# Exact score execution
# ---------------------------------------------------------------------

def logits_to_binary_z(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected deterministic logits shape={tuple(logits.shape)}"
        )

    c = int(logits.shape[1])
    if c == 1:
        z = logits[0, 0]
    elif c == 2:
        z = logits[0, 1] - logits[0, 0]
    else:
        raise RuntimeError(f"Expected 1 or 2 channels, got {c}")

    arr = z.detach().float().cpu().numpy()
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise RuntimeError("Invalid binary source logit map for CCD.")
    return arr


def generate_scores(
    *,
    device: torch.device,
    ccd_mod,
    ccd_helper,
    adic_mod,
    r05,
    deep_helper,
    training,
    states: List[dict],
    indexes: Dict[int, pd.DataFrame],
    image_paths: Sequence[Path],
    source_packs: Dict[int, Tuple[Path, np.ndarray]],
    seed_binding: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Exact retained CCD stream.
    rng_seed = int(getattr(ccd_mod, "RNG_SEED"))
    if rng_seed != EXPECTED_CCD_RNG_SEED:
        raise RuntimeError(
            f"CCD RNG seed drift={rng_seed} expected={EXPECTED_CCD_RNG_SEED}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng_seed)

    if int(getattr(adic_mod, "N_DROPOUT")) != EXPECTED_N_DROPOUT:
        raise RuntimeError("ADIC/MC N_DROPOUT drift.")
    if int(getattr(adic_mod, "BASE_RNG_SEED")) != 20260909:
        raise RuntimeError("ADIC/MC BASE_RNG_SEED drift.")

    score_rows = []
    parity_rows = []

    for state_ord, state in enumerate(states):
        seed = int(state["training_seed"])
        state_id = str(state["model_state_id"])

        if seed not in SEEDS:
            raise RuntimeError(f"Unexpected state seed={seed}")
        if state_id != f"{FAMILY}::{seed}":
            raise RuntimeError(f"Unexpected state ID={state_id}")

        idx = indexes[seed]
        npz_path, frozen_pack = source_packs[seed]

        # Exact old R17A model capture route.
        adic_mod.set_all_seeds(seed)
        model = adic_mod.capture_exact_loaded_deeplab(
            r05,
            None,
            state,
        )
        model = model.to(device)

        drops = adic_mod.set_source_deterministic_mode(model)
        if len(drops) != 1:
            raise RuntimeError(
                f"{state_id}: expected exactly one native nn.Dropout, got {drops}"
            )
        if not math.isclose(
            float(drops[0]["p"]),
            EXPECTED_NATIVE_DROPOUT_P,
            rel_tol=0,
            abs_tol=0,
        ):
            raise RuntimeError(
                f"{state_id}: dropout p={drops[0]['p']} expected=0.5"
            )

        pbar = tqdm(
            range(ALL_CASES),
            total=ALL_CASES,
            desc=f"R32B2B CCD+ADIC+MC seed {seed}",
            unit="img",
            dynamic_ncols=True,
        )

        parity_match_count = 0

        for i in pbar:
            mr = idx.iloc[i]

            # Recover exact old R17A per-sample stochastic seed rule from
            # source. Deterministic SOURCE forward is unaffected by RNG,
            # but setting it here matches the old execution order.
            sample_seed = eval_seed_rule(
                seed_binding,
                base_rng_seed=int(getattr(adic_mod, "BASE_RNG_SEED")),
                state_ord=state_ord,
                i=i,
            )
            adic_mod.set_all_seeds(sample_seed)

            with Image.open(image_paths[i]) as im:
                native = im.convert("RGB")

            x = deep_helper.image_to_model_tensor(
                training,
                native,
            ).to(device, non_blocking=True)

            adic_mod.set_source_deterministic_mode(model)
            with torch.no_grad():
                det_logits = adic_mod.extract_logits_tensor(model(x))
                det_p, det_hard = adic_mod.binary_probability_and_hard(
                    det_logits
                )

            # Exact frozen SOURCE-state parity against R05D3.
            current_mask = (
                det_hard[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            packed = np.asarray(
                r05.pack_mask(current_mask),
                dtype=np.uint8,
            )
            frozen = frozen_pack[i]
            exact = bool(np.array_equal(packed, frozen))
            mismatch_pixels = 0

            if not exact:
                frozen_mask = r05.unpack_mask(frozen)
                mismatch_pixels = int(
                    np.count_nonzero(current_mask != frozen_mask)
                )
                raise RuntimeError(
                    f"SOURCE hard-mask parity failure "
                    f"state={state_id} row={i} sample={mr['sample_id']} "
                    f"mismatch_pixels={mismatch_pixels}"
                )

            parity_match_count += 1

            # CCD exact helper and exact continuous RNG stream.
            z_source = logits_to_binary_z(det_logits)
            ccd_risk = float(
                ccd_helper.ccd_from_binary_logits(
                    z_source,
                    generator,
                )
            )

            # Exact R17A ADIC + MC implementation.
            metrics = adic_mod.compute_adic_mc(
                model=model,
                x=x,
                deterministic_p=det_p,
                deterministic_hard=det_hard,
            )

            row = {
                "row_index": int(i),
                "sample_id": str(mr["sample_id"]),
                "image_path": str(mr["image_path"]),
                "image_raw_sha256": str(mr["image_raw_sha256"]).lower(),
                "model_family": FAMILY,
                "model_state_id": state_id,
                "training_seed": seed,
                "checkpoint_sha256": str(state["checkpoint_sha256"]),
                "n_dropout": EXPECTED_N_DROPOUT,
                "native_dropout_p": EXPECTED_NATIVE_DROPOUT_P,
                "stochastic_seed": int(sample_seed),
                "ccd_risk": ccd_risk,
                **metrics,
            }
            score_rows.append(row)

            parity_rows.append({
                "row_index": int(i),
                "sample_id": str(mr["sample_id"]),
                "model_state_id": state_id,
                "training_seed": seed,
                "source_npz": str(npz_path),
                "source_npz_sha256": sha256_file(npz_path),
                "packed_mask_exact": True,
                "mismatch_pixels": 0,
            })

            pbar.set_postfix(
                parity=parity_match_count,
                ccd=f"{ccd_risk:.4f}",
            )

            del (
                x,
                det_logits,
                det_p,
                det_hard,
                current_mask,
                packed,
                frozen,
                z_source,
            )

        pbar.close()

        if parity_match_count != ALL_CASES:
            raise RuntimeError(
                f"{state_id}: parity matches={parity_match_count}/{ALL_CASES}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(score_rows)
    parity = pd.DataFrame(parity_rows)

    if len(scores) != EXPECTED_ALL_ROWS:
        raise RuntimeError(
            f"All-source score rows={len(scores)} expected={EXPECTED_ALL_ROWS}"
        )
    if scores["sample_id"].nunique() != ALL_CASES:
        raise RuntimeError("All-source unique sample count mismatch.")
    if scores["model_state_id"].nunique() != N_STATES:
        raise RuntimeError("All-source state count mismatch.")
    if scores.duplicated(["sample_id", "model_state_id"]).any():
        raise RuntimeError("Duplicate SOURCE score row.")

    for c in SCORE_COLUMNS:
        if c not in scores.columns:
            raise RuntimeError(f"Missing score column={c}")
        vals = scores[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Non-finite score column={c}")

    # Frozen orientation identity; no outcomes inspected.
    if not np.array_equal(
        scores["adic_harm_risk"].to_numpy(dtype=float),
        -scores["adic_quality"].to_numpy(dtype=float),
    ):
        raise RuntimeError("ADIC HARM-risk orientation identity failed.")

    if len(parity) != EXPECTED_ALL_ROWS:
        raise RuntimeError("Parity row count mismatch.")
    if not parity["packed_mask_exact"].all():
        raise RuntimeError("Non-exact SOURCE hard-mask parity exists.")

    return scores, parity


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(
            ["model_family", "training_seed", "model_state_id"],
            as_index=False,
        )
        .agg(
            rows=("sample_id", "size"),
            ccd_mean=("ccd_risk", "mean"),
            ccd_std=("ccd_risk", "std"),
            adic_quality_mean=("adic_quality", "mean"),
            adic_quality_std=("adic_quality", "std"),
            adic_harm_risk_mean=("adic_harm_risk", "mean"),
            mc_entropy_mean=("mc_predictive_entropy_risk", "mean"),
            mc_entropy_std=("mc_predictive_entropy_risk", "std"),
            mc_variance_mean=("mc_probability_variance", "mean"),
            dropout_dice_min=("dropout_dice_min", "min"),
            dropout_dice_max=("dropout_dice_max", "max"),
        )
        .sort_values("training_seed", kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default="cuda",
        help="torch device; default cuda",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R32B2B Exact CCD / ADIC / MC SOURCE Score Generation")
    print("Version                       :", VERSION)
    print("Scientific status             : POST-R31B3 RETROSPECTIVE MATCHED BASELINE")
    print("All SOURCE cases scored       : 1000 x 3 = 3000")
    print("Frozen common subset          : 800 x 3 = 2400")
    print("R31B3 HARM/GT read            : NO")
    print("Target calibration            : NO")
    print("Post-hoc score reversal       : NO")
    print("New TTA-action inference      : NO")
    print("External cohort access        : NO")
    print("=" * 124)

    r32b2a_final = verify_upstream()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32B2B will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    # -----------------------------------------------------------------
    # Bind exact retained implementations before any inference.
    # -----------------------------------------------------------------
    ccd_mod = import_module(
        CCD_FULL9_SCRIPT,
        "r32b2b_ccd_full9_fix2",
    )
    adic_mod = import_module(
        ADIC_MC_SCRIPT,
        "r32b2b_adic_mc_fix2",
    )
    r05 = import_module(
        R05D3_SCRIPT,
        "r32b2b_r05d3_fix1",
    )

    if int(getattr(ccd_mod, "RNG_SEED")) != EXPECTED_CCD_RNG_SEED:
        raise RuntimeError("CCD RNG_SEED drift.")
    if int(getattr(adic_mod, "N_DROPOUT")) != EXPECTED_N_DROPOUT:
        raise RuntimeError("ADIC/MC N_DROPOUT drift.")

    ccd_helper_path, ccd_candidates, ccd_semantic_sha = (
        discover_exact_ccd_helper()
    )
    ccd_helper = import_module(
        ccd_helper_path,
        "r32b2b_exact_ccd_helper",
    )

    if not hasattr(ccd_helper, "ccd_from_binary_logits"):
        raise RuntimeError("Exact CCD helper missing required function.")

    seed_binding = extract_stochastic_seed_binding(ADIC_MC_SCRIPT)

    # Exact DeepLab preprocessor from R05D3.
    deep_helper_path = Path(getattr(r05, "DEEPLAB_HELPER"))
    if not deep_helper_path.is_absolute():
        deep_helper_path = ROOT / deep_helper_path
    if not deep_helper_path.is_file():
        raise FileNotFoundError(
            f"R05D3 DEEPLAB_HELPER not found: {deep_helper_path}"
        )

    deep_helper = import_module(
        deep_helper_path,
        "r32b2b_exact_deeplab_helper",
    )
    training = deep_helper.import_training_helper()

    states = adic_mod.load_deeplab_states(ROOT)
    if [int(s["training_seed"]) for s in states] != list(SEEDS):
        raise RuntimeError(
            f"DeepLab state seeds drift: {[s['training_seed'] for s in states]}"
        )
    if [str(s["model_state_id"]) for s in states] != list(STATE_IDS):
        raise RuntimeError(
            f"DeepLab state IDs drift: {[s['model_state_id'] for s in states]}"
        )

    indexes = load_all_state_indexes()
    image_paths = verify_images_once(indexes)
    source_packs = load_frozen_source_packs()
    common_ids = load_common_ids()

    # Common IDs must be an exact subset of the complete 1000-case cohort.
    all_ids = set(indexes[SEEDS[0]]["sample_id"].astype(str))
    if not common_ids.issubset(all_ids):
        raise RuntimeError("R32B1 common IDs are not a subset of NeoPolyp 1000.")

    # -----------------------------------------------------------------
    # PRE-INFERENCE input lock. Contains no HARM/GT.
    # -----------------------------------------------------------------
    helper_binding = {
        "ccd_helper_path": str(ccd_helper_path),
        "ccd_helper_sha256": sha256_file(ccd_helper_path),
        "ccd_helper_semantic_ast_sha256": ccd_semantic_sha,
        "ccd_equivalent_candidate_paths": [
            str(p) for p in ccd_candidates
        ],
        "deeplab_helper_path": str(deep_helper_path),
        "deeplab_helper_sha256": sha256_file(deep_helper_path),
        "stochastic_seed_binding": seed_binding,
    }

    input_lock = {
        "status": "PASS_R32B2B_PRE_INFERENCE_INPUT_LOCK",
        "version": VERSION,
        "scientific_status": (
            "POST_R31B3_RETROSPECTIVE_OUTCOME_BLIND_BASELINE_SCORE_GENERATION"
        ),
        "R32B2A_final_sha256": EXPECTED_R32B2A_FINAL_SHA256,
        "R32B2A_bundle_sha256": EXPECTED_R32B2A_BUNDLE_SHA256,
        "R32B1_final_sha256": EXPECTED_R32B1_FINAL_SHA256,
        "CCD_fix2_sha256": EXPECTED_CCD_FULL9_SHA256,
        "ADIC_MC_fix2_sha256": EXPECTED_ADIC_MC_SHA256,
        "R05D3_fix1_sha256": EXPECTED_R05D3_SHA256,
        "all_source_cases": ALL_CASES,
        "common_cases": COMMON_CASES,
        "states": list(STATE_IDS),
        "CCD_RNG_SEED": EXPECTED_CCD_RNG_SEED,
        "ADIC_MC_N_DROPOUT": EXPECTED_N_DROPOUT,
        "native_dropout_p": EXPECTED_NATIVE_DROPOUT_P,
        "helper_binding": helper_binding,
        "R31B3_HARM_GT_read": False,
        "target_calibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
    }

    input_lock_path = out / "R32B2B_PRE_INFERENCE_INPUT_LOCK.json"
    atomic_json(input_lock_path, input_lock)

    print("\nR32B2B EXACT IMPLEMENTATION BINDINGS")
    print("  CCD helper              :", ccd_helper_path)
    print("  CCD helper SHA256       :", sha256_file(ccd_helper_path))
    print("  CCD semantic AST SHA256 :", ccd_semantic_sha)
    print("  DeepLab helper          :", deep_helper_path)
    print("  DeepLab helper SHA256   :", sha256_file(deep_helper_path))
    print("  ADIC seed assignment    :", seed_binding["assignment_expr"])
    print(
        "  ADIC seed call          :",
        f"{seed_binding['seed_call_function']}"
        f"({seed_binding['seed_call_arg_expr']})",
    )
    print("  R31B3 HARM/GT read      : NO")

    # -----------------------------------------------------------------
    # Score generation.
    # -----------------------------------------------------------------
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    scores_all, parity = generate_scores(
        device=device,
        ccd_mod=ccd_mod,
        ccd_helper=ccd_helper,
        adic_mod=adic_mod,
        r05=r05,
        deep_helper=deep_helper,
        training=training,
        states=states,
        indexes=indexes,
        image_paths=image_paths,
        source_packs=source_packs,
        seed_binding=seed_binding,
    )

    # Extract common 800 only AFTER all 1000-case score generation.
    scores_common = scores_all[
        scores_all["sample_id"].astype(str).isin(common_ids)
    ].copy()

    if len(scores_common) != EXPECTED_COMMON_ROWS:
        raise RuntimeError(
            f"Common score rows={len(scores_common)} "
            f"expected={EXPECTED_COMMON_ROWS}"
        )
    if scores_common["sample_id"].nunique() != COMMON_CASES:
        raise RuntimeError("Common unique sample count mismatch.")
    if scores_common["model_state_id"].nunique() != N_STATES:
        raise RuntimeError("Common state count mismatch.")
    if scores_common.duplicated(
        ["sample_id", "model_state_id"]
    ).any():
        raise RuntimeError("Duplicate common SOURCE score row.")

    # Preserve full-cohort row order within each state.
    scores_all = scores_all.sort_values(
        ["training_seed", "row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    scores_common = scores_common.sort_values(
        ["training_seed", "row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    parity = parity.sort_values(
        ["training_seed", "row_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary_all = summary_table(scores_all)
    summary_common = summary_table(scores_common)

    all_path = out / "R32B2B_NEOPOLYP_ALL1000_DEEPLAB_SOURCE_RELIABILITY_SCORE_LOCK.csv"
    common_path = out / "R32B2B_NEOPOLYP_COMMON800_DEEPLAB_SOURCE_RELIABILITY_SCORE_LOCK.csv"
    parity_path = out / "R32B2B_SOURCE_HARD_MASK_EXACT_PARITY.csv"
    summary_all_path = out / "R32B2B_ALL1000_SCORE_SUMMARY.csv"
    summary_common_path = out / "R32B2B_COMMON800_SCORE_SUMMARY.csv"
    binding_path = out / "R32B2B_EXACT_IMPLEMENTATION_BINDING.json"

    atomic_csv(scores_all, all_path)
    atomic_csv(scores_common, common_path)
    atomic_csv(parity, parity_path)
    atomic_csv(summary_all, summary_all_path)
    atomic_csv(summary_common, summary_common_path)
    atomic_json(binding_path, helper_binding)

    # -----------------------------------------------------------------
    # POST-SCORE lock before any HARM evaluation.
    # -----------------------------------------------------------------
    score_lock = {
        "status": "PASS_R32B2B_SOURCE_RELIABILITY_SCORE_LOCK",
        "version": VERSION,
        "scientific_status": (
            "POST_R31B3_RETROSPECTIVE_OUTCOME_BLIND_BASELINE_SCORE_GENERATION"
        ),
        "all1000_rows": int(len(scores_all)),
        "common800_rows": int(len(scores_common)),
        "physical_cases_all": int(scores_all["sample_id"].nunique()),
        "physical_cases_common": int(scores_common["sample_id"].nunique()),
        "states": int(scores_all["model_state_id"].nunique()),
        "score_columns": SCORE_COLUMNS,
        "orientation": {
            "SicTTA_CCD": "higher ccd_risk = higher HARM risk",
            "TEGDA_ADIC": "adic_harm_risk = -adic_quality; higher = higher HARM risk",
            "MC_dropout": "higher mc_predictive_entropy_risk = higher HARM risk",
        },
        "all1000_score_sha256": sha256_file(all_path),
        "common800_score_sha256": sha256_file(common_path),
        "parity_sha256": sha256_file(parity_path),
        "all1000_summary_sha256": sha256_file(summary_all_path),
        "common800_summary_sha256": sha256_file(summary_common_path),
        "implementation_binding_sha256": sha256_file(binding_path),
        "source_hard_mask_exact_matches": int(
            parity["packed_mask_exact"].sum()
        ),
        "source_hard_mask_total_rows": int(len(parity)),
        "R31B3_HARM_GT_read": False,
        "target_calibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
        "next": "R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION",
    }

    score_lock_path = out / "R32B2B_SCORE_LOCK.json"
    atomic_json(score_lock_path, score_lock)

    final = {
        "status": "PASS_R32B2B_EXACT_CCD_ADIC_MC_SCORE_GENERATION_COMPLETE",
        "version": VERSION,
        "scientific_status": (
            "POST_R31B3_RETROSPECTIVE_OUTCOME_BLIND_BASELINE_SCORE_GENERATION"
        ),
        "R32B2A_final_sha256": EXPECTED_R32B2A_FINAL_SHA256,
        "pre_inference_input_lock_sha256": sha256_file(input_lock_path),
        "score_lock_sha256": sha256_file(score_lock_path),
        "all1000_score_sha256": sha256_file(all_path),
        "common800_score_sha256": sha256_file(common_path),
        "parity_sha256": sha256_file(parity_path),
        "R31B3_HARM_GT_read": False,
        "target_calibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
        "next": "R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION",
    }

    final_path = out / "R32B2B_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\n" + "=" * 124)
    print("R32B2B SOURCE HARD-MASK PARITY")
    print("=" * 124)
    print(
        f"  exact matches : {int(parity['packed_mask_exact'].sum())}"
        f"/{len(parity)}"
    )
    print("  mismatches    : 0")

    print("\n" + "=" * 124)
    print("R32B2B ALL-1000 SCORE SUMMARY")
    print("=" * 124)
    print(summary_all.to_string(index=False))

    print("\n" + "=" * 124)
    print("R32B2B COMMON-800 SCORE SUMMARY")
    print("=" * 124)
    print(summary_common.to_string(index=False))

    print("\nR32B2B PRE-HARM SCORE LOCK")
    print("  all1000 score SHA256 :", sha256_file(all_path))
    print("  common800 score SHA256:", sha256_file(common_path))
    print("  score lock SHA256    :", sha256_file(score_lock_path))
    print("  R31B3 HARM/GT read   : NO")

    print(
        "\nFINAL STATUS : "
        "PASS_R32B2B_EXACT_CCD_ADIC_MC_SCORE_GENERATION_COMPLETE"
    )
    print("Target calibration       : NO")
    print("Post-hoc score reversal  : NO")
    print("External cohort access   : NO")
    print("Final lock SHA256        :", sha256_file(final_path))
    print("Output                   :", out)
    print("NEXT                     : R32B3_MATCHED_THREE_ACTION_HARM_EVALUATION")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
