#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1.py

SafeTTA R17A — exact frozen-model instantiation audit for TEGDA-ADIC on the
PolypGen matched-6 panel.

WHY THIS AUDIT IS NEEDED
------------------------
The preceding code-search feasibility audit returned candidate native-dropout
routes, but its PraNet evidence came only from the audit script itself. That is
not sufficient evidence that the actual frozen PraNet model contains a native
torch.nn.Dropout module.

This script therefore instantiates the *actual historical frozen models* via
the authoritative R05D3 family runners and intercepts the exact checkpoint
load before any target inference is performed.

OFFICIAL TEGDA FIDELITY RULE
----------------------------
The official TEGDA ADIC implementation switches existing `nn.Dropout` modules
to train mode for stochastic inference. R17A therefore counts exact
`torch.nn.Dropout` modules separately from Dropout2d/AlphaDropout and does NOT
inject new dropout or rewrite dropout probability p.

INFORMATION BOUNDARY
--------------------
Training: NO
Target inference: NO
Target GT loading: NO
HARM/outcome loading: NO
Model modification: NO
Checkpoint modification: NO

The only side effect is writing this audit's own report files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

VERSION = "2026-09-09-Q1-R17A-ADIC-MATCHED6-EXACT-MODEL-INSTANTIATION-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1")

R05_REL = Path(r"code\Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py")
R05_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"

R03_REL = Path(r"code\Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py")
R03_SHA = "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"

R10_REL = Path(r"code\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2.py")
R10_SHA = "4cae02c30e82781c6e7d3b74b6a312974789213a9cc59a44ca99bbfb94405708"

CHECKPOINT_MANIFEST_REL = Path(
    r"outputs\Q1_R10L3A_polypgen_source_a1_prediction_lock_fix2_v1"
    r"\checkpoint_manifest.csv"
)

ALLOWED = {
    "DeepLabV3-R50": {20260817, 20260818, 20260819},
    "PraNet": {20260817, 20260818, 20260819},
}

RUNNER_NAME = {
    "DeepLabV3-R50": "run_deeplab_state",
    "PraNet": "run_pranet_state",
}


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
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\npath={path}\nexpected={expected}\nactual={actual}"
        )
    return actual


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
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


def resolve_checkpoint_path(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([root / p, root / str(raw).replace("/", "\\")])
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Checkpoint path cannot be resolved: {raw}")


class _CapturedModel(Exception):
    pass


def module_inventory(model: nn.Module) -> dict:
    exact_dropout = []
    dropout2d = []
    alpha_dropout = []
    all_dropout_like = []
    bn = []
    ln = []

    for name, m in model.named_modules():
        row = {
            "name": name,
            "type": type(m).__name__,
        }
        if type(m) is nn.Dropout:
            row["p"] = float(m.p)
            exact_dropout.append(row)
            all_dropout_like.append(row)
        elif type(m) is nn.Dropout2d:
            row["p"] = float(m.p)
            dropout2d.append(row)
            all_dropout_like.append(row)
        elif type(m) is nn.AlphaDropout:
            row["p"] = float(m.p)
            alpha_dropout.append(row)
            all_dropout_like.append(row)

        if isinstance(m, nn.BatchNorm2d):
            bn.append({"name": name, "type": type(m).__name__})
        if isinstance(m, nn.LayerNorm):
            ln.append({"name": name, "type": type(m).__name__})

    return {
        "model_class": type(model).__name__,
        "module_count": sum(1 for _ in model.modules()),
        "exact_nn_dropout_count": len(exact_dropout),
        "exact_nn_dropout": exact_dropout,
        "dropout2d_count": len(dropout2d),
        "dropout2d": dropout2d,
        "alpha_dropout_count": len(alpha_dropout),
        "alpha_dropout": alpha_dropout,
        "all_dropout_like_count": len(all_dropout_like),
        "batchnorm2d_count": len(bn),
        "layernorm_count": len(ln),
    }


def invoke_runner_until_exact_checkpoint_loaded(
    runner,
    state: dict,
    r03,
    expected_state_keys: set[str],
) -> dict:
    captured = {}

    original = nn.Module.load_state_dict

    def wrapped(self, state_dict, *args, **kwargs):
        result = original(self, state_dict, *args, **kwargs)

        try:
            keys = set(str(k) for k in state_dict.keys())
        except Exception:
            keys = set()

        if keys == expected_state_keys:
            captured["inventory"] = module_inventory(self)
            captured["load_result_missing"] = list(
                getattr(result, "missing_keys", [])
            )
            captured["load_result_unexpected"] = list(
                getattr(result, "unexpected_keys", [])
            )
            raise _CapturedModel()

        return result

    nn.Module.load_state_dict = wrapped
    try:
        sig = inspect.signature(runner)
        kwargs = {}
        for name in sig.parameters:
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
            elif sig.parameters[name].default is inspect._empty:
                raise RuntimeError(
                    f"Unsupported required runner argument: {name} "
                    f"for {runner.__name__}{sig}"
                )

        try:
            runner(**kwargs)
        except _CapturedModel:
            pass
        except Exception as e:
            # A failure before exact checkpoint load means model construction
            # could not be audited faithfully.
            if "inventory" not in captured:
                raise RuntimeError(
                    f"{runner.__name__} failed before exact checkpoint capture: "
                    f"{type(e).__name__}: {e}"
                ) from e
            # If the capture already happened, later failure is irrelevant;
            # normally _CapturedModel aborts immediately.
    finally:
        nn.Module.load_state_dict = original

    if "inventory" not in captured:
        raise RuntimeError(
            f"Exact target checkpoint load was never observed in {runner.__name__}."
        )

    return captured


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A ADIC MATCHED-6 EXACT MODEL INSTANTIATION AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Training: NO")
    print("Target inference: NO")
    print("Target GT loading: NO")
    print("HARM/outcome loading: NO")
    print("Model modification: NO")
    print()

    r05_path = root / R05_REL
    r03_path = root / R03_REL
    r10_path = root / R10_REL

    print("[1/4] Validate authoritative scripts...")
    validate_sha(r05_path, R05_SHA, "R05D3")
    validate_sha(r03_path, R03_SHA, "R03")
    validate_sha(r10_path, R10_SHA, "R10L3A")

    print("[2/4] Load exact matched-6 checkpoint manifest...")
    manifest_path = root / CHECKPOINT_MANIFEST_REL
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    import pandas as pd
    mf = pd.read_csv(manifest_path, low_memory=False)

    required = {
        "model_family",
        "training_seed",
        "model_state_id",
        "checkpoint",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(mf.columns))
    if missing:
        raise RuntimeError(f"Checkpoint manifest missing columns: {missing}")

    keep = []
    for row in mf.itertuples(index=False):
        family = str(row.model_family)
        seed = int(row.training_seed)
        if family in ALLOWED and seed in ALLOWED[family]:
            keep.append({
                "model_family": family,
                "training_seed": seed,
                "model_state_id": str(row.model_state_id),
                "checkpoint": str(row.checkpoint),
                "checkpoint_sha256": str(row.checkpoint_sha256).lower(),
            })

    if len(keep) != 6:
        raise RuntimeError(f"Matched checkpoint states={len(keep)}, expected=6")

    keep = sorted(keep, key=lambda r: (r["model_family"], r["training_seed"]))

    print("[3/4] Import authoritative runners and instantiate exact models...")
    r05 = import_from_path(r05_path, "q1_r05d3_r17a_adic_instantiation")
    r03 = import_from_path(r03_path, "q1_r03_r17a_adic_instantiation")

    audit_rows = []
    family_details: Dict[str, List[dict]] = {
        "DeepLabV3-R50": [],
        "PraNet": [],
    }

    for st in keep:
        family = st["model_family"]
        seed = int(st["training_seed"])
        runner_name = RUNNER_NAME[family]

        if not hasattr(r05, runner_name):
            raise RuntimeError(f"Authoritative R05 missing runner: {runner_name}")
        runner = getattr(r05, runner_name)

        ckpt = resolve_checkpoint_path(root, st["checkpoint"])
        actual_sha = validate_sha(
            ckpt,
            st["checkpoint_sha256"],
            f"{family} seed={seed} checkpoint",
        )

        payload = robust_torch_load(ckpt)
        state_dict, container = extract_state_dict(payload)
        state_keys = set(str(k) for k in state_dict.keys())

        captured = invoke_runner_until_exact_checkpoint_loaded(
            runner=runner,
            state=st,
            r03=r03,
            expected_state_keys=state_keys,
        )

        inv = captured["inventory"]
        exact_ps = [x["p"] for x in inv["exact_nn_dropout"]]

        row = {
            "model_family": family,
            "training_seed": seed,
            "model_state_id": st["model_state_id"],
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": actual_sha,
            "checkpoint_state_container": container,
            "checkpoint_state_key_count": len(state_keys),
            "model_class": inv["model_class"],
            "module_count": inv["module_count"],
            "exact_nn_dropout_count": inv["exact_nn_dropout_count"],
            "exact_nn_dropout_p": "|".join(str(x) for x in exact_ps),
            "dropout2d_count": inv["dropout2d_count"],
            "alpha_dropout_count": inv["alpha_dropout_count"],
            "all_dropout_like_count": inv["all_dropout_like_count"],
            "batchnorm2d_count": inv["batchnorm2d_count"],
            "layernorm_count": inv["layernorm_count"],
            "strict_load_missing_count": len(captured["load_result_missing"]),
            "strict_load_unexpected_count": len(captured["load_result_unexpected"]),
            "official_tegda_native_nn_dropout_supported": (
                inv["exact_nn_dropout_count"] > 0
            ),
            "exact_nn_dropout_modules_json": json.dumps(
                inv["exact_nn_dropout"],
                ensure_ascii=False,
            ),
            "other_dropout_like_modules_json": json.dumps(
                inv["dropout2d"] + inv["alpha_dropout"],
                ensure_ascii=False,
            ),
        }

        audit_rows.append(row)
        family_details[family].append(row)

        del payload, state_dict

        print(
            f"  {family} seed={seed}: class={inv['model_class']} "
            f"nn.Dropout={inv['exact_nn_dropout_count']} "
            f"Dropout2d={inv['dropout2d_count']} "
            f"AlphaDropout={inv['alpha_dropout_count']}"
        )

    print("[4/4] Freeze family-level ADIC support decision...")

    family_summary = []
    for family in ["DeepLabV3-R50", "PraNet"]:
        rows = family_details[family]
        counts = sorted(set(int(r["exact_nn_dropout_count"]) for r in rows))
        ps = sorted(set(r["exact_nn_dropout_p"] for r in rows))

        if len(counts) != 1:
            decision = "BLOCKED_DROPOUT_SCHEMA_DIFFERS_BY_SEED"
        elif counts[0] <= 0:
            decision = "UNSUPPORTED_NO_NATIVE_NN_DROPOUT"
        else:
            decision = "SUPPORTED_NATIVE_NN_DROPOUT"

        family_summary.append({
            "model_family": family,
            "states": len(rows),
            "exact_nn_dropout_counts_by_seed_unique": counts,
            "exact_nn_dropout_p_schema_unique": ps,
            "decision": decision,
        })

    supported = [
        r["model_family"]
        for r in family_summary
        if r["decision"] == "SUPPORTED_NATIVE_NN_DROPOUT"
    ]

    if set(supported) == {"DeepLabV3-R50", "PraNet"}:
        gate = "PASS_ADIC_MATCHED6_BOTH_FAMILIES_NATIVE_NN_DROPOUT"
        next_step = (
            "BUILD_MATCHED6_PREGT_ADIC_MC_SCORE_LOCK_WITH_SOURCE_PARITY_GATE"
        )
    elif supported:
        gate = "PARTIAL_ADIC_SUPPORT_FAMILY_SPECIFIC"
        next_step = (
            "BUILD_ADIC_MC_SCORE_LOCK_ONLY_FOR_SUPPORTED_FAMILY_AND_KEEP_"
            "UNSUPPORTED_FAMILY_EXCLUDED"
        )
    else:
        gate = "STOP_ADIC_NO_FAITHFUL_NATIVE_NN_DROPOUT"
        next_step = "DO_NOT_APPROXIMATE_ADIC_OR_INJECT_DROPOUT"

    write_csv(
        out_dir / "R17A_ADIC_EXACT_MODEL_INSTANTIATION.csv",
        audit_rows,
    )
    write_csv(
        out_dir / "R17A_ADIC_FAMILY_SUPPORT_DECISION.csv",
        family_summary,
    )

    summary = {
        "version": VERSION,
        "information_boundary": {
            "training": False,
            "target_inference": False,
            "target_gt_loading": False,
            "harm_outcome_loading": False,
            "model_modification": False,
        },
        "official_tegda_fidelity": {
            "required_native_type": "torch.nn.Dropout",
            "new_dropout_injected": False,
            "dropout_p_rewritten": False,
        },
        "families": family_summary,
        "gate": gate,
        "next": next_step,
    }

    (out_dir / "R17A_ADIC_EXACT_MODEL_INSTANTIATION_LOCK.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A ADIC MATCHED-6 EXACT MODEL INSTANTIATION REPORT =====",
        f"Version: {VERSION}",
        "",
        "INFORMATION BOUNDARY:",
        "Training: NO",
        "Target inference: NO",
        "Target GT loading: NO",
        "HARM/outcome loading: NO",
        "Model modification: NO",
        "",
        "OFFICIAL TEGDA FIDELITY:",
        "- exact native torch.nn.Dropout only",
        "- no injected dropout",
        "- no dropout-p rewriting",
        "",
        "STATE INVENTORY:",
    ]

    for r in audit_rows:
        lines.append(
            f"{r['model_family']} seed={r['training_seed']} "
            f"class={r['model_class']} "
            f"nn.Dropout={r['exact_nn_dropout_count']} "
            f"p={r['exact_nn_dropout_p'] or 'NONE'} "
            f"Dropout2d={r['dropout2d_count']} "
            f"AlphaDropout={r['alpha_dropout_count']}"
        )

    lines += ["", "FAMILY DECISION:"]
    for r in family_summary:
        lines.append(
            f"{r['model_family']}: {r['decision']} | "
            f"native_nn_dropout_counts={r['exact_nn_dropout_counts_by_seed_unique']} | "
            f"p_schema={r['exact_nn_dropout_p_schema_unique']}"
        )

    lines += [
        "",
        f"GATE={gate}",
        f"NEXT={next_step}",
        "",
        "IMPORTANT:",
        "- A family with zero exact nn.Dropout modules is not eligible for a",
        "  faithful TEGDA-ADIC baseline under the official implementation.",
        "- Do not inject dropout to rescue an unsupported family.",
    ]

    report = out_dir / "R17A_ADIC_EXACT_MODEL_INSTANTIATION_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"GATE={gate}")
    print(f"NEXT={next_step}")
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
