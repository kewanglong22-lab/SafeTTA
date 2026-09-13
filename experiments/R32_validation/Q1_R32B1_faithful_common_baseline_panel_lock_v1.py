#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R32B1
Freeze Faithful Common Published-Reliability Baseline Panel

Purpose
-------
Freeze the exact common panel on which the three retained published
reliability baselines can be compared fairly against SafeTTA:

  SicTTA-CCD
  TEGDA-ADIC
  MC-dropout

R32B0 found substantial retained R17A assets for all three methods, but the
historical R17A audit also established method-family restrictions:
  - ADIC / MC-dropout were faithfully executed on DeepLabV3-R50;
  - PraNet must not receive artificial dropout merely to make ADIC/MC possible;
  - regenerated SegFormer CCD states must not be mixed with historical HARM
    outcomes when SOURCE-state identity is inconsistent.

Therefore R32B1 freezes a MATCHED COMMON FAMILY:
  DeepLabV3-R50 only.

This is intentionally narrower than the full R32A 9-state mechanism panel.
R32A remains the full three-family / nine-state representation ablation.
R32B is the matched published-baseline comparison.

No CCD / ADIC / MC inference occurs here.
No GT-dependent score calibration occurs here.
No score direction is changed here.
No external cohort is accessed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


VERSION = "2026-09-12-R32B1-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

# ---------------------------------------------------------------------
# Upstream locks
# ---------------------------------------------------------------------

R32B0_SCRIPT = CODE / "Q1_R32B0_published_reliability_asset_comparability_audit_v1.py"
EXPECTED_R32B0_SCRIPT_SHA256 = (
    "10da7f6942ad845395d195c079df273a5bd8c90f1c85b240aa462aa1a3ef335d"
)

R32B0_DIR = ROOT / "R32B0_published_reliability_asset_comparability_audit_v1"
R32B0_FINAL = R32B0_DIR / "R32B0_FINAL_LOCK.json"
EXPECTED_R32B0_FINAL_SHA256 = (
    "a3ce6f403335ab59cc62867d5881d30cf39fe27fc1fbe1e899e98416eb4a6b77"
)
R32B0_ASSETS = R32B0_DIR / "R32B0_R17A_RETAINED_ASSET_INVENTORY.csv"
R32B0_SUPPORT = R32B0_DIR / "R32B0_BASELINE_ASSET_SUPPORT_SUMMARY.csv"

R32A1_DIR = ROOT / "R32A1_three_action_loao_comparison_execution_v1_fix1"
R32A1_FINAL = R32A1_DIR / "R32A1_FIX1_FINAL_LOCK.json"
EXPECTED_R32A1_FINAL_SHA256 = (
    "0822cf8c6db035e836dcc6d3f67ae4b0e9d58622e72ba1086edf60fa90e8873b"
)

R31B3_DIR = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"
R31B3_TABLE = R31B3_DIR / "R31B3_THREE_ACTION_MODELCASE_TABLE.csv"
R31B3_FINAL = R31B3_DIR / "R31B3_FINAL_LOCK.json"
EXPECTED_R31B3_FINAL_SHA256 = (
    "eb91d991752c50e71b1993e4ee0a50e0b9e9cdb35fffdd58b36553e0cebaaadc"
)

DEFAULT_OUT = ROOT / "R32B1_faithful_common_baseline_panel_lock_v1"

# ---------------------------------------------------------------------
# Frozen R32B1 scope
# ---------------------------------------------------------------------

COMMON_FAMILY = "DeepLabV3-R50"
EXPECTED_COMMON_STATES = 3
EXPECTED_CASES = 800
ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
EXPECTED_COMMON_ROWS = (
    EXPECTED_CASES * EXPECTED_COMMON_STATES * len(ACTIONS)
)

BASELINES = ("SicTTA-CCD", "TEGDA-ADIC", "MC-dropout")
SAFETTA_REFERENCE = "SHARED_TRANSITION_Q66_DSEM64"

# Exact retained R17A evidence highlighted by R32B0.
KEY_R17A_ASSETS = {
    "CCD_full9_preGT_scores": (
        ROOT
        / "outputs"
        / "Q1_R17A_CCD_polypgen_full9_preGT_score_lock_v1_fix2"
        / "R17A_POLYPGEN_CCD_FULL9_PREGT_SCORE_LOCK.csv"
    ),
    "CCD_6state_preGT_scores": (
        ROOT
        / "outputs"
        / "Q1_R17A_CCD_polypgen_6state_preGT_score_lock_v1"
        / "R17A_POLYPGEN_CCD_6STATE_PREGT_SCORE_LOCK.csv"
    ),
    "ADIC_MC_deeplab3_preGT_scores": (
        ROOT
        / "outputs"
        / "Q1_R17A_ADIC_MC_deeplab3_preGT_score_lock_v1_fix2"
        / "R17A_POLYPGEN_DEEPLAB3_ADIC_MC_PREGT_SCORE_LOCK.csv"
    ),
    "ADIC_exact_model_instantiation": (
        ROOT
        / "outputs"
        / "Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1"
        / "R17A_ADIC_EXACT_MODEL_INSTANTIATION.csv"
    ),
    "ADIC_family_support_decision": (
        ROOT
        / "outputs"
        / "Q1_R17A_ADIC_matched6_exact_model_instantiation_audit_v1"
        / "R17A_ADIC_FAMILY_SUPPORT_DECISION.csv"
    ),
    "DeepLab3_fourway_matched_panel": (
        ROOT
        / "outputs"
        / "Q1_R17A_polypgen_deeplab3_fourway_postGT_evaluation_v1"
        / "R17A_POLYPGEN_DEEPLAB3_FOURWAY_MATCHED_PANEL.csv"
    ),
}

SCORE_KEYWORDS = (
    "score",
    "risk",
    "uncert",
    "entropy",
    "ccd",
    "adic",
    "dropout",
    "confidence",
    "quality",
)

ID_KEYWORDS = (
    "sample_id",
    "image_id",
    "image",
    "case_id",
    "model_state",
    "model_state_id",
    "state",
    "family",
    "seed",
)


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
            f"{label} SHA mismatch\nexpected={expected}\n"
            f"observed={got}\npath={path}"
        )
    return got


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


def verify_upstream() -> Dict[str, Any]:
    require_sha(R32B0_SCRIPT, EXPECTED_R32B0_SCRIPT_SHA256, "R32B0 script")
    require_sha(R32B0_FINAL, EXPECTED_R32B0_FINAL_SHA256, "R32B0 final")
    require_sha(R32A1_FINAL, EXPECTED_R32A1_FINAL_SHA256, "R32A1 fix1 final")
    require_sha(R31B3_FINAL, EXPECTED_R31B3_FINAL_SHA256, "R31B3 final")

    b0 = json.loads(R32B0_FINAL.read_text(encoding="utf-8"))
    a1 = json.loads(R32A1_FINAL.read_text(encoding="utf-8"))

    if b0.get("status") != (
        "PASS_R32B0_PUBLISHED_RELIABILITY_ASSET_AUDIT_COMPLETE"
    ):
        raise RuntimeError("R32B0 final status changed.")

    if a1.get("status") != (
        "PASS_R32A1_FIX1_THREE_ACTION_LOAO_COMPARISON_COMPLETE"
    ):
        raise RuntimeError("R32A1 fix1 final status changed.")

    if bool(b0.get("baseline_inference", True)):
        raise RuntimeError("R32B0 unexpectedly reports baseline inference.")
    if bool(b0.get("target_recalibration", True)):
        raise RuntimeError("R32B0 target-recalibration guard changed.")
    if bool(b0.get("post_hoc_score_reversal", True)):
        raise RuntimeError("R32B0 score-direction guard changed.")
    if bool(b0.get("external_data_access", True)):
        raise RuntimeError("R32B0 external-access guard changed.")

    if not R32B0_ASSETS.is_file():
        raise FileNotFoundError(R32B0_ASSETS)
    if sha256_file(R32B0_ASSETS) != str(b0.get("asset_inventory_sha256")):
        raise RuntimeError("R32B0 asset inventory SHA changed.")

    if not R32B0_SUPPORT.is_file():
        raise FileNotFoundError(R32B0_SUPPORT)
    if sha256_file(R32B0_SUPPORT) != str(b0.get("support_summary_sha256")):
        raise RuntimeError("R32B0 support summary SHA changed.")

    if not R31B3_TABLE.is_file():
        raise FileNotFoundError(R31B3_TABLE)

    return {"R32B0": b0, "R32A1": a1}


def verify_support_summary() -> pd.DataFrame:
    d = pd.read_csv(R32B0_SUPPORT, low_memory=False)

    required = {
        "method",
        "matching_assets",
        "python_scripts",
        "csv_assets",
        "row_level_score_candidates",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R32B0 support summary missing={missing}")

    for method in BASELINES:
        g = d[d["method"] == method]
        if len(g) != 1:
            raise RuntimeError(
                f"Expected one R32B0 support row for {method}; got {len(g)}"
            )
        if int(g.iloc[0]["matching_assets"]) <= 0:
            raise RuntimeError(f"No retained assets found for {method}.")
        if int(g.iloc[0]["python_scripts"]) <= 0:
            raise RuntimeError(f"No retained implementation scripts for {method}.")

    return d


def load_common_panel() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    d = pd.read_csv(
        R31B3_TABLE,
        dtype={
            "sample_id": str,
            "model_state_id": str,
            "model_family": str,
            "action": str,
        },
        low_memory=False,
    )

    required = {
        "sample_id",
        "model_state_id",
        "model_family",
        "action",
        "r31b3_harm_label",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"R31B3 table missing={missing}")

    common = d[d["model_family"] == COMMON_FAMILY].copy()

    if common["sample_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError(
            f"{COMMON_FAMILY} physical cases={common['sample_id'].nunique()} "
            f"expected={EXPECTED_CASES}"
        )
    if common["model_state_id"].nunique() != EXPECTED_COMMON_STATES:
        raise RuntimeError(
            f"{COMMON_FAMILY} states={common['model_state_id'].nunique()} "
            f"expected={EXPECTED_COMMON_STATES}"
        )
    if set(common["action"].unique()) != set(ACTIONS):
        raise RuntimeError(
            f"{COMMON_FAMILY} action set={sorted(common['action'].unique())}"
        )
    if len(common) != EXPECTED_COMMON_ROWS:
        raise RuntimeError(
            f"common panel rows={len(common)} expected={EXPECTED_COMMON_ROWS}"
        )
    if common.duplicated(
        ["sample_id", "model_state_id", "action"]
    ).any():
        raise RuntimeError("Duplicate common-panel case/state/action row.")

    state_manifest = (
        common[
            [
                "model_state_id",
                "model_family",
            ]
        ]
        .drop_duplicates()
        .sort_values("model_state_id", kind="mergesort")
        .reset_index(drop=True)
    )

    action_summary = (
        common.groupby(
            ["model_state_id", "action"],
            as_index=False,
        )
        .agg(
            rows=("sample_id", "size"),
            physical_cases=("sample_id", "nunique"),
            harm_count=("r31b3_harm_label", "sum"),
            harm_rate=("r31b3_harm_label", "mean"),
        )
        .sort_values(
            ["model_state_id", "action"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    info = {
        "family": COMMON_FAMILY,
        "physical_cases": int(common["sample_id"].nunique()),
        "model_states": int(common["model_state_id"].nunique()),
        "state_ids": state_manifest["model_state_id"].tolist(),
        "actions": sorted(common["action"].unique().tolist()),
        "action_rows": int(len(common)),
    }

    return common, action_summary, info


def asset_schema(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "sha256": "",
            "rows": None,
            "columns": "",
            "score_like_columns": "",
            "id_like_columns": "",
        }

    try:
        d = pd.read_csv(path, low_memory=False)
        cols = [str(c) for c in d.columns]
        lowcols = [c.lower() for c in cols]

        score_cols = [
            c
            for c, lc in zip(cols, lowcols)
            if any(tok in lc for tok in SCORE_KEYWORDS)
        ]
        id_cols = [
            c
            for c, lc in zip(cols, lowcols)
            if any(tok in lc for tok in ID_KEYWORDS)
        ]

        return {
            "exists": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(d)),
            "columns": " | ".join(cols),
            "score_like_columns": " | ".join(score_cols),
            "id_like_columns": " | ".join(id_cols),
        }
    except Exception as exc:
        return {
            "exists": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": None,
            "columns": "",
            "score_like_columns": "",
            "id_like_columns": "",
            "read_error": repr(exc),
        }


def key_asset_inventory() -> pd.DataFrame:
    rows = []
    for label, path in KEY_R17A_ASSETS.items():
        info = asset_schema(path)
        info["asset_label"] = label
        rows.append(info)

    out = pd.DataFrame(rows)

    # Core evidence needed to freeze this common panel must exist.
    required_labels = {
        "CCD_full9_preGT_scores",
        "ADIC_MC_deeplab3_preGT_scores",
        "ADIC_exact_model_instantiation",
        "ADIC_family_support_decision",
        "DeepLab3_fourway_matched_panel",
    }

    for label in required_labels:
        g = out[out["asset_label"] == label]
        if len(g) != 1 or not bool(g.iloc[0]["exists"]):
            raise RuntimeError(f"Required retained R17A asset missing: {label}")

    return out


def implementation_candidates() -> pd.DataFrame:
    d = pd.read_csv(R32B0_ASSETS, low_memory=False)

    if len(d) == 0:
        raise RuntimeError("R32B0 retained asset inventory is empty.")

    joined = (
        d["relative_path"].astype(str)
        + " "
        + d["filename_hits"].fillna("").astype(str)
        + " "
        + d["content_hits"].fillna("").astype(str)
    ).str.lower()

    rows = []

    method_tokens = {
        "SicTTA-CCD": ("sict", "ccd"),
        "TEGDA-ADIC": ("tegd", "adic"),
        "MC-dropout": ("mc-drop", "mc_dropout", "mcdrop", "dropout"),
    }

    for method, tokens in method_tokens.items():
        mask = d["suffix"].eq(".py")
        method_mask = pd.Series(False, index=d.index)

        for tok in tokens:
            method_mask = method_mask | joined.str.contains(
                re.escape(tok),
                regex=True,
            )

        g = d[mask & method_mask].copy()

        # Deterministic evidence ranking only; R32B1 does NOT choose one script.
        g["method"] = method
        g["name_has_R17A"] = (
            g["relative_path"].str.lower().str.contains("r17a").astype(int)
        )
        g["name_has_preGT"] = (
            g["relative_path"].str.lower().str.contains("pregt").astype(int)
        )
        g["name_has_score"] = (
            g["relative_path"].str.lower().str.contains("score").astype(int)
        )
        g["candidate_rank"] = (
            100 * g["name_has_R17A"]
            + 20 * g["name_has_preGT"]
            + 10 * g["name_has_score"]
            + pd.to_numeric(g.get("priority", 0), errors="coerce").fillna(0)
        )

        keep = [
            "method",
            "relative_path",
            "path",
            "sha256",
            "candidate_rank",
            "filename_hits",
            "content_hits",
        ]
        rows.append(
            g.sort_values(
                ["candidate_rank", "relative_path"],
                ascending=[False, True],
                kind="mergesort",
            )[keep].head(25)
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=" * 124)
    print("SafeTTA R32B1 Freeze Faithful Common Published-Baseline Panel")
    print("Version                     :", VERSION)
    print("Common family               :", COMMON_FAMILY)
    print("CCD / ADIC / MC inference   : NO")
    print("GT-dependent calibration    : NO")
    print("Post-hoc score reversal     : NO")
    print("External cohort access      : NO")
    print("=" * 124)

    upstream = verify_upstream()
    support = verify_support_summary()
    common, action_summary, panel_info = load_common_panel()
    key_assets = key_asset_inventory()
    impl_candidates = implementation_candidates()

    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(
            f"Output exists; R32B1 will not overwrite: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)

    # Freeze only identifiers + labels needed for later matched evaluation;
    # no score generation and no score/outcome join occurs here.
    common_manifest = common[
        [
            "sample_id",
            "model_state_id",
            "model_family",
            "action",
            "r31b3_harm_label",
        ]
    ].copy().sort_values(
        ["sample_id", "model_state_id", "action"],
        kind="mergesort",
    )

    manifest_path = out / "R32B1_DEEPLAB3_COMMON_PANEL_MANIFEST.csv"
    action_summary_path = out / "R32B1_DEEPLAB3_STATE_ACTION_SUMMARY.csv"
    key_asset_path = out / "R32B1_KEY_R17A_ASSET_BINDINGS.csv"
    impl_path = out / "R32B1_IMPLEMENTATION_CANDIDATES.csv"

    atomic_csv(common_manifest, manifest_path)
    atomic_csv(action_summary, action_summary_path)
    atomic_csv(key_assets, key_asset_path)
    atomic_csv(impl_candidates, impl_path)

    protocol = {
        "status": "LOCKED_R32B1_FAITHFUL_COMMON_BASELINE_PANEL",
        "version": VERSION,
        "scientific_role": (
            "MATCHED PUBLISHED-RELIABILITY BASELINE COMPARISON; "
            "narrower than full 9-state R32A mechanism panel"
        ),
        "upstream": {
            "R32B0_final_sha256": EXPECTED_R32B0_FINAL_SHA256,
            "R32A1_fix1_final_sha256": EXPECTED_R32A1_FINAL_SHA256,
            "R31B3_final_sha256": EXPECTED_R31B3_FINAL_SHA256,
        },
        "common_panel": panel_info,
        "published_baselines": list(BASELINES),
        "SafeTTA_reference": SAFETTA_REFERENCE,
        "comparison_semantics": {
            "baseline_score_scope": (
                "one SOURCE-reliability score per physical sample x model state; "
                "same frozen score is evaluated against each candidate action's "
                "future HARM outcome"
            ),
            "baseline_action_specific_training": False,
            "target_HARM_recalibration": False,
            "target_threshold_tuning": False,
            "post_hoc_score_reversal": False,
            "feature_selection": False,
            "score_orientation": (
                "must reuse exact orientation/semantics frozen in retained R17A "
                "implementation; R32B2 must bind this before new score generation"
            ),
        },
        "family_policy": {
            "included": [COMMON_FAMILY],
            "excluded": {
                "PraNet": (
                    "Do not inject artificial dropout solely to create faithful "
                    "ADIC/MC-dropout support."
                ),
                "SegFormer-B0": (
                    "Do not mix regenerated CCD SOURCE states with historical "
                    "HARM outcomes if state identity is inconsistent."
                ),
            },
        },
        "actions": list(ACTIONS),
        "metrics": [
            "AUROC",
            "AUPRC",
            "AUPRC_LIFT=AUPRC/HARM_PREVALENCE",
        ],
        "reporting": {
            "per_action": True,
            "macro_across_three_actions": True,
            "state_pooled": True,
            "per_state_secondary": True,
            "paired_physical_case_bootstrap": (
                "deferred until all frozen baseline scores are generated"
            ),
        },
        "prohibitions": [
            "no baseline score fitting using R31B3 HARM outcomes",
            "no target score calibration",
            "no post-hoc score reversal",
            "no baseline-specific favorable subset",
            "no artificial dropout injection into unsupported architectures",
            "no inconsistent regenerated SOURCE-state/HARM pairing",
            "no external cohort access",
        ],
        "key_R17A_assets_sha256": {
            str(r["asset_label"]): str(r["sha256"])
            for _, r in key_assets.iterrows()
            if bool(r["exists"])
        },
        "common_manifest_sha256": sha256_file(manifest_path),
        "next": "R32B2_EXACT_BASELINE_IMPLEMENTATION_BINDING_AND_SCORE_GENERATION",
    }

    protocol_path = out / "R32B1_COMMON_PANEL_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)

    final = {
        "status": "PASS_R32B1_FAITHFUL_COMMON_BASELINE_PANEL_LOCK_COMPLETE",
        "version": VERSION,
        "common_family": COMMON_FAMILY,
        "physical_cases": panel_info["physical_cases"],
        "model_states": panel_info["model_states"],
        "state_ids": panel_info["state_ids"],
        "actions": panel_info["actions"],
        "action_rows": panel_info["action_rows"],
        "baseline_inference": False,
        "target_recalibration": False,
        "post_hoc_score_reversal": False,
        "external_data_access": False,
        "common_manifest_sha256": sha256_file(manifest_path),
        "key_asset_bindings_sha256": sha256_file(key_asset_path),
        "implementation_candidates_sha256": sha256_file(impl_path),
        "protocol_sha256": sha256_file(protocol_path),
        "next": "R32B2_EXACT_BASELINE_IMPLEMENTATION_BINDING_AND_SCORE_GENERATION",
    }

    final_path = out / "R32B1_FINAL_LOCK.json"
    atomic_json(final_path, final)

    print("\nR32B1 COMMON PANEL")
    print("  family          :", panel_info["family"])
    print("  physical cases  :", panel_info["physical_cases"])
    print("  model states    :", panel_info["model_states"])
    print("  state ids       :", panel_info["state_ids"])
    print("  actions         :", panel_info["actions"])
    print("  action rows     :", panel_info["action_rows"])

    print("\nR32B1 BASELINE ASSET SUPPORT")
    print(
        support[
            support["method"].isin(BASELINES)
        ].to_string(index=False)
    )

    print("\nR32B1 KEY R17A ASSET BINDINGS")
    print(
        key_assets[
            [
                "asset_label",
                "exists",
                "rows",
                "score_like_columns",
                "id_like_columns",
            ]
        ].to_string(index=False)
    )

    print("\nR32B1 IMPLEMENTATION CANDIDATES (TOP 10 EACH)")
    if len(impl_candidates):
        for method in BASELINES:
            g = impl_candidates[
                impl_candidates["method"] == method
            ].head(10)
            print(f"\n[{method}]")
            if len(g):
                print(
                    g[
                        [
                            "relative_path",
                            "candidate_rank",
                            "filename_hits",
                        ]
                    ].to_string(index=False)
                )
            else:
                print("  NONE")
    else:
        print("  NONE")

    print("\nFINAL STATUS : PASS_R32B1_FAITHFUL_COMMON_BASELINE_PANEL_LOCK_COMPLETE")
    print("Baseline inference         : NO")
    print("Target recalibration       : NO")
    print("Post-hoc score reversal    : NO")
    print("External cohort access     : NO")
    print("Common manifest SHA256     :", sha256_file(manifest_path))
    print("Protocol SHA256            :", sha256_file(protocol_path))
    print("Final lock SHA256          :", sha256_file(final_path))
    print("Output                     :", out)
    print("NEXT                       : R32B2_EXACT_BASELINE_IMPLEMENTATION_BINDING_AND_SCORE_GENERATION")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
