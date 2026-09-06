#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(r"F:\MEDSEG_SAFETTA")

R11C_DIR = ROOT / "outputs" / "Q1_R11C_neopolyp_tent2_prediction_lock_fix1_v1"
R11C_LOCK = R11C_DIR / "R11C_NEOPOLYP_TENT2_PREDICTION_LOCK.json"
R11C_MANIFEST = R11C_DIR / "R11C_model_case_a2_prediction_manifest.csv"
EXPECTED_R11C_SHA = "4744e04de406649f70b7cec3773803e9a1514c8b82d2feef761dd562d68a55eb"
EXPECTED_R11C_DECISION = (
    "NEOPOLYP_A2_TENT_2STEP_PREDICTIONS_LOCKED_FOR_ACTION_INTENSITY_EVALUATION"
)

R05D4_SCRIPT = ROOT / "code" / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1.py"
R05D4_DIR = ROOT / "outputs" / "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
R05D4_LOCK = R05D4_DIR / "Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
R05D4_OUTCOMES = R05D4_DIR / "model_case_outcomes.csv"

R10K2B_PRED = (
    ROOT / "outputs"
    / "Q1_R10K2B_dinov2_image_mask_LOMO_evaluation_fix2_v1"
    / "R10K2B_target_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "Q1_R11D_tent2_frozen_safety_transfer_evaluation_fix4_v1"

CASES = 1000
STATES = 9
MODEL_CASES = 9000
MASK_H = MASK_W = 352
PACKED_BYTES = 15488
BITORDER = "little"

HARM_THR = -0.02
BENEFIT_THR = 0.02
PPV_PREV = 0.01
TARGET_RECALL = 0.90

FAMILIES = ("DeepLabV3-R50", "PraNet", "SegFormer-B0")
SEEDS = (20260816, 20260817, 20260818, 20260819, 20260820)
METHODS = (
    "M2_backbone",
    "ImageCLS_PCA64",
    "CondDINO_PCA64",
    "M2_plus_CondDINO_PCA64",
)
PRIMARY = "M2_plus_CondDINO_PCA64"
BASELINE = "M2_backbone"

BOOTSTRAPS = 2000
BOOT_SEED = 20260826
DECISION_YES = "TENT2_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED"
DECISION_NO = "TENT2_FROZEN_SAFETY_RANKING_TRANSFER_NOT_SUPPORTED"


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
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def unpack_mask(packed):
    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Packed mask shape={p.shape}")
    return np.unpackbits(
        p, count=MASK_H * MASK_W, bitorder=BITORDER
    ).reshape(MASK_H, MASK_W).astype(np.uint8, copy=False)


def outcome(delta):
    if delta <= HARM_THR:
        return "HARM"
    if delta >= BENEFIT_THR:
        return "BENEFIT"
    return "NEUTRAL"


def threshold_for_recall(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return math.nan
    k = min(max(int(np.ceil(TARGET_RECALL * len(pos))), 1), len(pos))
    return float(pos[k - 1])


def metric(y, p, thr, sample_weight=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    thr = np.asarray(thr, dtype=float)
    w = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if np.sum(w * (y == 1)) <= 0 or np.sum(w * (y == 0)) <= 0:
        return None

    auc = float(roc_auc_score(y, p, sample_weight=w))
    ap = float(average_precision_score(y, p, sample_weight=w))
    pred = p >= thr
    tp = float(np.sum(w * (pred & (y == 1))))
    fp = float(np.sum(w * (pred & (y == 0))))
    tn = float(np.sum(w * ((~pred) & (y == 0))))
    fn = float(np.sum(w * ((~pred) & (y == 1))))
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    denom = recall * PPV_PREV + fpr * (1 - PPV_PREV)
    ppv1 = recall * PPV_PREV / denom if denom > 0 else math.nan
    return {"AUROC": auc, "AUPRC": ap, "Recall": recall, "FPR": fpr, "PPV1pct": ppv1}


def oracle_fpr90(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    t = threshold_for_recall(y, p)
    pred = p >= t
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum((~pred) & (y == 0)))
    return fp / (fp + tn)


def verify_r11c():
    if not R11C_LOCK.exists():
        raise FileNotFoundError(R11C_LOCK)
    actual = sha256_file(R11C_LOCK)
    if actual.lower() != EXPECTED_R11C_SHA.lower():
        raise RuntimeError(f"R11C lock SHA mismatch: {actual}")
    lock = json.loads(R11C_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision") != EXPECTED_R11C_DECISION:
        raise RuntimeError(f"R11C decision changed: {lock.get('decision')}")
    if (
        int(lock.get("target_cases", -1)) != CASES
        or int(lock.get("model_states", -1)) != STATES
        or int(lock.get("model_case_rows", -1)) != MODEL_CASES
        or int(lock.get("steps", -1)) != 2
        or int(lock.get("source_parity_mismatches", -1)) != 0
    ):
        raise RuntimeError("R11C cardinality/action/parity changed")
    # R11C stores the information-boundary audit as a separately frozen
    # artifact, not nested inside the main lock JSON. Validate that artifact
    # against the SHA recorded by the R11C lock, then inspect its fields.
    boundary_name = "R11C_information_boundary_audit.json"
    boundary_path = R11C_DIR / boundary_name
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)

    boundary_sha_expected = lock.get("artifacts", {}).get(boundary_name)
    if not isinstance(boundary_sha_expected, str) or not boundary_sha_expected:
        raise RuntimeError(
            "R11C lock does not contain the frozen information-boundary SHA."
        )
    boundary_sha_actual = sha256_file(boundary_path)
    if boundary_sha_actual.lower() != boundary_sha_expected.lower():
        raise RuntimeError(
            "R11C information-boundary artifact SHA mismatch: "
            f"{boundary_sha_actual} != {boundary_sha_expected}"
        )

    ib = json.loads(boundary_path.read_text(encoding="utf-8"))
    required_false = (
        "r05d3_a1_mask_values_read",
        "neopolyp_gt_pixels_decoded",
        "safety_scores_read",
        "polypgen_data_read",
    )
    for key in required_false:
        if key not in ib:
            raise RuntimeError(f"R11C boundary field missing: {key}")
        if bool(ib[key]):
            raise RuntimeError(f"R11C boundary failed: {key}")

    if int(ib.get("source_parity_mismatches", -1)) != 0:
        raise RuntimeError("R11C boundary reports SOURCE parity mismatch.")
    if bool(ib.get("hyperparameter_sweep", True)):
        raise RuntimeError("R11C boundary reports a hyperparameter sweep.")

    if not R11C_MANIFEST.exists():
        raise FileNotFoundError(R11C_MANIFEST)

    manifest_sha_expected = lock.get("artifacts", {}).get(R11C_MANIFEST.name)
    if not isinstance(manifest_sha_expected, str) or not manifest_sha_expected:
        raise RuntimeError("R11C lock missing frozen manifest SHA.")
    manifest_sha_actual = sha256_file(R11C_MANIFEST)
    if manifest_sha_actual.lower() != manifest_sha_expected.lower():
        raise RuntimeError(
            "R11C manifest SHA mismatch: "
            f"{manifest_sha_actual} != {manifest_sha_expected}"
        )

    return actual


def load_gt_and_r05d4():
    if not R05D4_LOCK.exists() or not R05D4_OUTCOMES.exists():
        raise FileNotFoundError("R05D4 artifacts missing")
    lock = json.loads(R05D4_LOCK.read_text(encoding="utf-8"))
    if int(lock.get("target_cases", -1)) != CASES:
        raise RuntimeError("R05D4 target count changed")
    if not bool(lock.get("target_gt_pixels_revealed", False)):
        raise RuntimeError("R05D4 historical GT reveal not confirmed")

    mod = import_module(R05D4_SCRIPT, "q1_r11d_r05d4")
    _, _, _, manifest_path, _, gt_rule = mod.validate_upstream()
    mod.validate_gt_rule(gt_rule)
    gt_manifest = mod.load_d1_manifest(manifest_path)
    gt_masks, _ = mod.reveal_gt(gt_manifest)

    d4 = pd.read_csv(R05D4_OUTCOMES, low_memory=False)
    if len(d4) != MODEL_CASES:
        raise RuntimeError("R05D4 outcomes rows !=9000")
    return mod, d4, gt_masks



def resolve_r11c_state_asset(raw_path: str, expected_kind: str) -> Path:
    """
    R11C wrote absolute state-asset paths while its output directory still
    ended in ``__building`` and then atomically renamed that directory to the
    final R11C directory.

    Deterministic technical relocation only:
      1) use the recorded path if it exists;
      2) otherwise require its parent to be exactly the known R11C
         ``__building/state_predictions`` directory;
      3) relocate only the basename into final
         ``R11C_DIR/state_predictions``;
      4) require the relocated file to exist.

    No globbing, recursive search, newest-file selection, content-based
    selection, or filename guessing is allowed.
    """
    recorded = Path(str(raw_path))
    if recorded.exists():
        return recorded

    expected_old_parent = Path(str(R11C_DIR) + "__building") / "state_predictions"

    if str(recorded.parent).lower() != str(expected_old_parent).lower():
        raise FileNotFoundError(
            "Missing R11C state asset is not under the known atomic-build "
            f"parent: kind={expected_kind} recorded={recorded}"
        )

    relocated = R11C_DIR / "state_predictions" / recorded.name
    if not relocated.exists():
        raise FileNotFoundError(
            f"Relocated R11C {expected_kind} not found: {relocated}"
        )

    return relocated


def audit_r11c_manifest_relocation():
    man = pd.read_csv(
        R11C_MANIFEST,
        usecols=[
            "state_prediction_npz",
            "state_index_csv",
            "state_lock_json",
        ],
        low_memory=False,
    )

    rows = []
    for col, kind in (
        ("state_prediction_npz", "prediction_npz"),
        ("state_index_csv", "state_index_csv"),
        ("state_lock_json", "state_lock_json"),
    ):
        vals = sorted(man[col].astype(str).unique())
        direct = 0
        relocated = 0
        for value in vals:
            p = Path(value)
            if p.exists():
                direct += 1
            else:
                resolve_r11c_state_asset(value, kind)
                relocated += 1

        rows.append({
            "artifact_kind": kind,
            "unique_recorded_paths": len(vals),
            "direct_existing_paths": direct,
            "stale_building_paths_relocated": relocated,
        })

    return pd.DataFrame(rows)


def build_a2_panel(r05d4, d4, gt_masks):
    man = pd.read_csv(R11C_MANIFEST, low_memory=False)
    # R11C's frozen manifest schema uses `row_index`, inherited directly
    # from R11C INDEX_FIELDS. It does NOT use the R05D3-style name
    # `state_row_index`. This is an upstream-schema naming difference only.
    req = [
        "row_index","sample_id","model_family","model_state_id","training_seed",
        "checkpoint_sha256","state_prediction_npz","state_index_csv","state_lock_json",
    ]
    miss = [c for c in req if c not in man.columns]
    if miss:
        raise RuntimeError(f"R11C manifest missing: {miss}")
    if len(man) != MODEL_CASES:
        raise RuntimeError("R11C manifest rows !=9000")

    rows = []
    for state_id, g in man.groupby("model_state_id", sort=True):
        npzs = g["state_prediction_npz"].astype(str).unique()
        idxs = g["state_index_csv"].astype(str).unique()
        locks = g["state_lock_json"].astype(str).unique()
        if len(npzs) != 1 or len(idxs) != 1 or len(locks) != 1:
            raise RuntimeError(f"{state_id}: ambiguous state assets")
        npz_path = resolve_r11c_state_asset(
            npzs[0], "prediction_npz"
        )
        index_path = resolve_r11c_state_asset(
            idxs[0], "state_index_csv"
        )
        lock_path = resolve_r11c_state_asset(
            locks[0], "state_lock_json"
        )

        side = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            side.get("decision") != "STATE_COMPLETE"
            or side.get("action") != "A2_TENT_2STEP"
            or int(side.get("steps", -1)) != 2
            or int(side.get("source_parity_mismatches", -1)) != 0
        ):
            raise RuntimeError(f"{state_id}: invalid state lock")

        if sha256_file(npz_path).lower() != str(side["prediction_npz_sha256"]).lower():
            raise RuntimeError(f"{state_id}: prediction SHA mismatch")
        if sha256_file(index_path).lower() != str(side["index_csv_sha256"]).lower():
            raise RuntimeError(f"{state_id}: index CSV SHA mismatch")

        # Exact R11C row-index contract:
        # one state has exactly row_index=0..999, each once.
        g = g.copy()
        g["row_index"] = pd.to_numeric(g["row_index"], errors="raise").astype(int)
        if len(g) != CASES:
            raise RuntimeError(f"{state_id}: manifest state rows={len(g)}")
        if sorted(g["row_index"].tolist()) != list(range(CASES)):
            raise RuntimeError(
                f"{state_id}: row_index is not an exact 0..{CASES-1} permutation"
            )
        if g["sample_id"].astype(str).nunique() != CASES:
            raise RuntimeError(f"{state_id}: sample_id is not unique")

        # Cross-check the global manifest against the separately SHA-locked
        # per-state index CSV. This avoids any implicit row-order inference.
        idx_df = pd.read_csv(index_path, low_memory=False)
        if "row_index" not in idx_df.columns or "sample_id" not in idx_df.columns:
            raise RuntimeError(f"{state_id}: state index missing row_index/sample_id")
        idx_df["row_index"] = pd.to_numeric(
            idx_df["row_index"], errors="raise"
        ).astype(int)
        if len(idx_df) != CASES:
            raise RuntimeError(f"{state_id}: state index rows={len(idx_df)}")

        audit_left = (
            g[["row_index", "sample_id"]]
            .assign(sample_id=lambda x: x["sample_id"].astype(str))
            .sort_values("row_index", kind="mergesort")
            .reset_index(drop=True)
        )
        audit_right = (
            idx_df[["row_index", "sample_id"]]
            .assign(sample_id=lambda x: x["sample_id"].astype(str))
            .sort_values("row_index", kind="mergesort")
            .reset_index(drop=True)
        )
        if not audit_left.equals(audit_right):
            raise RuntimeError(
                f"{state_id}: global manifest and SHA-locked state index disagree"
            )

        with np.load(npz_path, allow_pickle=False) as z:
            if set(z.files) != {"a2_masks_packed"}:
                raise RuntimeError(f"{state_id}: unexpected NPZ keys")
            a2 = np.asarray(z["a2_masks_packed"], dtype=np.uint8)
        if a2.shape != (CASES, PACKED_BYTES):
            raise RuntimeError(f"{state_id}: A2 array shape={a2.shape}")

        for r in tqdm(
            g.sort_values("row_index", kind="mergesort").itertuples(index=False),
            total=len(g), desc=f"A2 utility {state_id}", unit="case", dynamic_ncols=True,
        ):
            sid = str(r.sample_id)
            i = int(r.row_index)
            a2_dice = r05d4.binary_dice(unpack_mask(a2[i]), gt_masks[sid])
            rows.append({
                "sample_id":sid,
                "model_family":str(r.model_family),
                "model_state_id":str(r.model_state_id),
                "training_seed":int(r.training_seed),
                "checkpoint_sha256":str(r.checkpoint_sha256),
                "a2_dice":float(a2_dice),
            })

    a2 = pd.DataFrame(rows)
    keep = [
        "sample_id","model_family","model_state_id","training_seed","checkpoint_sha256",
        "source_dice","a1_dice","delta_dice","adaptation_outcome","harm_label","benefit_label",
    ]
    d4 = d4[keep].copy()
    panel = a2.merge(
        d4, on=["sample_id","model_state_id"], how="left",
        validate="one_to_one", suffixes=("_a2meta","_a1meta"),
    )
    if len(panel) != MODEL_CASES:
        raise RuntimeError("A2/R05D4 join !=9000")
    for c in ("model_family","training_seed","checkpoint_sha256"):
        if not (panel[f"{c}_a2meta"].astype(str) == panel[f"{c}_a1meta"].astype(str)).all():
            raise RuntimeError(f"Metadata mismatch: {c}")
        panel[c] = panel[f"{c}_a2meta"]

    panel["a2_delta_dice"] = panel["a2_dice"].to_numpy(float) - panel["source_dice"].to_numpy(float)
    panel["a2_adaptation_outcome"] = [outcome(x) for x in panel["a2_delta_dice"]]
    panel["a2_harm_label"] = panel["a2_adaptation_outcome"].eq("HARM").astype(int)
    panel["a2_benefit_label"] = panel["a2_adaptation_outcome"].eq("BENEFIT").astype(int)
    panel = panel.rename(columns={
        "delta_dice":"a1_delta_dice",
        "adaptation_outcome":"a1_adaptation_outcome",
        "harm_label":"a1_harm_label",
        "benefit_label":"a1_benefit_label",
    })
    drop = [
        f"{c}_{s}"
        for c in ("model_family","training_seed","checkpoint_sha256")
        for s in ("a2meta","a1meta")
    ]
    return panel.drop(columns=drop)


def join_scores(panel):
    pred = pd.read_csv(R10K2B_PRED, low_memory=False)
    req = [
        "target_family","seed","method","sample_id","model_state_id",
        "harm_label","probability","source_only_threshold",
    ]
    miss = [c for c in req if c not in pred.columns]
    if miss:
        raise RuntimeError(f"R10K2B missing: {miss}")
    expected = len(FAMILIES)*len(SEEDS)*len(METHODS)*3000
    if len(pred) != expected:
        raise RuntimeError(f"R10K2B rows={len(pred)} expected={expected}")

    pred["seed"] = pd.to_numeric(pred["seed"]).astype(int)
    pred["harm_label"] = pd.to_numeric(pred["harm_label"]).astype(int)
    pred["probability"] = pd.to_numeric(pred["probability"]).astype(float)
    pred["source_only_threshold"] = pd.to_numeric(pred["source_only_threshold"]).astype(float)

    labels = panel[
        ["sample_id","model_state_id","model_family","a1_harm_label","a2_harm_label"]
    ]
    j = pred.merge(labels, on=["sample_id","model_state_id"], how="left", validate="many_to_one")
    if not np.array_equal(j["harm_label"].to_numpy(int), j["a1_harm_label"].to_numpy(int)):
        raise RuntimeError("R10K2B A1 labels != authoritative R05D4")
    if not (j["target_family"].astype(str) == j["model_family"].astype(str)).all():
        raise RuntimeError("target_family/model_family mismatch")
    return j


def point_metrics(j):
    rows = []
    for fam in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                sub = j[
                    (j["target_family"]==fam)&(j["seed"]==seed)&(j["method"]==method)
                ]
                if len(sub) != 3000:
                    raise RuntimeError(f"{fam}/{seed}/{method}: rows={len(sub)}")
                for action,label in (
                    ("A1_TENT_1STEP","a1_harm_label"),
                    ("A2_TENT_2STEP","a2_harm_label"),
                ):
                    m = metric(sub[label], sub["probability"], sub["source_only_threshold"])
                    m["OracleR90FPR"] = oracle_fpr90(sub[label], sub["probability"])
                    rows.append({
                        "target_family":fam,"seed":seed,"method":method,"outcome_action":action,**m
                    })
    return pd.DataFrame(rows)


def summarize(points):
    fam = (
        points.groupby(["target_family","method","outcome_action"], sort=False)
        .agg(
            AUROC_mean=("AUROC","mean"),AUPRC_mean=("AUPRC","mean"),
            Recall_mean=("Recall","mean"),FPR_mean=("FPR","mean"),
            PPV1pct_mean=("PPV1pct","mean"),OracleR90FPR_mean=("OracleR90FPR","mean"),
        ).reset_index()
    )
    seedmacro = (
        points.groupby(["method","outcome_action","seed"], sort=False)
        .agg(
            AUROC=("AUROC","mean"),AUPRC=("AUPRC","mean"),Recall=("Recall","mean"),
            FPR=("FPR","mean"),PPV1pct=("PPV1pct","mean"),OracleR90FPR=("OracleR90FPR","mean"),
        ).reset_index()
    )
    macro = (
        seedmacro.groupby(["method","outcome_action"], sort=False)
        .agg(
            AUROC_mean=("AUROC","mean"),AUROC_std=("AUROC","std"),
            AUPRC_mean=("AUPRC","mean"),AUPRC_std=("AUPRC","std"),
            Recall_mean=("Recall","mean"),FPR_mean=("FPR","mean"),
            PPV1pct_mean=("PPV1pct","mean"),OracleR90FPR_mean=("OracleR90FPR","mean"),
        ).reset_index()
    )
    return fam, macro


def bootstrap(j):
    sids = sorted(j["sample_id"].astype(str).unique())
    sid_to_i = {sid:i for i,sid in enumerate(sids)}
    scenarios = {
        "PRIMARY_A1":(PRIMARY,"a1_harm_label"),
        "PRIMARY_A2":(PRIMARY,"a2_harm_label"),
        "M2_A2":(BASELINE,"a2_harm_label"),
    }
    payload = {}
    for name,(method,label) in scenarios.items():
        for fam in FAMILIES:
            for seed in SEEDS:
                sub = j[
                    (j["method"]==method)&(j["target_family"]==fam)&(j["seed"]==seed)
                ].copy()
                sub["_ci"] = sub["sample_id"].astype(str).map(sid_to_i)
                payload[(name,fam,seed)] = (
                    sub[label].to_numpy(int),
                    sub["probability"].to_numpy(float),
                    sub["source_only_threshold"].to_numpy(float),
                    sub["_ci"].to_numpy(int),
                )

    rng = np.random.default_rng(BOOT_SEED)
    names = ("AUROC","AUPRC","Recall","FPR","PPV1pct")
    store = {s:{m:[] for m in names} for s in scenarios}

    for _ in tqdm(range(BOOTSTRAPS), desc="R11D clustered bootstrap", unit="rep", dynamic_ncols=True):
        counts = np.bincount(
            rng.integers(0, CASES, size=CASES), minlength=CASES
        ).astype(float)

        for scenario in scenarios:
            fam_vals = {m:[] for m in names}
            for fam in FAMILIES:
                seed_vals = {m:[] for m in names}
                for seed in SEEDS:
                    y,p,t,ci = payload[(scenario,fam,seed)]
                    mm = metric(y,p,t,sample_weight=counts[ci])
                    if mm is None:
                        raise RuntimeError("Bootstrap class degeneration")
                    for m in names:
                        seed_vals[m].append(mm[m])
                for m in names:
                    fam_vals[m].append(float(np.mean(seed_vals[m])))
            for m in names:
                store[scenario][m].append(float(np.mean(fam_vals[m])))

    rows = []
    def add(level,group,m,x):
        x = np.asarray(x,dtype=float)
        rows.append({
            "level":level,"group":group,"metric":m,
            "mean":float(x.mean()),
            "ci95_low":float(np.quantile(x,0.025)),
            "ci95_high":float(np.quantile(x,0.975)),
        })

    for s in scenarios:
        for m in names:
            add("SCENARIO",s,m,store[s][m])

    for group,a,b in (
        ("PRIMARY_A2_MINUS_PRIMARY_A1","PRIMARY_A2","PRIMARY_A1"),
        ("PRIMARY_A2_MINUS_M2_A2","PRIMARY_A2","M2_A2"),
    ):
        for m in names:
            add("PAIRED_DELTA",group,m,np.asarray(store[a][m])-np.asarray(store[b][m]))

    return pd.DataFrame(rows)


def self_test():
    m = metric([1,1,0,0],[.9,.8,.2,.1],[.5,.5,.5,.5])
    assert abs(m["AUROC"]-1.0) < 1e-12
    assert outcome(-0.02)=="HARM"
    assert outcome(0.02)=="BENEFIT"
    assert outcome(0.0)=="NEUTRAL"
    print("SELF_TEST_PASS")


def run(args):
    print("===== R11D TENT2 FROZEN SAFETY TRANSFER EVALUATION FIX4 =====")
    print("adaptation-intensity transfer=YES")
    print("cross-TTA-algorithm transfer=NO")
    print("independent external validation=NO")
    print("post-PolypGen exploratory robustness extension=YES")
    print("training/refit=NO")
    print("threshold reselection=NO")
    print("score reversal=NO")
    print("PolypGen access=NO")

    r11c_sha = verify_r11c()

    relocation_audit = audit_r11c_manifest_relocation()
    print("\n===== R11C STATE-ASSET PATH RELOCATION AUDIT =====")
    print(relocation_audit.to_string(index=False))
    print("filesystem search/glob selection=NO")
    print("relocation rule=known __building parent -> final R11C state_predictions basename")

    r05d4,d4,gt = load_gt_and_r05d4()
    panel = build_a2_panel(r05d4,d4,gt)

    print("\n===== OUTCOME SHIFT =====")
    print("A1 HARM:", int(panel["a1_harm_label"].sum()))
    print("A2 HARM:", int(panel["a2_harm_label"].sum()))
    print("A1 BENEFIT:", int(panel["a1_benefit_label"].sum()))
    print("A2 BENEFIT:", int(panel["a2_benefit_label"].sum()))
    print("HARM agreement:", float(np.mean(panel["a1_harm_label"]==panel["a2_harm_label"])))
    print("DeltaDice Spearman:", float(panel["a1_delta_dice"].corr(panel["a2_delta_dice"],method="spearman")))

    j = join_scores(panel)
    print("R10K2B A1 label reproduction=100% PASS")
    print("frozen probability/threshold reuse=YES")

    points = point_metrics(j)
    fam,macro = summarize(points)

    print("\n===== PRIMARY / M2 MACRO =====")
    print(macro[macro["method"].isin([PRIMARY,BASELINE])].to_string(index=False))
    print("\n===== PRIMARY BY FAMILY =====")
    print(fam[fam["method"]==PRIMARY].to_string(index=False))

    boot = bootstrap(j)
    key = boot[
        ((boot["level"]=="SCENARIO")&(boot["group"]=="PRIMARY_A2"))
        | (boot["level"]=="PAIRED_DELTA")
    ]
    print("\n===== CLUSTERED BOOTSTRAP =====")
    print(key.to_string(index=False))

    auc_ci = boot[
        (boot["level"]=="SCENARIO")&(boot["group"]=="PRIMARY_A2")&(boot["metric"]=="AUROC")
    ].iloc[0]
    decision = DECISION_YES if float(auc_ci["ci95_low"]) > 0.5 else DECISION_NO

    pa2 = macro[
        (macro["method"]==PRIMARY)&(macro["outcome_action"]=="A2_TENT_2STEP")
    ].iloc[0]

    print("\n===== FINAL DECISION =====")
    print("Decision:", decision)
    print("Primary A2 AUROC:", float(pa2["AUROC_mean"]))
    print("Primary A2 AUROC CI95:", [float(auc_ci["ci95_low"]),float(auc_ci["ci95_high"])])
    print("Primary A2 AUPRC:", float(pa2["AUPRC_mean"]))
    print("Primary A2 Recall:", float(pa2["Recall_mean"]))
    print("Primary A2 FPR:", float(pa2["FPR_mean"]))
    print("Primary A2 Oracle FPR@R90:", float(pa2["OracleR90FPR_mean"]))

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=False)

    relocation_audit.to_csv(
        args.output_dir/"R11D_R11C_state_asset_relocation_audit.csv",
        index=False,
    )
    panel.to_csv(args.output_dir/"R11D_A1_A2_outcome_panel.csv",index=False)
    points.to_csv(args.output_dir/"R11D_per_seed_family_method_metrics.csv",index=False)
    fam.to_csv(args.output_dir/"R11D_by_family_summary.csv",index=False)
    macro.to_csv(args.output_dir/"R11D_macro_summary.csv",index=False)
    boot.to_csv(args.output_dir/"R11D_clustered_bootstrap_ci.csv",index=False)

    pd.crosstab(
        panel["a1_harm_label"],panel["a2_harm_label"],
        rownames=["A1_harm"],colnames=["A2_harm"],
    ).to_csv(args.output_dir/"R11D_A1_to_A2_harm_transition.csv")

    lock = {
        "decision":decision,
        "r11c_lock_sha256":r11c_sha,
        "r10k2b_target_predictions_sha256":sha256_file(R10K2B_PRED),
        "r05d4_lock_sha256":sha256_file(R05D4_LOCK),
        "cases":CASES,"states":STATES,"model_case_rows":MODEL_CASES,
        "a1_harm_rows":int(panel["a1_harm_label"].sum()),
        "a2_harm_rows":int(panel["a2_harm_label"].sum()),
        "primary_a2_macro":{
            "auroc":float(pa2["AUROC_mean"]),
            "auprc":float(pa2["AUPRC_mean"]),
            "recall":float(pa2["Recall_mean"]),
            "fpr":float(pa2["FPR_mean"]),
            "ppv1pct":float(pa2["PPV1pct_mean"]),
            "oracle_r90_fpr":float(pa2["OracleR90FPR_mean"]),
            "auroc_ci95":[float(auc_ci["ci95_low"]),float(auc_ci["ci95_high"])],
        },
        "information_boundary":{
            "training_or_refit":False,
            "threshold_reselection":False,
            "score_reversal":False,
            "polypgen_access":False,
            "cross_tta_algorithm_claim":False,
            "adaptation_intensity_transfer_claim":True,
            "independent_external_validation_claim":False,
            "post_polypgen_exploratory_extension":True,
            "r11c_state_asset_path_relocation":
                "deterministic_known_building_parent_to_final_dir_basename",
            "filesystem_search_or_glob_for_relocation":False,
        },
    }
    lock_path = args.output_dir/"R11D_TENT2_SAFETY_TRANSFER_EVALUATION_LOCK.json"
    write_json(lock_path,lock)
    print("R11D LOCK:",lock_path)
    print("R11D LOCK SHA256:",sha256_file(lock_path))
    print("PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    p.add_argument("--self-test",action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__=="__main__":
    main()
