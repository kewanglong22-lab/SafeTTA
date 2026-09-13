#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTTA R31B3
FIRST 800-case GT Reveal + Three-Action Leave-One-Action-Out HARM Evaluation

Primary LOAO:
  Train TENT1 + PL-CONF90   -> Test unseen MEMO-SEG4-1STEP
  Train TENT1 + MEMO        -> Test unseen PL-CONF90
  Train PL-CONF90 + MEMO    -> Test unseen TENT1

Physical sample_id is split BEFORE action selection, so the held-out action
and held-out physical images are absent from predictor fitting.

Frozen classifier:
  StandardScaler(train only) + balanced LogisticRegression(C=1)
  no hyperparameter search

Primary feature contract:
  q_source66 + delta_semantic64 + ActionID + FamilyActionID

Decision locked before first 800-case GT reveal:
  GO:
    LOAO macro AUROC >= 0.65
    every held-out-action AUROC >= 0.60
    pooled shared-action macro AUROC - separate-action macro AUROC >= -0.03
  STRONG GO:
    LOAO macro AUROC >= 0.70
    every held-out-action AUROC >= 0.65
    shared-minus-separate >= -0.02

No post-GT MEMO tuning, action redesign, representation redesign, classifier
hyperparameter search, or external-cohort access is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "2026-09-12-R31B3-v1"
ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R31A_SCRIPT = CODE / "Q1_R31A_action_conditional_harm_predictor_v1.py"
R31A_SCRIPT_SHA = "28159535a2994eadadc091ed3fae85f128cec401c8e38a9e12149d70dbf0227e"
R31A_DIR = ROOT / "R31A_action_conditional_harm_predictor_v1"
R31A_FINAL = R31A_DIR / "R31A_FINAL_LOCK.json"
R31A_FINAL_SHA = "a03d707e8f3c4054feba867a252b270f58a2838b141ce11f8cd70774c9cf3b62"
R31A_LINEAGE = R31A_DIR / "R31A_SOURCE_TABLE_LINEAGE.json"

R31B0_DIR = ROOT / "R31B0_memo_seg4_third_action_protocol_lock_v1"
R31B0_FINAL = R31B0_DIR / "R31B0_FINAL_LOCK.json"
R31B0_FINAL_SHA = "03e16b4064611a927946b577d8a0740353e9037bd4eca72426d441c709010951"
R31B0_PARTITION = R31B0_DIR / "R31B0_SOURCE_CASE_PARTITION.csv"

R31B1_SCRIPT = CODE / "Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py"
R31B1_SCRIPT_SHA = "3391fe4d7161df488f372379a65d72c0dc0cc4e1e774e02e90d68c9ad17f911b"
R31B1_DIR = ROOT / "R31B1_memo_seg4_action_design_and_lr_freeze_v1"
R31B1_SELECTED = R31B1_DIR / "R31B1_SELECTED_MEMO_LR_LOCK.json"
R31B1_SELECTED_SHA = "6915ead8a8ccffc3c1b00cc5003a118c20b390b4303345943609fa94741b4551"
R31B1_FINAL = R31B1_DIR / "R31B1_FINAL_LOCK.json"
R31B1_FINAL_SHA = "09668378b0083876938bf67e0ebbbee17ba774760147ccce9dba5d1683869044"

R31B2_SCRIPT = CODE / "Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py"
R31B2_SCRIPT_SHA = "fa4802f62bb510b953b216f2a3211df8c556dee943f612493ae75463870bbf5b"
R31B2_DIR = ROOT / "R31B2_800case_memo_prediction_and_semantic_lock_v1"
R31B2_FINAL = R31B2_DIR / "R31B2_FINAL_PRE_GT_LOCK.json"
R31B2_FINAL_SHA = "f2d5c7b6edbffe596d40cc839e9ec437349f8e1bc127a3c7ab5df0c345a120e8"

R30A0_SCRIPT = CODE / "Q1_R30A0_paired_action_crc_protocol_and_asset_audit_v1_fix1.py"
R30A0_SHA = "9dc5d3a13f3d64608c4c599f811f929f731d7fac05e2aaa66eb344e24ace2298"

DEFAULT_OUT = ROOT / "R31B3_first_800case_gt_reveal_and_three_action_loao_v1"

ACTIONS = ("TENT1", "PL-CONF90", "MEMO-SEG4-1STEP")
MEMO = "MEMO-SEG4-1STEP"
MEMO_LR = 1e-5
N_CASES = 800
N_STATES = 9
ROWS_PER_ACTION = N_CASES * N_STATES
TOTAL_ROWS = ROWS_PER_ACTION * 3
IMAGE_SIZE = 352
PACKED_BYTES = (IMAGE_SIZE * IMAGE_SIZE + 7) // 8

HARM_THR = -0.02
BENEFIT_THR = 0.02
CV_SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 5000

GO_MACRO = 0.65
GO_MIN = 0.60
GO_GAP = -0.03
STRONG_MACRO = 0.70
STRONG_MIN = 0.65
STRONG_GAP = -0.02
Q_ALIGN_TOL = 5e-3

PRIMARY = (
    (("TENT1", "PL-CONF90"), MEMO),
    (("TENT1", MEMO), "PL-CONF90"),
    (("PL-CONF90", MEMO), "TENT1"),
)

FORBIDDEN = ("endotect", "polypgen", "sun-seg", "sunseg", "promise12")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def req(p: Path, expected: str, label: str):
    if not p.is_file():
        raise FileNotFoundError(f"{label}: {p}")
    got = sha256_file(p)
    if got != expected:
        raise RuntimeError(f"{label} SHA mismatch expected={expected} observed={got}")
    return got

def atomic_json(p: Path, obj: Any):
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(t, p)

def atomic_csv(d: pd.DataFrame, p: Path):
    t = p.with_name(p.name + ".tmp")
    d.to_csv(t, index=False)
    os.replace(t, p)

def imp(p: Path, name: str):
    s = importlib.util.spec_from_file_location(name, str(p))
    if s is None or s.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m

def reject_external(p: Path):
    low = str(p).lower()
    if any(x in low for x in FORBIDDEN):
        raise RuntimeError(f"External path forbidden: {p}")

def norm_action(v: Any) -> str:
    s = str(v).strip().upper().replace("_", "-")
    aliases = {
        "TENT1": "TENT1", "TENT-1": "TENT1", "A1": "TENT1",
        "A1-TENT-1STEP": "TENT1", "A1-TENT1": "TENT1",
        "PL-CONF90": "PL-CONF90", "PLCONF90": "PL-CONF90",
        "PL-CONF-90": "PL-CONF90", "PL90": "PL-CONF90",
        "MEMO-SEG4-1STEP": MEMO,
    }
    return aliases.get(s, str(v).strip())

def resolve_harm(d: pd.DataFrame) -> Tuple[np.ndarray, str]:
    for c in ["harm_label", "is_harm", "harm"]:
        if c in d.columns:
            y = pd.to_numeric(d[c], errors="raise").to_numpy(int)
            if set(np.unique(y)).issubset({0, 1}):
                return y, c
    for c in ["delta_dice", "candidate_delta_dice", "true_delta_dice", "action_delta_dice", "utility_target"]:
        if c in d.columns:
            x = pd.to_numeric(d[c], errors="raise").to_numpy(float)
            return (x <= HARM_THR).astype(int), f"{c}<=-0.02"
    raise RuntimeError("Cannot resolve TENT/PL HARM label")

def verify_chain() -> Dict[str, Any]:
    req(R31A_SCRIPT, R31A_SCRIPT_SHA, "R31A script")
    req(R31A_FINAL, R31A_FINAL_SHA, "R31A final")
    req(R31B0_FINAL, R31B0_FINAL_SHA, "R31B0 final")
    req(R31B1_SCRIPT, R31B1_SCRIPT_SHA, "R31B1 script")
    req(R31B1_SELECTED, R31B1_SELECTED_SHA, "R31B1 selected")
    req(R31B1_FINAL, R31B1_FINAL_SHA, "R31B1 final")
    req(R31B2_SCRIPT, R31B2_SCRIPT_SHA, "R31B2 script")
    req(R31B2_FINAL, R31B2_FINAL_SHA, "R31B2 final PRE-GT")
    req(R30A0_SCRIPT, R30A0_SHA, "R30 schema")

    b2 = json.loads(R31B2_FINAL.read_text(encoding="utf-8"))
    if b2.get("status") != "PASS_R31B2_800CASE_PRE_GT_LOCK_COMPLETE":
        raise RuntimeError("R31B2 not PASS")
    if list(b2.get("evaluation_gt_read_hash_decode", [1,1,1])) != [False, False, False]:
        raise RuntimeError("R31B2 indicates GT access")
    if float(b2.get("selected_learning_rate")) != MEMO_LR:
        raise RuntimeError("MEMO LR drift")
    if int(b2.get("physical_cases")) != N_CASES or int(b2.get("model_states")) != N_STATES:
        raise RuntimeError("R31B2 cohort/state drift")

    for k, sk in [
        ("rgb_manifest", "rgb_manifest_sha256"),
        ("prediction_inventory", "prediction_inventory_sha256"),
        ("semantic_lock", "semantic_lock_sha256"),
        ("feature_table", "feature_table_sha256"),
    ]:
        p = Path(str(b2[k]))
        if not p.is_file() or sha256_file(p) != str(b2[sk]):
            raise RuntimeError(f"R31B2 artifact drift: {k}")
    return b2

def write_protocol(out: Path, b2: Dict[str, Any]) -> Tuple[Path, str]:
    p = out / "R31B3_PROTOCOL_LOCK_PRE_GT.json"
    obj = {
        "status": "LOCKED_BEFORE_FIRST_800CASE_GT_REVEAL",
        "version": VERSION,
        "upstream": {
            "R31A_final_sha256": R31A_FINAL_SHA,
            "R31B0_final_sha256": R31B0_FINAL_SHA,
            "R31B1_selected_sha256": R31B1_SELECTED_SHA,
            "R31B1_final_sha256": R31B1_FINAL_SHA,
            "R31B2_final_pre_gt_sha256": R31B2_FINAL_SHA,
            "R31B2_semantic_lock_sha256": b2["semantic_lock_sha256"],
        },
        "information_boundary": {
            "evaluation_gt_read": False,
            "evaluation_gt_hash": False,
            "evaluation_gt_decode": False,
            "external_data_access": False,
        },
        "actions": list(ACTIONS),
        "feature_contract": "q_source66 + delta_semantic64 + ActionID + FamilyActionID",
        "classifier": {
            "model": "balanced LogisticRegression", "C": LR_C,
            "max_iter": LR_MAX_ITER, "hyperparameter_search": False,
        },
        "cv": {"folds": N_FOLDS, "seeds": CV_SEEDS, "group": "sample_id"},
        "primary_loao": [
            {"train_actions": list(a), "test_action": t}
            for a, t in PRIMARY
        ],
        "harm": "delta_dice <= -0.02",
        "decision": {
            "GO": {"macro_auroc": GO_MACRO, "min_action_auroc": GO_MIN, "shared_minus_separate": GO_GAP},
            "STRONG_GO": {"macro_auroc": STRONG_MACRO, "min_action_auroc": STRONG_MIN, "shared_minus_separate": STRONG_GAP},
        },
        "post_gt_prohibitions": [
            "MEMO LR change", "action redesign", "feature redesign",
            "classifier hyperparameter search", "external cohort access",
        ],
    }
    atomic_json(p, obj)
    return p, sha256_file(p)

def load_tent_pl(qnames, dnames, eval_ids):
    lineage = json.loads(R31A_LINEAGE.read_text(encoding="utf-8"))
    info = lineage["source_table"]
    p = Path(str(info["path"]))
    reject_external(p)
    if sha256_file(p) != str(info["sha256"]):
        raise RuntimeError("R31A source table SHA drift")
    d = pd.read_csv(p, low_memory=False)
    d["sample_id"] = d["sample_id"].astype(str)
    d["action"] = d["action"].map(norm_action)
    d = d[d["sample_id"].isin(eval_ids) & d["action"].isin(["TENT1", "PL-CONF90"])].copy()
    if len(d) != ROWS_PER_ACTION * 2:
        raise RuntimeError(f"TENT/PL rows={len(d)}")
    y, ysrc = resolve_harm(d)
    d["r31b3_harm_label"] = y
    if "delta_dice" not in d.columns:
        for c in ["candidate_delta_dice", "true_delta_dice", "action_delta_dice", "utility_target"]:
            if c in d.columns:
                d["delta_dice"] = pd.to_numeric(d[c], errors="raise")
                break
    for c in list(qnames) + list(dnames):
        if c not in d.columns:
            raise RuntimeError(f"Missing frozen feature {c}")
    return d, {"path": str(p), "sha256": sha256_file(p), "harm_label_source": ysrc}

def unpack_mask(x: np.ndarray, bitorder: str) -> np.ndarray:
    b = np.unpackbits(np.asarray(x, dtype=np.uint8), count=IMAGE_SIZE*IMAGE_SIZE, bitorder=bitorder)
    return b.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)

def reveal_memo(out: Path, b2: Dict[str, Any], qnames, dnames):
    r31b1 = imp(R31B1_SCRIPT, "r31b3_r31b1")
    part = pd.read_csv(R31B0_PARTITION, low_memory=False)
    part["sample_id"] = part["sample_id"].astype(str)
    ev = part[part["r31b_partition"] == "R31B_LOAO_EVALUATION"].copy()
    ev = ev.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(ev) != N_CASES:
        raise RuntimeError("Evaluation partition drift")

    feat = pd.read_csv(Path(str(b2["feature_table"])), low_memory=False)
    feat["sample_id"] = feat["sample_id"].astype(str)
    feat["action"] = feat["action"].map(norm_action)
    if len(feat) != ROWS_PER_ACTION:
        raise RuntimeError("MEMO feature rows drift")

    inv = pd.read_csv(Path(str(b2["prediction_inventory"])), low_memory=False)
    if len(inv) != N_STATES:
        raise RuntimeError("Prediction inventory drift")

    print("\n" + "!"*124)
    print("FIRST R31B 800-CASE EVALUATION GT REVEAL STARTS NOW")
    print("MEMO predictions, LR and semantic transitions are already frozen.")
    print("From this point onward: EVALUATION ONLY; NO tuning.")
    print("!"*124)

    gt_cache = {}
    gtr = []
    for _, r in tqdm(ev.iterrows(), total=N_CASES, desc="FIRST R31B 800-case GT reveal", unit="mask", dynamic_ncols=True):
        sid = str(r["sample_id"])
        gp = Path(str(r["gt_path"]))
        reject_external(gp)
        if not gp.is_file():
            raise FileNotFoundError(gp)
        gt = np.asarray(r31b1.load_gt352(gp), dtype=np.uint8)
        gt_cache[sid] = gt
        gtr.append({
            "sample_id": sid, "gt_path": str(gp), "gt_bytes": int(gp.stat().st_size),
            "gt_sha256": sha256_file(gp), "gt_foreground_pixels": int(gt.sum()),
        })
    gt_manifest = pd.DataFrame(gtr)
    atomic_csv(gt_manifest, out / "R31B3_800CASE_GT_REVEAL_MANIFEST.csv")

    outs = []
    for _, st in inv.sort_values("state_index", kind="mergesort").iterrows():
        npz = Path(str(st["prediction_npz"]))
        lockp = Path(str(st["state_lock"]))
        if sha256_file(npz) != str(st["prediction_npz_sha256"]):
            raise RuntimeError("Prediction NPZ drift")
        lock = json.loads(lockp.read_text(encoding="utf-8"))
        bitorder = str(lock["bitorder"])
        with np.load(npz, allow_pickle=False) as z:
            src = np.asarray(z["source_masks_packed"], dtype=np.uint8)
            mem = np.asarray(z["memo_masks_packed"], dtype=np.uint8)
            sids = np.asarray(z["sample_id"]).astype(str)
        for i, sid in enumerate(sids):
            gt = gt_cache[str(sid)]
            sm = unpack_mask(src[i], bitorder)
            mm = unpack_mask(mem[i], bitorder)
            sd = float(r31b1.dice_binary(sm, gt))
            md = float(r31b1.dice_binary(mm, gt))
            dd = md - sd
            outs.append({
                "sample_id": str(sid),
                "model_state_id": str(st["model_state_id"]),
                "model_family": str(st["model_family"]),
                "training_seed": int(st["training_seed"]),
                "action": MEMO, "source_dice": sd, "memo_dice": md,
                "delta_dice": dd, "r31b3_harm_label": int(dd <= HARM_THR),
                "benefit": int(dd >= BENEFIT_THR),
                "neutral": int(HARM_THR < dd < BENEFIT_THR),
            })
    outcome = pd.DataFrame(outs)
    keys = ["sample_id","model_state_id","model_family","training_seed","action"]
    memo = feat.merge(outcome, on=keys, how="inner", validate="one_to_one")
    if len(memo) != ROWS_PER_ACTION:
        raise RuntimeError("MEMO feature/outcome merge drift")
    atomic_csv(memo, out / "R31B3_MEMO_800CASE_MODELCASE_OUTCOMES.csv")
    return memo, gt_manifest

def check_q(panel: pd.DataFrame, qnames):
    key = ["sample_id","model_state_id"]
    ref = panel[panel["action"]=="TENT1"].sort_values(key, kind="mergesort").reset_index(drop=True)
    maxd = 0.0
    for action in ["PL-CONF90", MEMO]:
        g = panel[panel["action"]==action].sort_values(key, kind="mergesort").reset_index(drop=True)
        if not ref[key].equals(g[key]):
            raise RuntimeError(f"Key misalignment TENT1 vs {action}")
        d = float(np.max(np.abs(ref[list(qnames)].to_numpy(float) - g[list(qnames)].to_numpy(float))))
        maxd = max(maxd, d)
        if d > Q_ALIGN_TOL:
            raise RuntimeError(f"q_source mismatch vs {action}: {d} > {Q_ALIGN_TOL}")
    return {"status":"PASS","max_abs_q_source_difference":maxd,"tolerance":Q_ALIGN_TOL}

def categories(d: pd.DataFrame):
    acts = list(ACTIONS)
    fams = sorted(d["model_family"].astype(str).unique().tolist())
    fas = [f"{f}::{a}" for f in fams for a in acts]
    return acts, fas

def xy(d, qnames, dnames, mode, acts, fas):
    q, ds = list(qnames), list(dnames)
    if mode == "q66":
        X = d[q].to_numpy(float); names=q
    elif mode == "transition":
        names=q+ds; X=d[names].to_numpy(float)
    elif mode == "action_conditional":
        names=q+ds
        blocks=[d[names].to_numpy(float)]
        av=d["action"].astype(str).to_numpy()
        for a in acts:
            blocks.append((av==a).astype(float)[:,None]); names.append("ActionID::"+a)
        fav=(d["model_family"].astype(str)+"::"+d["action"].astype(str)).to_numpy()
        for fa in fas:
            blocks.append((fav==fa).astype(float)[:,None]); names.append("FamilyActionID::"+fa)
        X=np.concatenate(blocks,axis=1)
    else:
        raise ValueError(mode)
    y=d["r31b3_harm_label"].to_numpy(int)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite features")
    return X,y,names

def fit_lr(X,y):
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training fold one-class")
    m=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=LR_C,class_weight="balanced",max_iter=LR_MAX_ITER,solver="lbfgs",random_state=0)),
    ])
    m.fit(X,y); return m

def metrics(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); prev=float(np.mean(y))
    if len(np.unique(y))!=2:
        return {"auroc":math.nan,"auprc":math.nan,"brier":math.nan,"prevalence":prev,"auprc_lift":math.nan,"n":len(y)}
    ap=float(average_precision_score(y,s))
    return {"auroc":float(roc_auc_score(y,s)),"auprc":ap,"brier":float(brier_score_loss(y,s)),
            "prevalence":prev,"auprc_lift":float(ap/prev) if prev>0 else math.nan,"n":len(y)}

def folds(d, seed):
    g=(d.groupby("sample_id",as_index=False).agg(harm_rate=("r31b3_harm_label","mean"))
       .sort_values("sample_id",kind="mergesort").reset_index(drop=True))
    ranks=g["harm_rate"].rank(method="first",pct=True).to_numpy(float)
    strata=np.minimum((ranks*10).astype(int),9)
    if int(pd.Series(strata).value_counts().min())<N_FOLDS:
        strata=np.minimum((ranks*5).astype(int),4)
    cv=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
    out={}
    for f,(_,te) in enumerate(cv.split(np.zeros(len(g)),strata)):
        for i in te: out[str(g.iloc[int(i)]["sample_id"])]=f
    if len(out)!=N_CASES: raise RuntimeError("Fold map incomplete")
    return out

def addpred(rows,test,y,s,seed,fold,model,direction):
    for r,yy,ss in zip(test.itertuples(index=False),y,s):
        rows.append({"split_seed":seed,"fold":fold,"model":model,"direction":direction,
                     "sample_id":str(r.sample_id),"model_state_id":str(r.model_state_id),
                     "model_family":str(r.model_family),"action":str(r.action),
                     "y_harm":int(yy),"score":float(ss)})

def run_seed(d,qnames,dnames,seed):
    fm=folds(d,seed); w=d.copy()
    w["fold"]=w["sample_id"].astype(str).map(fm).astype(int)
    acts,fas=categories(w); rows=[]
    for fold in range(N_FOLDS):
        tr=w[w["fold"]!=fold].copy(); te=w[w["fold"]==fold].copy()
        if set(tr["sample_id"]) & set(te["sample_id"]): raise RuntimeError("Case leakage")
        for name,mode in [("SOURCE_ONLY_Q66","q66"),("SHARED_TRANSITION","transition"),
                          ("SHARED_ACTION_CONDITIONAL","action_conditional")]:
            X,y,_=xy(tr,qnames,dnames,mode,acts,fas); Xt,yt,_=xy(te,qnames,dnames,mode,acts,fas)
            m=fit_lr(X,y); addpred(rows,te,yt,m.predict_proba(Xt)[:,1],seed,fold,name,"POOLED")
        for a in ACTIONS:
            a_tr=tr[tr["action"]==a]; a_te=te[te["action"]==a]
            X,y,_=xy(a_tr,qnames,dnames,"transition",acts,fas); Xt,yt,_=xy(a_te,qnames,dnames,"transition",acts,fas)
            m=fit_lr(X,y); addpred(rows,a_te,yt,m.predict_proba(Xt)[:,1],seed,fold,"SEPARATE_ACTION",a)
        for train_actions,test_action in PRIMARY:
            a_tr=tr[tr["action"].isin(train_actions)]; a_te=te[te["action"]==test_action]
            X,y,_=xy(a_tr,qnames,dnames,"action_conditional",acts,fas); Xt,yt,_=xy(a_te,qnames,dnames,"action_conditional",acts,fas)
            m=fit_lr(X,y); addpred(rows,a_te,yt,m.predict_proba(Xt)[:,1],seed,fold,
                                   "PRIMARY_LOAO_ACTION_CONDITIONAL",f"{'+'.join(train_actions)}->{test_action}")
            X,y,_=xy(a_tr,qnames,dnames,"transition",acts,fas); Xt,yt,_=xy(a_te,qnames,dnames,"transition",acts,fas)
            m=fit_lr(X,y); addpred(rows,a_te,yt,m.predict_proba(Xt)[:,1],seed,fold,
                                   "LOAO_TRANSITION_ONLY",f"{'+'.join(train_actions)}->{test_action}")
    pred=pd.DataFrame(rows)
    if len(pred)!=TOTAL_ROWS*6: raise RuntimeError(f"Prediction rows={len(pred)}")
    mr=[]
    for name,g in pred.groupby("model",sort=True):
        mr.append({"split_seed":seed,"model":name,"scope":"POOLED","action":"ALL",**metrics(g.y_harm,g.score)})
        for a,ga in g.groupby("action",sort=True):
            mr.append({"split_seed":seed,"model":name,"scope":"ACTION","action":str(a),**metrics(ga.y_harm,ga.score)})
    return pred,pd.DataFrame(mr)

def summarize(d):
    rows=[]
    for (m,s,a),g in d.groupby(["model","scope","action"],sort=True):
        r={"model":m,"scope":s,"action":a,"seeds":int(g.split_seed.nunique())}
        for k in ["auroc","auprc","brier","prevalence","auprc_lift","n"]:
            v=pd.to_numeric(g[k],errors="coerce").to_numpy(float)
            r[k+"_mean"]=float(np.nanmean(v)); r[k+"_std"]=float(np.nanstd(v,ddof=1)) if len(v)>1 else 0.0
            r[k+"_min"]=float(np.nanmin(v)); r[k+"_max"]=float(np.nanmax(v))
        rows.append(r)
    return pd.DataFrame(rows)

def macro(summary,model):
    g=summary[(summary.model==model)&(summary.scope=="ACTION")&summary.action.isin(ACTIONS)]
    if len(g)!=3: raise RuntimeError(f"{model} action metrics={len(g)}")
    return float(g.auroc_mean.mean())

def decide(summary):
    g=summary[(summary.model=="PRIMARY_LOAO_ACTION_CONDITIONAL")&(summary.scope=="ACTION")&summary.action.isin(ACTIONS)]
    if len(g)!=3: raise RuntimeError("Primary LOAO metrics incomplete")
    held={str(r.action):float(r.auroc_mean) for _,r in g.iterrows()}
    vals=list(held.values()); lm=float(np.mean(vals)); lmin=float(np.min(vals))
    sh=macro(summary,"SHARED_ACTION_CONDITIONAL"); sep=macro(summary,"SEPARATE_ACTION"); gap=sh-sep
    strong=lm>=STRONG_MACRO and lmin>=STRONG_MIN and gap>=STRONG_GAP
    go=lm>=GO_MACRO and lmin>=GO_MIN and gap>=GO_GAP
    if strong: dec="GO_STRONG_ACTION_TRANSFERABLE_SAFETY"; nxt="R31C_MULTI_ACTION_RISK_CONTROLLED_CONTROLLER"
    elif go: dec="GO_ACTION_TRANSFERABLE_SAFETY"; nxt="R31C_MULTI_ACTION_RISK_CONTROLLED_CONTROLLER"
    else: dec="NO_GO_THREE_ACTION_TRANSFER"; nxt="STOP_ACTION_TRANSFER_CLAIM_AND_INTERPRET"
    return {"decision":dec,"metrics":{"primary_loao_macro_auroc":lm,"primary_loao_min_action_auroc":lmin,
            "heldout_action_aurocs":held,"shared_action_conditional_macro_action_auroc":sh,
            "separate_action_macro_action_auroc":sep,"shared_minus_separate_macro_action_auroc":gap},
            "next":nxt}

def inventory(out):
    return pd.DataFrame([{"relative_path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha256_file(p)}
                         for p in sorted(out.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); args=ap.parse_args()
    print("="*124)
    print("SafeTTA R31B3 FIRST 800-case GT Reveal + Three-Action LOAO")
    print("Version:",VERSION)
    print("MEMO LR                       :",MEMO_LR,"(frozen)")
    print("Grouped CV                    : 5 folds x 5 seeds by physical sample_id")
    print("Hyperparameter search         : NO")
    print("External cohort access        : NO")
    print("GT access before verification : NO")
    print("="*124)

    b2=verify_chain()
    a0=imp(R30A0_SCRIPT,"r31b3_a0"); qnames=list(a0.QSOURCE); dnames=list(a0.DSEM)
    if len(qnames)!=66 or len(dnames)!=64: raise RuntimeError("Frozen schema drift")

    part=pd.read_csv(R31B0_PARTITION,low_memory=False); part["sample_id"]=part["sample_id"].astype(str)
    eval_ids=set(part.loc[part.r31b_partition=="R31B_LOAO_EVALUATION","sample_id"])
    if len(eval_ids)!=N_CASES: raise RuntimeError("Evaluation ID count drift")
    tentpl,source_info=load_tent_pl(qnames,dnames,eval_ids)

    out=args.output_dir.resolve()
    if out.exists(): raise RuntimeError(f"Output exists; first-reveal stage cannot overwrite: {out}")
    out.mkdir(parents=True)
    proto,proto_sha=write_protocol(out,b2)

    atomic_json(out/"R31B3_PRE_GT_VERIFICATION.json",{
        "status":"PASS_R31B3_PRE_GT_VERIFICATION","version":VERSION,
        "R31B2_final_sha256":R31B2_FINAL_SHA,"R31B2_semantic_lock_sha256":b2["semantic_lock_sha256"],
        "R31A_source_action_table":source_info,"TENT_PL_rows":len(tentpl),
        "protocol_sha256":proto_sha,"evaluation_gt_read_hash_decode":[False,False,False],
        "external_data_access":False,
    })
    print("\nPRE-GT LOCK VERIFICATION: PASS")
    print("  R31B2 final PRE-GT SHA :",R31B2_FINAL_SHA)
    print("  Frozen TENT/PL rows    :",len(tentpl))
    print("  Frozen MEMO rows       :",ROWS_PER_ACTION)
    print("  Protocol SHA256        :",proto_sha)
    print("  Evaluation GT access   : NO / NO / NO up to this point")

    memo,gtm=reveal_memo(out,b2,qnames,dnames)

    required=["sample_id","model_state_id","model_family","action",*qnames,*dnames,"r31b3_harm_label"]
    for name,d in [("TENTPL",tentpl),("MEMO",memo)]:
        miss=[c for c in required if c not in d.columns]
        if miss: raise RuntimeError(f"{name} missing={miss[:10]}")
        d["action"]=d["action"].map(norm_action); d["sample_id"]=d["sample_id"].astype(str)
    panel=pd.concat([tentpl[required],memo[required]],ignore_index=True)
    if len(panel)!=TOTAL_ROWS: raise RuntimeError(f"3-action rows={len(panel)}")
    align=check_q(panel,qnames)
    atomic_csv(panel,out/"R31B3_THREE_ACTION_MODELCASE_TABLE.csv")

    hs=(panel.groupby("action",as_index=False).agg(model_cases=("r31b3_harm_label","size"),
                                                   harm_count=("r31b3_harm_label","sum"),
                                                   harm_rate=("r31b3_harm_label","mean")))
    atomic_csv(hs,out/"R31B3_THREE_ACTION_HARM_PREVALENCE.csv")

    preds=[]; mets=[]
    for seed in tqdm(CV_SEEDS,desc="R31B3 grouped-CV seeds",unit="seed",dynamic_ncols=True):
        p,m=run_seed(panel,qnames,dnames,seed); preds.append(p); mets.append(m)
    pred=pd.concat(preds,ignore_index=True); per=pd.concat(mets,ignore_index=True); summ=summarize(per); dec=decide(summ)

    atomic_csv(pred,out/"R31B3_OOF_PREDICTIONS.csv")
    atomic_csv(per,out/"R31B3_PER_SEED_METRICS.csv")
    atomic_csv(summ,out/"R31B3_METRIC_SUMMARY.csv")
    dec.update({"version":VERSION,"protocol_lock_sha256":proto_sha,"q_source_alignment":align,
                "three_action_harm_prevalence":hs.to_dict(orient="records"),
                "post_gt_tuning":False,"external_data_access":False})
    atomic_json(out/"R31B3_DECISION.json",dec)
    atomic_csv(inventory(out),out/"R31B3_SHA256_INVENTORY.csv")

    final={"status":"PASS_R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO_COMPLETE",
           "version":VERSION,"decision":dec["decision"],"R31B2_final_pre_gt_sha256":R31B2_FINAL_SHA,
           "protocol_lock_sha256":proto_sha,"GT_read_hash_decode":[True,True,True],
           "post_gt_tuning":False,"external_data_access":False,"next":dec["next"]}
    fp=out/"R31B3_FINAL_LOCK.json"; atomic_json(fp,final)

    print("\n"+"="*124); print("R31B3 THREE-ACTION HARM PREVALENCE"); print("="*124); print(hs.to_string(index=False))
    print("\n"+"="*124); print("R31B3 KEY RESULTS"); print("="*124)
    show=summ[(summ.scope=="ACTION") & summ.model.isin(
        ["SOURCE_ONLY_Q66","SHARED_TRANSITION","SHARED_ACTION_CONDITIONAL",
         "SEPARATE_ACTION","PRIMARY_LOAO_ACTION_CONDITIONAL","LOAO_TRANSITION_ONLY"])]
    print(show[["model","action","auroc_mean","auroc_std","auprc_mean","auprc_std","prevalence_mean","auprc_lift_mean"]].to_string(index=False))
    print("\nR31B3 DECISION:",dec["decision"])
    print("  primary LOAO macro AUROC      :",dec["metrics"]["primary_loao_macro_auroc"])
    print("  primary LOAO min-action AUROC :",dec["metrics"]["primary_loao_min_action_auroc"])
    print("  held-out action AUROCs        :",dec["metrics"]["heldout_action_aurocs"])
    print("  shared - separate             :",dec["metrics"]["shared_minus_separate_macro_action_auroc"])
    print("  q_source max alignment diff   :",align["max_abs_q_source_difference"])
    print("\nFINAL STATUS : PASS_R31B3_FIRST_800CASE_GT_REVEAL_AND_THREE_ACTION_LOAO_COMPLETE")
    print("800-case GT read/hash/decode : YES / YES / YES (FIRST REVEAL)")
    print("Post-GT tuning              : NO")
    print("External cohort access      : NO")
    print("Final lock SHA256           :",sha256_file(fp))
    print("Output                      :",out)
    print("NEXT                        :",dec["next"])
    print("="*124)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
