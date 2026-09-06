#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE, OUT = ROOT/"code", ROOT/"outputs"

R12B_DIR = OUT/"Q1_R12B_full_neopolyp_sar_prediction_lock_fix1_v1"
R12B_LOCK = R12B_DIR/"R12B_NEOPOLYP_SAR_PREDICTION_LOCK.json"
R12B_MANIFEST = R12B_DIR/"R12B_model_case_sar_prediction_manifest.csv"
R12B_SHA = "d9c92dbbede2a90c8fc3e968d884335a50bbd37ef0406be80e4976ae4aea7639"
R12B_DEC = "NEOPOLYP_SAR_DENSE_EPISODIC_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"

R05D4_SCRIPT = CODE/"Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1.py"
R05D4_DIR = OUT/"Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
R05D4_LOCK = R05D4_DIR/"Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
R05D4_OUT = R05D4_DIR/"model_case_outcomes.csv"

OUTDIR = OUT/"Q1_R12C_sar_gt_utility_reveal_and_outcome_lock_fix1_v1"

CASES, STATES, ROWS = 1000, 9, 9000
H, W, PACKED = 352, 352, 15488
HARM_THR, BENEFIT_THR = -0.02, 0.02
ACTION = "A3_SAR_DENSE_EPISODIC_1STEP"
DECISION = "NEOPOLYP_SAR_GT_UTILITY_AND_OUTCOMES_LOCKED"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def imod(p,n):
    s=importlib.util.spec_from_file_location(n,str(p))
    if s is None or s.loader is None: raise RuntimeError(p)
    m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m

def unpack(x):
    x=np.asarray(x,dtype=np.uint8)
    if x.shape!=(PACKED,): raise RuntimeError(x.shape)
    return np.unpackbits(x,count=H*W,bitorder="little").reshape(H,W).astype(np.uint8,copy=False)

def outcome(d):
    if d<=HARM_THR: return "HARM"
    if d>=BENEFIT_THR: return "BENEFIT"
    return "NEUTRAL"

def verify_r12b():
    if not R12B_LOCK.exists(): raise FileNotFoundError(R12B_LOCK)
    if sha(R12B_LOCK)!=R12B_SHA: raise RuntimeError("R12B lock SHA mismatch")
    lock=json.loads(R12B_LOCK.read_text(encoding="utf-8"))
    if lock.get("decision")!=R12B_DEC: raise RuntimeError("R12B decision mismatch")
    if (int(lock.get("target_cases",-1)),int(lock.get("model_states",-1)),int(lock.get("model_case_rows",-1)))!=(CASES,STATES,ROWS):
        raise RuntimeError("R12B cardinality mismatch")
    if int(lock.get("source_parity_mismatches",-1))!=0: raise RuntimeError("R12B SOURCE parity failed")
    if int(lock.get("second_pass_subset_violations",-1))!=0: raise RuntimeError("R12B nested filter failed")
    ib=lock.get("information_boundary",{})
    for k in ("r05d3_a1_mask_values_read","target_gt_used","target_gt_pixels_decoded","dice_computed",
              "delta_dice_computed","harm_label_computed","benefit_label_computed",
              "safety_scores_read","polypgen_data_or_performance_read","hyperparameter_sweep",
              "target_outcome_based_selection","absolute_building_paths_written_to_manifest"):
        if bool(ib.get(k,True)): raise RuntimeError(f"R12B boundary failed: {k}")
    if not bool(ib.get("state_asset_paths_are_relative",False)):
        raise RuntimeError("R12B relative path policy missing")
    if not R12B_MANIFEST.exists(): raise FileNotFoundError(R12B_MANIFEST)
    exp=lock.get("artifacts",{}).get(R12B_MANIFEST.name)
    if not exp or sha(R12B_MANIFEST)!=exp: raise RuntimeError("R12B manifest SHA mismatch")
    return lock

def load_gt_old():
    if not R05D4_SCRIPT.exists() or not R05D4_LOCK.exists() or not R05D4_OUT.exists():
        raise FileNotFoundError("R05D4 artifact missing")
    lk=json.loads(R05D4_LOCK.read_text(encoding="utf-8"))
    if int(lk.get("target_cases",-1))!=CASES or int(lk.get("model_case_rows",-1))!=ROWS:
        raise RuntimeError("R05D4 cardinality mismatch")
    if not bool(lk.get("target_gt_pixels_revealed",False)):
        raise RuntimeError("R05D4 GT reveal not confirmed")
    d4=imod(R05D4_SCRIPT,"q1_r12c_r05d4")
    _,_,_,mp,_,gp=d4.validate_upstream()
    d4.validate_gt_rule(gp)
    gm=d4.load_d1_manifest(mp)
    gt,_=d4.reveal_gt(gm)
    old=pd.read_csv(R05D4_OUT,usecols=[
        "sample_id","model_family","model_state_id","training_seed","checkpoint_sha256",
        "source_dice","a1_dice","delta_dice","adaptation_outcome","harm_label","benefit_label"
    ],low_memory=False)
    if len(old)!=ROWS: raise RuntimeError("R05D4 rows !=9000")
    old=old.rename(columns={
        "a1_dice":"tent1_dice","delta_dice":"tent1_delta_dice",
        "adaptation_outcome":"tent1_adaptation_outcome",
        "harm_label":"tent1_harm_label","benefit_label":"tent1_benefit_label"
    })
    return d4,gt,old

def rel(v):
    p=Path(str(v))
    if p.is_absolute(): raise RuntimeError(f"Absolute R12B path found: {p}")
    q=R12B_DIR/p
    if not q.exists(): raise FileNotFoundError(q)
    return q

def build_panel(d4,gt,old):
    man=pd.read_csv(R12B_MANIFEST,low_memory=False)
    req=["row_index","sample_id","model_family","model_state_id","training_seed","checkpoint_sha256",
         "state_prediction_npz_relpath","state_index_csv_relpath","state_lock_json_relpath"]
    miss=[c for c in req if c not in man.columns]
    if miss: raise RuntimeError(f"R12B manifest missing {miss}")
    if len(man)!=ROWS or man[["sample_id","model_state_id"]].drop_duplicates().shape[0]!=ROWS:
        raise RuntimeError("R12B manifest key/cardinality failure")
    rows=[]
    for sid,g in man.groupby("model_state_id",sort=True):
        if len(g)!=CASES: raise RuntimeError(f"{sid}: rows={len(g)}")
        pv=g.state_prediction_npz_relpath.astype(str).unique()
        iv=g.state_index_csv_relpath.astype(str).unique()
        lv=g.state_lock_json_relpath.astype(str).unique()
        if len(pv)!=1 or len(iv)!=1 or len(lv)!=1: raise RuntimeError(f"{sid}: ambiguous assets")
        pred,idxp,sidep=rel(pv[0]),rel(iv[0]),rel(lv[0])
        side=json.loads(sidep.read_text(encoding="utf-8"))
        if side.get("decision")!="STATE_COMPLETE" or side.get("action")!=ACTION: raise RuntimeError(f"{sid}: state lock")
        if int(side.get("source_parity_mismatches",-1))!=0 or int(side.get("second_pass_subset_violations",-1))!=0:
            raise RuntimeError(f"{sid}: state audit failed")
        for k in ("target_gt_used","dice_computed","delta_dice_computed","harm_label_computed",
                  "benefit_label_computed","safety_scores_read","polypgen_data_or_performance_read","hyperparameter_sweep"):
            if bool(side.get("information_boundary",{}).get(k,True)): raise RuntimeError(f"{sid}: boundary {k}")
        if sha(pred)!=side["prediction_npz_sha256"] or sha(idxp)!=side["index_csv_sha256"]:
            raise RuntimeError(f"{sid}: state SHA mismatch")
        idx=pd.read_csv(idxp,low_memory=False)
        g=g.copy()
        g["row_index"]=pd.to_numeric(g["row_index"],errors="raise").astype(int)
        idx["row_index"]=pd.to_numeric(idx["row_index"],errors="raise").astype(int)
        if sorted(g.row_index.tolist())!=list(range(CASES)) or sorted(idx.row_index.tolist())!=list(range(CASES)):
            raise RuntimeError(f"{sid}: row_index failure")
        L=g[["row_index","sample_id"]].assign(sample_id=lambda x:x.sample_id.astype(str)).sort_values("row_index").reset_index(drop=True)
        R=idx[["row_index","sample_id"]].assign(sample_id=lambda x:x.sample_id.astype(str)).sort_values("row_index").reset_index(drop=True)
        if not L.equals(R): raise RuntimeError(f"{sid}: manifest/index mismatch")
        with np.load(pred,allow_pickle=False) as z:
            if set(z.files)!={"sar_masks_packed"}: raise RuntimeError(f"{sid}: NPZ keys")
            masks=np.asarray(z["sar_masks_packed"],dtype=np.uint8)
        if masks.shape!=(CASES,PACKED): raise RuntimeError(f"{sid}: mask shape {masks.shape}")
        for r in tqdm(g.sort_values("row_index").itertuples(index=False),total=CASES,desc=f"R12C {sid}",unit="case",dynamic_ncols=True):
            sample=str(r.sample_id); i=int(r.row_index)
            dice=float(d4.binary_dice(unpack(masks[i]),gt[sample]))
            rows.append(dict(sample_id=sample,model_family=str(r.model_family),model_state_id=str(r.model_state_id),
                             training_seed=int(r.training_seed),checkpoint_sha256=str(r.checkpoint_sha256),sar_dice=dice))
    sar=pd.DataFrame(rows)
    panel=sar.merge(old,on=["sample_id","model_state_id"],how="left",validate="one_to_one",suffixes=("_sar","_old"))
    if len(panel)!=ROWS: raise RuntimeError("SAR/R05D4 join !=9000")
    for c in ("model_family","training_seed","checkpoint_sha256"):
        if not (panel[f"{c}_sar"].astype(str)==panel[f"{c}_old"].astype(str)).all():
            raise RuntimeError(f"metadata mismatch: {c}")
        panel[c]=panel[f"{c}_sar"]
    panel["sar_delta_dice"]=panel.sar_dice.to_numpy(float)-panel.source_dice.to_numpy(float)
    panel["sar_adaptation_outcome"]=[outcome(x) for x in panel.sar_delta_dice]
    panel["sar_harm_label"]=panel.sar_adaptation_outcome.eq("HARM").astype(int)
    panel["sar_benefit_label"]=panel.sar_adaptation_outcome.eq("BENEFIT").astype(int)
    return panel.drop(columns=[f"{c}_{s}" for c in ("model_family","training_seed","checkpoint_sha256") for s in ("sar","old")])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test:
        assert outcome(-.02)=="HARM" and outcome(.02)=="BENEFIT" and outcome(0)=="NEUTRAL"
        print("SELF_TEST_PASS"); return
    print("===== R12C SAR GT UTILITY REVEAL + OUTCOME LOCK FIX1 =====")
    print("safety scores read=NO"); print("fit/refit/recalibration/score reversal=NO"); print("PolypGen access=NO"); print("SAR retuning=NO")
    verify_r12b(); print("R12B exact pre-GT lock gate=PASS")
    d4,gt,old=load_gt_old(); print("R05D4 GT/outcome lineage=PASS")
    panel=build_panel(d4,gt,old)
    fam=(panel.groupby("model_family",as_index=False).agg(
        rows=("sample_id","size"),sar_harm=("sar_harm_label","sum"),
        sar_harm_prevalence=("sar_harm_label","mean"),sar_delta_mean=("sar_delta_dice","mean"),
        sar_delta_median=("sar_delta_dice","median")))
    print("\n===== SAR OUTCOME SUMMARY =====")
    print("rows=",len(panel),"cases=",panel.sample_id.nunique(),"states=",panel.model_state_id.nunique())
    print("SAR HARM=",int(panel.sar_harm_label.sum()))
    print("SAR NEUTRAL=",int(panel.sar_adaptation_outcome.eq("NEUTRAL").sum()))
    print("SAR BENEFIT=",int(panel.sar_benefit_label.sum()))
    print("SAR HARM prevalence=",float(panel.sar_harm_label.mean()))
    print("SAR DeltaDice mean/median=",float(panel.sar_delta_dice.mean()),float(panel.sar_delta_dice.median()))
    print("TENT1↔SAR HARM agreement=",float(np.mean(panel.tent1_harm_label.to_numpy(int)==panel.sar_harm_label.to_numpy(int))))
    print("TENT1↔SAR DeltaDice Spearman=",float(panel.tent1_delta_dice.corr(panel.sar_delta_dice,method="spearman")))
    print("\n===== BY FAMILY ====="); print(fam.to_string(index=False))
    if OUTDIR.exists(): raise FileExistsError(OUTDIR)
    OUTDIR.mkdir(parents=True)
    p1=OUTDIR/"R12C_SAR_model_case_outcomes.csv"; p2=OUTDIR/"R12C_SAR_family_summary.csv"
    p3=OUTDIR/"R12C_TENT1_to_SAR_harm_transition.csv"; p4=OUTDIR/"R12C_TENT1_to_SAR_tristate_transition.csv"
    panel.to_csv(p1,index=False); fam.to_csv(p2,index=False)
    pd.crosstab(panel.tent1_harm_label,panel.sar_harm_label,rownames=["TENT1_harm"],colnames=["SAR_harm"]).to_csv(p3)
    pd.crosstab(panel.tent1_adaptation_outcome,panel.sar_adaptation_outcome,rownames=["TENT1_outcome"],colnames=["SAR_outcome"]).to_csv(p4)
    lock={"decision":DECISION,"r12b_lock_sha256":sha(R12B_LOCK),"r05d4_lock_sha256":sha(R05D4_LOCK),
          "target_cases":CASES,"model_states":STATES,"model_case_rows":ROWS,
          "sar_harm_rows":int(panel.sar_harm_label.sum()),"sar_neutral_rows":int(panel.sar_adaptation_outcome.eq("NEUTRAL").sum()),
          "sar_benefit_rows":int(panel.sar_benefit_label.sum()),"sar_harm_prevalence":float(panel.sar_harm_label.mean()),
          "sar_delta_dice_mean":float(panel.sar_delta_dice.mean()),"sar_delta_dice_median":float(panel.sar_delta_dice.median()),
          "information_boundary":{"safety_scores_read":False,"safety_model_fit_or_refit":False,"threshold_recalibration":False,
          "score_reversal":False,"polypgen_access":False,"sar_hyperparameter_tuning":False},
          "artifacts":{p.name:sha(p) for p in (p1,p2,p3,p4)},"next_stage":"R12D_FROZEN_SAFETY_SCORE_TRANSFER_TO_SAR"}
    lp=OUTDIR/"R12C_SAR_OUTCOME_LOCK.json"; lp.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print("\nDecision=",DECISION); print("R12C LOCK:",lp); print("R12C LOCK SHA256:",sha(lp)); print("PASS")

if __name__=="__main__": main()
