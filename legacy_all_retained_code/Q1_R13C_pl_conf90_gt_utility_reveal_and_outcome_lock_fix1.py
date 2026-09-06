#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT=Path(r"F:\MEDSEG_SAFETTA")
CODE,OUT=ROOT/"code",ROOT/"outputs"
R13B_SCRIPT=CODE/"Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix2.py"
EXPECTED_R13B_SCRIPT_SHA="6f7ce6a3b15abf1205dc77703320ba823b79d392dc4f43beb51e0a7b94a2521d"
R13B_DIR=OUT/"Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix2_v1"
R13B_LOCK=R13B_DIR/"R13B_NEOPOLYP_PL_CONF90_PREDICTION_LOCK.json"
R13B_MANIFEST=R13B_DIR/"R13B_model_case_pl_prediction_manifest.csv"
EXPECTED_R13B_LOCK_SHA="d1275dba4d4e898c3cdff503ad94ae58a43a4793faf6ee7ca418fb8c39d8e944"
EXPECTED_R13B_DECISION="NEOPOLYP_PL_CONF90_FULL_EPISODIC_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"

R05D4_SCRIPT=CODE/"Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1.py"
R05D4_DIR=OUT/"Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1"
R05D4_LOCK=R05D4_DIR/"Q1_R05D4_NEOPOLYP_CONFIRMATORY_EVALUATION_LOCK.json"
R05D4_OUTCOMES=R05D4_DIR/"model_case_outcomes.csv"

OUTPUT_DIR=OUT/"Q1_R13C_pl_conf90_gt_utility_reveal_and_outcome_lock_fix1_v1"
ACTION="A4_PL_CONF90_1STEP"
CASES,STATES,ROWS=1000,9,9000
H,W,PACKED=352,352,15488
HARM_THR,BENEFIT_THR=-0.02,0.02
DECISION="NEOPOLYP_PL_CONF90_GT_UTILITY_AND_OUTCOMES_LOCKED"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def imod(p,n):
    s=importlib.util.spec_from_file_location(n,str(p))
    if s is None or s.loader is None: raise RuntimeError(p)
    m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m

def unpack(x):
    x=np.asarray(x,dtype=np.uint8)
    if x.shape!=(PACKED,): raise RuntimeError(f"packed shape={x.shape}")
    return np.unpackbits(x,count=H*W,bitorder="little").reshape(H,W).astype(np.uint8,copy=False)

def outcome(d):
    if d<=HARM_THR: return "HARM"
    if d>=BENEFIT_THR: return "BENEFIT"
    return "NEUTRAL"

def rel(v):
    p=Path(str(v))
    if p.is_absolute(): raise RuntimeError(f"absolute R13B path: {p}")
    q=R13B_DIR/p
    if not q.exists(): raise FileNotFoundError(q)
    return q

def verify_r13b():
    for p in (R13B_SCRIPT,R13B_LOCK,R13B_MANIFEST):
        if not p.exists(): raise FileNotFoundError(p)
    if sha(R13B_SCRIPT)!=EXPECTED_R13B_SCRIPT_SHA: raise RuntimeError("R13B script SHA mismatch")
    if sha(R13B_LOCK)!=EXPECTED_R13B_LOCK_SHA: raise RuntimeError("R13B lock SHA mismatch")
    lk=json.loads(R13B_LOCK.read_text(encoding="utf-8"))
    if lk.get("decision")!=EXPECTED_R13B_DECISION or lk.get("action")!=ACTION:
        raise RuntimeError("R13B decision/action mismatch")
    if (int(lk.get("target_cases",-1)),int(lk.get("model_states",-1)),int(lk.get("model_case_rows",-1)))!=(CASES,STATES,ROWS):
        raise RuntimeError("R13B cardinality mismatch")
    if int(lk.get("source_parity_mismatches",-1))!=0: raise RuntimeError("R13B parity failed")
    ep=lk.get("episodic_reset",{})
    if not ep.get("trainable_parameters",False) or not ep.get("all_model_buffers",False):
        raise RuntimeError("R13B episodic reset audit missing")
    ib=lk.get("information_boundary",{})
    for k in ("r05d3_a1_mask_values_read","target_gt_used","target_gt_pixels_decoded","dice_computed","delta_dice_computed","harm_label_computed","benefit_label_computed","safety_scores_read","polypgen_access","hyperparameter_sweep","target_outcome_based_selection","r13b_fix1_partial_outputs_reused","absolute_building_paths_written_to_manifest"):
        if bool(ib.get(k,True)): raise RuntimeError(f"R13B boundary failed: {k}")
    if not ib.get("state_asset_paths_are_relative",False): raise RuntimeError("relative path policy missing")
    exp=lk.get("artifacts",{}).get(R13B_MANIFEST.name)
    if not exp or sha(R13B_MANIFEST)!=exp: raise RuntimeError("R13B manifest SHA mismatch")
    return lk

def load_gt_old():
    d4=imod(R05D4_SCRIPT,"q1_r13c_r05d4")
    lk=json.loads(R05D4_LOCK.read_text(encoding="utf-8"))
    if int(lk.get("target_cases",-1))!=CASES or int(lk.get("model_case_rows",-1))!=ROWS:
        raise RuntimeError("R05D4 cardinality mismatch")
    _,_,_,mp,_,gp=d4.validate_upstream()
    d4.validate_gt_rule(gp)
    gm=d4.load_d1_manifest(mp)
    gt,_=d4.reveal_gt(gm)
    old=pd.read_csv(R05D4_OUTCOMES,usecols=[
        "sample_id","model_family","model_state_id","training_seed","checkpoint_sha256",
        "source_dice","a1_dice","delta_dice","adaptation_outcome","harm_label","benefit_label"
    ],low_memory=False)
    old=old.rename(columns={
        "a1_dice":"tent1_dice","delta_dice":"tent1_delta_dice",
        "adaptation_outcome":"tent1_adaptation_outcome",
        "harm_label":"tent1_harm_label","benefit_label":"tent1_benefit_label"})
    return d4,gt,old

def build_panel(d4,gt,old):
    man=pd.read_csv(R13B_MANIFEST,low_memory=False)
    if len(man)!=ROWS or man[["sample_id","model_state_id"]].drop_duplicates().shape[0]!=ROWS:
        raise RuntimeError("R13B manifest key/cardinality failed")
    rows=[]
    for state_id,g in man.groupby("model_state_id",sort=True):
        pv=g.state_prediction_npz_relpath.astype(str).unique()
        iv=g.state_index_csv_relpath.astype(str).unique()
        lv=g.state_lock_json_relpath.astype(str).unique()
        if len(pv)!=1 or len(iv)!=1 or len(lv)!=1: raise RuntimeError(f"{state_id}: ambiguous assets")
        pred,idxp,sidep=rel(pv[0]),rel(iv[0]),rel(lv[0])
        side=json.loads(sidep.read_text(encoding="utf-8"))
        if side.get("decision")!="STATE_COMPLETE" or side.get("action")!=ACTION: raise RuntimeError(f"{state_id}: state lock")
        if int(side.get("source_parity_mismatches",-1))!=0: raise RuntimeError(f"{state_id}: parity")
        ep=side.get("episodic_reset",{})
        if not ep.get("trainable_parameters",False) or not ep.get("all_model_buffers",False): raise RuntimeError(f"{state_id}: reset flags")
        if sha(pred)!=side["prediction_npz_sha256"] or sha(idxp)!=side["index_csv_sha256"]:
            raise RuntimeError(f"{state_id}: SHA mismatch")
        idx=pd.read_csv(idxp,low_memory=False)
        g=g.copy()
        g["row_index"]=pd.to_numeric(g["row_index"],errors="raise").astype(int)
        idx["row_index"]=pd.to_numeric(idx["row_index"],errors="raise").astype(int)
        if sorted(g.row_index.tolist())!=list(range(CASES)) or sorted(idx.row_index.tolist())!=list(range(CASES)):
            raise RuntimeError(f"{state_id}: row index")
        L=g[["row_index","sample_id"]].assign(sample_id=lambda x:x.sample_id.astype(str)).sort_values("row_index").reset_index(drop=True)
        R=idx[["row_index","sample_id"]].assign(sample_id=lambda x:x.sample_id.astype(str)).sort_values("row_index").reset_index(drop=True)
        if not L.equals(R): raise RuntimeError(f"{state_id}: manifest/index mapping")
        with np.load(pred,allow_pickle=False) as z:
            if set(z.files)!={"pl_masks_packed"}: raise RuntimeError(f"{state_id}: NPZ keys")
            masks=np.asarray(z["pl_masks_packed"],dtype=np.uint8)
        if masks.shape!=(CASES,PACKED): raise RuntimeError(f"{state_id}: masks {masks.shape}")
        for r in tqdm(g.sort_values("row_index").itertuples(index=False),total=CASES,desc=f"R13C {state_id}",unit="case",dynamic_ncols=True):
            sid=str(r.sample_id); i=int(r.row_index)
            dice=float(d4.binary_dice(unpack(masks[i]),gt[sid]))
            rows.append(dict(sample_id=sid,model_family=str(r.model_family),model_state_id=str(r.model_state_id),training_seed=int(r.training_seed),checkpoint_sha256=str(r.checkpoint_sha256),pl_dice=dice))
    pl=pd.DataFrame(rows)
    panel=pl.merge(old,on=["sample_id","model_state_id"],how="left",validate="one_to_one",suffixes=("_pl","_old"))
    for c in ("model_family","training_seed","checkpoint_sha256"):
        if not (panel[f"{c}_pl"].astype(str)==panel[f"{c}_old"].astype(str)).all(): raise RuntimeError(f"metadata mismatch {c}")
        panel[c]=panel[f"{c}_pl"]
    panel["pl_delta_dice"]=panel.pl_dice.to_numpy(float)-panel.source_dice.to_numpy(float)
    panel["pl_adaptation_outcome"]=[outcome(x) for x in panel.pl_delta_dice]
    panel["pl_harm_label"]=panel.pl_adaptation_outcome.eq("HARM").astype(int)
    panel["pl_benefit_label"]=panel.pl_adaptation_outcome.eq("BENEFIT").astype(int)
    return panel.drop(columns=[f"{c}_{s}" for c in ("model_family","training_seed","checkpoint_sha256") for s in ("pl","old")])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    if args.self_test:
        assert outcome(-.02)=="HARM" and outcome(.02)=="BENEFIT" and outcome(0)=="NEUTRAL"
        print("SELF_TEST_PASS"); return
    print("===== R13C PL-CONF90 GT UTILITY REVEAL + OUTCOME LOCK FIX1 =====")
    print("safety score read=NO"); print("fit/refit/recalibration/score reversal=NO"); print("PolypGen access=NO"); print("PL retuning=NO")
    verify_r13b(); print("R13B exact pre-GT lock gate=PASS")
    d4,gt,old=load_gt_old(); print("R05D4 GT/outcome lineage=PASS")
    panel=build_panel(d4,gt,old)
    harm=int(panel.pl_harm_label.sum()); benefit=int(panel.pl_benefit_label.sum()); neutral=int(panel.pl_adaptation_outcome.eq("NEUTRAL").sum())
    unique_classes=int(panel.pl_harm_label.nunique())
    evaluable=bool(unique_classes==2 and harm>0 and harm<ROWS)
    fam=(panel.groupby("model_family",as_index=False).agg(
        rows=("sample_id","size"),pl_harm=("pl_harm_label","sum"),pl_benefit=("pl_benefit_label","sum"),
        pl_harm_prevalence=("pl_harm_label","mean"),pl_delta_mean=("pl_delta_dice","mean"),
        pl_delta_median=("pl_delta_dice","median"),pl_delta_min=("pl_delta_dice","min"),pl_delta_max=("pl_delta_dice","max")))
    print("\n===== PL OUTCOME SUMMARY =====")
    print("rows=",len(panel),"cases=",panel.sample_id.nunique(),"states=",panel.model_state_id.nunique())
    print("PL HARM=",harm); print("PL NEUTRAL=",neutral); print("PL BENEFIT=",benefit)
    print("PL HARM prevalence=",float(panel.pl_harm_label.mean()))
    print("PL DeltaDice mean/median=",float(panel.pl_delta_dice.mean()),float(panel.pl_delta_dice.median()))
    print("PL DeltaDice min/max=",float(panel.pl_delta_dice.min()),float(panel.pl_delta_dice.max()))
    print("TENT1↔PL HARM agreement=",float(np.mean(panel.tent1_harm_label.to_numpy(int)==panel.pl_harm_label.to_numpy(int))))
    print("TENT1↔PL DeltaDice Spearman=",float(panel.tent1_delta_dice.corr(panel.pl_delta_dice,method="spearman")))
    print("R13D transfer evaluable=",evaluable)
    print("\n===== BY FAMILY ====="); print(fam.to_string(index=False))
    if OUTPUT_DIR.exists(): raise FileExistsError(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    p1=OUTPUT_DIR/"R13C_PL_model_case_outcomes.csv"; p2=OUTPUT_DIR/"R13C_PL_family_summary.csv"
    p3=OUTPUT_DIR/"R13C_TENT1_to_PL_harm_transition.csv"; p4=OUTPUT_DIR/"R13C_TENT1_to_PL_tristate_transition.csv"
    panel.to_csv(p1,index=False); fam.to_csv(p2,index=False)
    pd.crosstab(panel.tent1_harm_label,panel.pl_harm_label,rownames=["TENT1_harm"],colnames=["PL_harm"]).to_csv(p3)
    pd.crosstab(panel.tent1_adaptation_outcome,panel.pl_adaptation_outcome,rownames=["TENT1_outcome"],colnames=["PL_outcome"]).to_csv(p4)
    lock={"decision":DECISION,"action":ACTION,"r13b_script_sha256":sha(R13B_SCRIPT),"r13b_lock_sha256":sha(R13B_LOCK),
          "target_cases":CASES,"model_states":STATES,"model_case_rows":ROWS,
          "pl_harm_rows":harm,"pl_neutral_rows":neutral,"pl_benefit_rows":benefit,
          "pl_harm_prevalence":float(panel.pl_harm_label.mean()),"pl_harm_unique_classes":unique_classes,
          "r13d_transfer_evaluable":evaluable,"pl_delta_dice_mean":float(panel.pl_delta_dice.mean()),
          "pl_delta_dice_median":float(panel.pl_delta_dice.median()),"pl_delta_dice_min":float(panel.pl_delta_dice.min()),
          "pl_delta_dice_max":float(panel.pl_delta_dice.max()),
          "information_boundary":{"safety_scores_read":False,"safety_model_fit_or_refit":False,"threshold_recalibration":False,
          "score_reversal":False,"polypgen_access":False,"pl_hyperparameter_tuning":False},
          "artifacts":{p.name:sha(p) for p in (p1,p2,p3,p4)},
          "next_stage":"R13D_FROZEN_PRE_ADAPTATION_SAFETY_SCORE_TRANSFER_TO_PL" if evaluable else "R13D_PL_TRANSFER_DEGENERACY_STOP"}
    lp=OUTPUT_DIR/"R13C_PL_OUTCOME_LOCK.json"; lp.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print("\nDecision=",DECISION); print("R13D transfer evaluable=",evaluable); print("R13C LOCK:",lp); print("R13C LOCK SHA256:",sha(lp)); print("PASS")

if __name__=="__main__":
    main()
