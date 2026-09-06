#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE, OUT = ROOT/"code", ROOT/"outputs"

R05D3 = CODE/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
R05D3_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
R05D3_OUT = OUT/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"

R11D = OUT/"Q1_R11D_tent2_frozen_safety_transfer_evaluation_fix4_v1"/"R11D_TENT2_SAFETY_TRANSFER_EVALUATION_LOCK.json"
R11D_SHA = "0c3be4849affa169f1476541dd59a953782eaec2925a2b156e0783fb4611f70f"
R11D_DEC = "TENT2_FROZEN_SAFETY_RANKING_TRANSFER_SUPPORTED"

POLYPGEN = OUT/"Q1_R10L3C_polypgen_gt_reveal_prospective_external_evaluation_fix1_v1"/"R10L3C_POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_LOCK.json"
POLYPGEN_DEC = "POLYPGEN_PROSPECTIVE_EXTERNAL_EVALUATION_COMPLETE_NO_RETUNING"

OUTDIR = OUT/"Q1_R12A_sar_dense_episdic_three_family_feasibility_fix1_v1"

ACTION = "A3_SAR_DENSE_EPISODIC_1STEP"
PROBES = (0,142,285,428,570,713,856,999)
CASES, STATES, PACKED = 1000, 9, 15488
MARGIN = 0.4*math.log(2.0)
RHO, MOM, WD = 0.05, 0.9, 0.0
LR_CNN, LR_TRANS = 1.5625e-5, 3.125e-5
PASS_DEC = "SAR_DENSE_EPISODIC_THREE_FAMILY_TECHNICAL_FEASIBILITY_PASS"
STOP_DEC = "SAR_DENSE_EPISODIC_TECHNICAL_FEASIBILITY_STOP"
EPS = 1e-12

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def imod(p,n):
    s=importlib.util.spec_from_file_location(n,str(p))
    if s is None or s.loader is None: raise RuntimeError(p)
    m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m

def seed_all(s):
    random.seed(s); np.random.seed(s%(2**32-1))
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def snap(ps): return [p.detach().clone() for p in ps]

def restore(ps,vs):
    import torch
    with torch.no_grad():
        for p,v in zip(ps,vs): p.copy_(v)

def pdelta(ps,vs):
    return max([float((p.detach()-v).abs().max().cpu()) for p,v in zip(ps,vs)] or [0.0])

def bent(z):
    import torch
    p=torch.sigmoid(z)
    return -(p*torch.log(p.clamp_min(EPS))+(1-p)*torch.log((1-p).clamp_min(EPS)))

def cent(z):
    import torch
    p=torch.softmax(z,1); return -(p*torch.log_softmax(z,1)).sum(1)

def rloss(e):
    m=e<MARGIN; n=int(m.sum().detach().cpu()); total=int(m.numel())
    return (None,n,total) if n==0 else (e[m].mean(),n,total)

class SAM:
    def __init__(self, ps, lr):
        import torch
        self.ps=list(ps); self.rho=RHO; self.e={}
        self.opt=torch.optim.SGD(self.ps,lr=lr,momentum=MOM,weight_decay=WD)
    def zero(self): self.opt.zero_grad(set_to_none=True)
    def first(self):
        import torch
        ns=[p.grad.detach().norm(2) for p in self.ps if p.grad is not None]
        g=torch.norm(torch.stack(ns),2) if ns else torch.tensor(0.,device=self.ps[0].device)
        gv=float(g.detach().cpu())
        if gv<=0: return 0.,0.
        sc=self.rho/(g+1e-12); sq=0.
        with torch.no_grad():
            for p in self.ps:
                if p.grad is None: continue
                e=p.grad*sc.to(p); p.add_(e); self.e[p]=e.detach().clone()
                sq+=float((e.detach()**2).sum().cpu())
        self.zero(); return gv,math.sqrt(sq)
    def second(self):
        import torch
        with torch.no_grad():
            for p,e in self.e.items(): p.sub_(e)
        self.opt.step(); self.zero(); self.e.clear()
    def cancel(self):
        import torch
        with torch.no_grad():
            for p,e in self.e.items(): p.sub_(e)
        self.zero(); self.e.clear()

def lineage():
    for p in (R05D3,R11D,POLYPGEN):
        if not p.exists(): raise FileNotFoundError(p)
    if sha(R05D3)!=R05D3_SHA: raise RuntimeError("R05D3 SHA mismatch")
    if sha(R11D)!=R11D_SHA: raise RuntimeError("R11D SHA mismatch")
    if json.loads(R11D.read_text(encoding="utf-8")).get("decision")!=R11D_DEC:
        raise RuntimeError("R11D decision mismatch")
    if json.loads(POLYPGEN.read_text(encoding="utf-8")).get("decision")!=POLYPGEN_DEC:
        raise RuntimeError("PolypGen lock mismatch")
    return {"r05d3":sha(R05D3),"r11d":sha(R11D),"polypgen_lock":sha(POLYPGEN)}

def source_pack(r05,state):
    d=R05D3_OUT/"state_predictions"
    pred,idx,lock=r05.state_paths(d,state)
    side=json.loads(lock.read_text(encoding="utf-8"))
    if sha(pred)!=side["prediction_npz_sha256"]: raise RuntimeError("SOURCE NPZ SHA")
    with np.load(pred,allow_pickle=False) as z:
        x=np.asarray(z["source_masks_packed"],dtype=np.uint8).copy()
    if x.shape!=(CASES,PACKED): raise RuntimeError(x.shape)
    return x

def row(state,target,i,lr,par,n1,n2,tot,l1,l2,gn,pn,dd,upd,chg):
    return dict(probe_index=i,sample_id=target["sample_id"],
        model_family=state["model_family"],model_state_id=state["model_state_id"],
        training_seed=int(state["training_seed"]),lr=lr,source_parity=int(par),
        reliable_pixels_first=n1,reliable_pixels_second=n2,total_pixels=tot,
        reliable_fraction_first=n1/tot,loss_first=l1,loss_second=l2,
        grad_norm_first=gn,sam_perturb_norm=pn,parameter_max_abs_delta=dd,
        updated=int(upd),source_to_sar_changed_pixels=chg)

def binary_family(r05,state,targets,dev,spack):
    import torch
    fam=state["model_family"]; sd=int(state["training_seed"]); seed_all(sd)
    if fam=="PraNet":
        h=r05.import_module(r05.PRANET_HELPER,f"r12p{sd}"); tr=h.import_training_helper()
        model=h.load_model(tr,sd,dev); ps=h.configure_tent(model)
        tensor=lambda im:h.image_to_model_tensor(tr,im).to(dev)
        srcmode=lambda:model.eval(); adpmode=lambda:model.train()
        logits=lambda x:h.final_logit(model,x)
    else:
        h=r05.import_module(r05.DEEPLAB_HELPER,f"r12d{sd}"); tr=h.import_training_helper()
        h.seed_everything(20260817); model=h.load_model(tr,sd,dev)
        ps,_,_,unsafe,_,drops=h.configure_singleton_safe_tent(model)
        tensor=lambda im:h.image_to_model_tensor(tr,im).to(dev)
        srcmode=lambda:model.eval()
        adpmode=lambda:h.set_singleton_safe_tent_mode(model,unsafe,drops)
        logits=lambda x:tr.deeplab_logits(model,x)
    sv=snap(ps); rows=[]
    for i in tqdm(PROBES,desc=f"R12A {fam} {sd}",unit="case",dynamic_ncols=True):
        t=targets[i]
        with Image.open(t["image_path"]) as im: x=tensor(im.convert("RGB"))
        restore(ps,sv); srcmode()
        with torch.no_grad(): zs=logits(x).detach()
        sm=r05.logit_to_mask(zs[0,0].float().cpu().numpy())
        ref=r05.unpack_mask(spack[i]); par=np.array_equal(sm,ref)
        if not par: raise RuntimeError(f"SOURCE parity {fam} {sd} {i}")
        restore(ps,sv); adpmode(); sam=SAM(ps,LR_CNN); sam.zero()
        z1=logits(x); l1,n1,tot=rloss(bent(z1))
        upd=False; l1v=l2v=math.nan; gn=pn=0.; n2=0
        if l1 is not None:
            if not torch.isfinite(l1): raise RuntimeError("loss1")
            l1v=float(l1.detach().cpu()); l1.backward(); gn,pn=sam.first()
            z2=logits(x); l2,n2,_=rloss(bent(z2))
            if l2 is not None and torch.isfinite(l2):
                l2v=float(l2.detach().cpu()); l2.backward(); sam.second(); upd=True
            else: sam.cancel()
        srcmode()
        with torch.no_grad(): zf=logits(x).detach()
        fm=r05.logit_to_mask(zf[0,0].float().cpu().numpy())
        rows.append(row(state,t,i,LR_CNN,par,n1,n2,tot,l1v,l2v,gn,pn,
                        pdelta(ps,sv),upd,int(np.count_nonzero(sm!=fm))))
        restore(ps,sv)
    if dev.type=="cuda": torch.cuda.empty_cache()
    return rows

def segformer(r05,r03,state,targets,dev,spack,ctx=None):
    if ctx is None: ctx=r03.build_segformer_context(dev)
    torch,nn,F,SF,cfg,mean,std=ctx; sd=int(state["training_seed"]); r05.set_runtime_seed(sd)
    model=r03.load_segformer_state(sd,dev,SF,cfg,torch)
    ps,_,bns,bntrack,drops,_=r03.configure_segformer_tent(model,nn); sv=snap(ps); rows=[]
    for i in tqdm(PROBES,desc=f"R12A SegFormer {sd}",unit="case",dynamic_ncols=True):
        t=targets[i]
        with Image.open(t["image_path"]) as im: x=r03.segformer_tensor(im.convert("RGB"),mean,std,torch).to(dev)
        restore(ps,sv); r03.segformer_source_mode(model,bns,bntrack,drops)
        with torch.no_grad(): _,zs=r03.segformer_logits_and_z(model,x,F)
        sm=r05.logit_to_mask(zs[0].float().cpu().numpy()); ref=r05.unpack_mask(spack[i])
        par=np.array_equal(sm,ref)
        if not par: raise RuntimeError(f"SOURCE parity SegFormer {sd} {i}")
        restore(ps,sv); r03.segformer_tent_mode(model,bns,drops); sam=SAM(ps,LR_TRANS); sam.zero()
        z1,_=r03.segformer_logits_and_z(model,x,F); l1,n1,tot=rloss(cent(z1))
        upd=False; l1v=l2v=math.nan; gn=pn=0.; n2=0
        if l1 is not None:
            if not torch.isfinite(l1): raise RuntimeError("loss1")
            l1v=float(l1.detach().cpu()); l1.backward(); gn,pn=sam.first()
            z2,_=r03.segformer_logits_and_z(model,x,F); l2,n2,_=rloss(cent(z2))
            if l2 is not None and torch.isfinite(l2):
                l2v=float(l2.detach().cpu()); l2.backward(); sam.second(); upd=True
            else: sam.cancel()
        r03.segformer_source_mode(model,bns,bntrack,drops)
        with torch.no_grad(): _,zf=r03.segformer_logits_and_z(model,x,F)
        fm=r05.logit_to_mask(zf[0].float().cpu().numpy())
        rows.append(row(state,t,i,LR_TRANS,par,n1,n2,tot,l1v,l2v,gn,pn,
                        pdelta(ps,sv),upd,int(np.count_nonzero(sm!=fm))))
        restore(ps,sv)
    if dev.type=="cuda": torch.cuda.empty_cache()
    return rows,ctx

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--cpu",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        assert abs(MARGIN-0.4*math.log(2))<1e-15 and RHO==0.05 and len(PROBES)==8
        print("SELF_TEST_PASS"); return
    print("===== R12A SAR-DENSE EPISODIC THREE-FAMILY FEASIBILITY FIX1 =====")
    print("action=",ACTION); print("GT/Dice/HARM=NO"); print("safety score read=NO")
    print("PolypGen data/performance read=NO"); print("hyperparameter sweep=NO")
    print("probe indices=",PROBES); print("margin=",MARGIN,"rho=",RHO)
    print("CNN lr=",LR_CNN,"Transformer lr=",LR_TRANS)
    prov=lineage(); r05=imod(R05D3,"r12_r05")
    _,_,_,mp,_=r05.validate_upstream(); targets,d2=r05.load_frozen_targets(mp); panel,r03=r05.build_frozen_panel(d2)
    import torch
    if not torch.cuda.is_available() and not a.cpu: raise RuntimeError("CUDA unavailable")
    dev=torch.device("cpu" if a.cpu else "cuda"); rows=[]; ctx=None
    for st in panel:
        sp=source_pack(r05,st)
        if st["model_family"] in ("PraNet","DeepLabV3-R50"):
            rows+=binary_family(r05,st,targets,dev,sp)
        else:
            rr,ctx=segformer(r05,r03,st,targets,dev,sp,ctx); rows+=rr
    df=pd.DataFrame(rows)
    ss=(df.groupby(["model_family","model_state_id","training_seed"],as_index=False)
        .agg(probe_cases=("updated","size"),source_parity_mismatches=("source_parity",lambda x:int((x==0).sum())),
             updated_cases=("updated","sum"),reliable_fraction_mean=("reliable_fraction_first","mean"),
             parameter_delta_max=("parameter_max_abs_delta","max"),
             changed_pixels_mean=("source_to_sar_changed_pixels","mean")))
    fs=(df.groupby("model_family",as_index=False)
        .agg(probe_rows=("updated","size"),updated_rows=("updated","sum"),
             reliable_fraction_mean=("reliable_fraction_first","mean"),
             parameter_delta_max=("parameter_max_abs_delta","max")))
    ok=bool((df.source_parity==1).all() and (ss.updated_cases>=1).all() and (ss.parameter_delta_max>0).all())
    dec=PASS_DEC if ok else STOP_DEC
    print("\n===== STATE SUMMARY ====="); print(ss.to_string(index=False))
    print("\n===== FAMILY SUMMARY ====="); print(fs.to_string(index=False))
    print("\nDecision:",dec)
    if OUTDIR.exists(): raise FileExistsError(OUTDIR)
    OUTDIR.mkdir(parents=True)
    p1=OUTDIR/"R12A_probe_detail.csv"; p2=OUTDIR/"R12A_state_summary.csv"; p3=OUTDIR/"R12A_family_summary.csv"
    df.to_csv(p1,index=False); ss.to_csv(p2,index=False); fs.to_csv(p3,index=False)
    lock={"decision":dec,"action":ACTION,"probe_indices":list(PROBES),"margin":MARGIN,
          "sam":{"rho":RHO,"momentum":MOM,"cnn_lr":LR_CNN,"transformer_lr":LR_TRANS},
          "information_boundary":{"target_gt_used":False,"dice_computed":False,"harm_label_computed":False,
          "safety_scores_read":False,"polypgen_data_or_performance_read":False,"hyperparameter_sweep":False},
          "provenance":prov,"next_stage":"R12B_FULL_NEOPOLYP_SAR_PREDICTION_LOCK" if ok else "STOP_SAR_BRANCH"}
    lp=OUTDIR/"R12A_SAR_DENSE_FEASIBILITY_LOCK.json"; lp.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print("R12A LOCK:",lp); print("R12A LOCK SHA256:",sha(lp)); print("PASS")

if __name__=="__main__": main()
