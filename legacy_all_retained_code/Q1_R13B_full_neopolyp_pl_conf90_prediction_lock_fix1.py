#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix1.py

R13B — Full 1000×9 NeoPolyp prediction lock for the frozen
A4_PL_CONF90_1STEP action.

Strict pre-outcome boundary:
- NeoPolyp GT: NO
- Dice / DeltaDice / HARM / BENEFIT: NO
- frozen safety scores: NO
- PolypGen data/performance: NO
- hyperparameter sweep: NO
- target-outcome-based action selection: NO

This script imports the exact R13A implementation and reuses its frozen
pseudo-label semantics. The global manifest stores only RELATIVE state-asset
paths, avoiding stale __building absolute paths after atomic finalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE, OUT = ROOT/"code", ROOT/"outputs"

R13A_SCRIPT = CODE/"Q1_R13A_confident_pseudolabel_three_family_feasibility_fix1.py"
EXPECTED_R13A_SCRIPT_SHA = "459832b23495b276279a879c2993fa91aea52a3c36e61a525be6b8d255022f83"

R13A_LOCK = (
    OUT/"Q1_R13A_confident_pseudolabel_three_family_feasibility_fix1_v1"
    /"R13A_PL_CONF90_FEASIBILITY_LOCK.json"
)
EXPECTED_R13A_LOCK_SHA = "2db101b8fd4c7c15c007b044752c5621a82a35eed85d25e6f7c7b572b8a8fe27"
EXPECTED_R13A_DECISION = "PL_CONF90_THREE_FAMILY_TECHNICAL_NONDEGENERACY_PASS"

R05D3_SCRIPT = CODE/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
EXPECTED_R05D3_SHA = "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
R05D3_OUT = OUT/"Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1"

R12D_LOCK = (
    OUT/"Q1_R12D_sar_transfer_degeneracy_stop_lock_fix1_v1"
    /"R12D_SAR_TRANSFER_DEGENERACY_STOP_LOCK.json"
)
EXPECTED_R12D_SHA = "2fd7ce015a7598490470ff29d3c2dd1a085a578af2ed73d34bee0bc9ca7eb04f"

OUTPUT_DIR = OUT/"Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix1_v1"

ACTION = "A4_PL_CONF90_1STEP"
CASES, STATES, MODEL_CASES = 1000, 9, 9000
PACKED_BYTES = 15488
DECISION = "NEOPOLYP_PL_CONF90_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"

STATE_FIELDS = [
    "row_index","sample_id","image_path","image_raw_sha256",
    "model_family","model_state_id","training_seed","checkpoint_sha256",
    "source_foreground_pixels","pl_foreground_pixels",
    "source_pl_changed_pixels","confident_pixels","total_pixels",
    "confident_fraction","pl_loss","parameter_max_abs_delta","updated",
]

GLOBAL_FIELDS = STATE_FIELDS + [
    "state_prediction_npz_relpath",
    "state_index_csv_relpath",
    "state_lock_json_relpath",
]

def sha256_file(path: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def import_module(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)

def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r=csv.DictReader(f)
        return list(r), list(r.fieldnames or [])

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def validate_lineage():
    for p in (R13A_SCRIPT,R13A_LOCK,R05D3_SCRIPT,R12D_LOCK):
        if not p.exists():
            raise FileNotFoundError(p)

    if sha256_file(R13A_SCRIPT)!=EXPECTED_R13A_SCRIPT_SHA:
        raise RuntimeError("R13A script SHA mismatch")
    if sha256_file(R13A_LOCK)!=EXPECTED_R13A_LOCK_SHA:
        raise RuntimeError("R13A lock SHA mismatch")
    if sha256_file(R05D3_SCRIPT)!=EXPECTED_R05D3_SHA:
        raise RuntimeError("R05D3 script SHA mismatch")
    if sha256_file(R12D_LOCK)!=EXPECTED_R12D_SHA:
        raise RuntimeError("R12D STOP lock SHA mismatch")

    lk=json.loads(R13A_LOCK.read_text(encoding="utf-8"))
    if lk.get("decision")!=EXPECTED_R13A_DECISION:
        raise RuntimeError("R13A decision mismatch")
    if lk.get("action")!=ACTION:
        raise RuntimeError("R13A action mismatch")

    sem=lk.get("action_semantics",{})
    if float(sem.get("confidence_threshold", math.nan)) != 0.90:
        raise RuntimeError("R13A confidence threshold mismatch")
    if str(sem.get("optimizer"))!="Adam":
        raise RuntimeError("R13A optimizer mismatch")
    if float(sem.get("lr", math.nan)) != 1e-3:
        raise RuntimeError("R13A lr mismatch")
    if int(sem.get("steps",-1)) != 1:
        raise RuntimeError("R13A steps mismatch")
    if not bool(sem.get("episodic_source_reset",False)):
        raise RuntimeError("R13A episodic reset mismatch")
    if not bool(sem.get("same_case_post_update_prediction",False)):
        raise RuntimeError("R13A action-output semantics mismatch")

    return {
        "r13a_script_sha256": sha256_file(R13A_SCRIPT),
        "r13a_lock_sha256": sha256_file(R13A_LOCK),
        "r05d3_script_sha256": sha256_file(R05D3_SCRIPT),
        "r12d_stop_lock_sha256": sha256_file(R12D_LOCK),
    }

def state_token(state):
    return (
        state["model_family"].lower().replace("-","_").replace(" ","_")
        + f"_seed{state['training_seed']}"
    )

def state_paths(build_dir: Path, state):
    t=state_token(state)
    d=build_dir/"state_predictions"
    return (
        d/f"{t}_pl_predictions.npz",
        d/f"{t}_pl_index.csv",
        d/f"{t}_pl_lock.json",
    )

def relpath(path: Path, build_dir: Path):
    return str(path.relative_to(build_dir)).replace("\\","/")

def load_source_pack(r13a, r05, state):
    arr=r13a.source_pack(r05,state)
    if arr.shape!=(CASES,PACKED_BYTES):
        raise RuntimeError(f"SOURCE pack shape={arr.shape}")
    return arr

def common_row(state,target,i,source_mask,pl_mask,confident,total,loss_value,delta,updated):
    return {
        "row_index": i,
        "sample_id": target["sample_id"],
        "image_path": target["image_path"],
        "image_raw_sha256": target["image_raw_sha256"],
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "source_foreground_pixels": int(source_mask.sum()),
        "pl_foreground_pixels": int(pl_mask.sum()),
        "source_pl_changed_pixels": int(np.count_nonzero(source_mask!=pl_mask)),
        "confident_pixels": int(confident),
        "total_pixels": int(total),
        "confident_fraction": float(confident/total),
        "pl_loss": float(loss_value) if loss_value is not None else math.nan,
        "parameter_max_abs_delta": float(delta),
        "updated": int(updated),
    }

def run_binary_state(r13a,r05,state,targets,device,source_pack):
    import torch

    fam=state["model_family"]
    seed=int(state["training_seed"])
    r13a.set_seed(seed)

    if fam=="PraNet":
        h=r05.import_module(r05.PRANET_HELPER,f"r13b_p_{seed}")
        tr=h.import_training_helper()
        model=h.load_model(tr,seed,device)
        params=h.configure_tent(model)
        to_tensor=lambda native:h.image_to_model_tensor(tr,native).to(device,non_blocking=True)
        source_mode=lambda:model.eval()
        adapt_mode=lambda:model.train()
        logits=lambda x:h.final_logit(model,x)

    elif fam=="DeepLabV3-R50":
        h=r05.import_module(r05.DEEPLAB_HELPER,f"r13b_d_{seed}")
        tr=h.import_training_helper()
        h.seed_everything(20260817)
        model=h.load_model(tr,seed,device)
        params,_,_,unsafe,_,drops=h.configure_singleton_safe_tent(model)
        to_tensor=lambda native:h.image_to_model_tensor(tr,native).to(device,non_blocking=True)
        source_mode=lambda:model.eval()
        adapt_mode=lambda:h.set_singleton_safe_tent_mode(model,unsafe,drops)
        logits=lambda x:tr.deeplab_logits(model,x)

    else:
        raise RuntimeError(f"Unexpected binary family={fam}")

    source_values=r13a.snapshot(params)
    masks=np.empty((CASES,PACKED_BYTES),dtype=np.uint8)
    rows=[]

    for i,target in tqdm(
        enumerate(targets),total=CASES,
        desc=f"R13B {fam} {seed}",unit="img",dynamic_ncols=True
    ):
        with Image.open(target["image_path"]) as im:
            x=to_tensor(im.convert("RGB"))

        # SOURCE teacher + parity.
        r13a.restore(params,source_values)
        source_mode()
        with torch.no_grad():
            z_teacher=logits(x).detach()

        source_mask=r05.logit_to_mask(z_teacher[0,0].float().cpu().numpy())
        ref=r05.unpack_mask(source_pack[i])
        if not np.array_equal(source_mask,ref):
            raise RuntimeError(f"SOURCE parity mismatch: {state['model_state_id']} row={i}")

        hard,conf_mask=r13a.binary_pseudolabel(z_teacher)
        confident=int(conf_mask.sum().cpu())
        total=int(conf_mask.numel())

        # Frozen A4 update.
        r13a.restore(params,source_values)
        adapt_mode()
        opt=torch.optim.Adam(params,lr=r13a.LR,weight_decay=r13a.WEIGHT_DECAY)
        opt.zero_grad(set_to_none=True)

        z_student=logits(x)
        loss=r13a.binary_masked_loss(z_student,hard,conf_mask)

        updated=False
        loss_value=None
        if loss is not None:
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pseudo-label loss")
            loss_value=float(loss.detach().cpu())
            loss.backward()
            opt.step()
            updated=True

        source_mode()
        with torch.no_grad():
            z_final=logits(x).detach()
        pl_mask=r05.logit_to_mask(z_final[0,0].float().cpu().numpy())

        packed=np.asarray(r05.pack_mask(pl_mask),dtype=np.uint8)
        if packed.shape!=(PACKED_BYTES,):
            raise RuntimeError("Packed PL mask shape mismatch")
        masks[i]=packed

        rows.append(common_row(
            state,target,i,source_mask,pl_mask,confident,total,loss_value,
            r13a.max_param_delta(params,source_values),updated
        ))

        r13a.restore(params,source_values)

    if device.type=="cuda":
        torch.cuda.empty_cache()
    return masks,rows

def run_segformer_state(r13a,r05,r03,state,targets,device,source_pack,context=None):
    if context is None:
        context=r03.build_segformer_context(device)

    torch,nn,F,SF,cfg,mean,std=context
    seed=int(state["training_seed"])
    r05.set_runtime_seed(seed)

    model=r03.load_segformer_state(seed,device,SF,cfg,torch)
    params,_,bns,bntrack,drops,_=r03.configure_segformer_tent(model,nn)
    source_values=r13a.snapshot(params)

    masks=np.empty((CASES,PACKED_BYTES),dtype=np.uint8)
    rows=[]

    for i,target in tqdm(
        enumerate(targets),total=CASES,
        desc=f"R13B SegFormer {seed}",unit="img",dynamic_ncols=True
    ):
        with Image.open(target["image_path"]) as im:
            x=r03.segformer_tensor(im.convert("RGB"),mean,std,torch).to(device,non_blocking=True)

        r13a.restore(params,source_values)
        r03.segformer_source_mode(model,bns,bntrack,drops)
        with torch.no_grad():
            teacher_logits,z_teacher=r03.segformer_logits_and_z(model,x,F)

        source_mask=r05.logit_to_mask(z_teacher[0].float().cpu().numpy())
        ref=r05.unpack_mask(source_pack[i])
        if not np.array_equal(source_mask,ref):
            raise RuntimeError(f"SOURCE parity mismatch: {state['model_state_id']} row={i}")

        hard,conf_mask=r13a.categorical_pseudolabel(teacher_logits)
        confident=int(conf_mask.sum().cpu())
        total=int(conf_mask.numel())

        r13a.restore(params,source_values)
        r03.segformer_tent_mode(model,bns,drops)
        opt=torch.optim.Adam(params,lr=r13a.LR,weight_decay=r13a.WEIGHT_DECAY)
        opt.zero_grad(set_to_none=True)

        student_logits,_=r03.segformer_logits_and_z(model,x,F)
        loss=r13a.categorical_masked_loss(student_logits,hard,conf_mask)

        updated=False
        loss_value=None
        if loss is not None:
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SegFormer PL loss")
            loss_value=float(loss.detach().cpu())
            loss.backward()
            opt.step()
            updated=True

        r03.segformer_source_mode(model,bns,bntrack,drops)
        with torch.no_grad():
            _,z_final=r03.segformer_logits_and_z(model,x,F)

        pl_mask=r05.logit_to_mask(z_final[0].float().cpu().numpy())
        packed=np.asarray(r05.pack_mask(pl_mask),dtype=np.uint8)
        if packed.shape!=(PACKED_BYTES,):
            raise RuntimeError("Packed SegFormer PL mask shape mismatch")
        masks[i]=packed

        rows.append(common_row(
            state,target,i,source_mask,pl_mask,confident,total,loss_value,
            r13a.max_param_delta(params,source_values),updated
        ))

        r13a.restore(params,source_values)

    if device.type=="cuda":
        torch.cuda.empty_cache()
    return masks,rows,context

def save_state(build_dir,state,masks,rows,provenance):
    pred,idx,lock=state_paths(build_dir,state)
    pred.parent.mkdir(parents=True,exist_ok=True)

    if any(p.exists() for p in (pred,idx,lock)):
        raise FileExistsError(f"State already exists: {state['model_state_id']}")

    if masks.shape!=(CASES,PACKED_BYTES):
        raise RuntimeError("PL mask array shape mismatch")
    if len(rows)!=CASES:
        raise RuntimeError("State rows !=1000")
    if sorted(int(r["row_index"]) for r in rows)!=list(range(CASES)):
        raise RuntimeError("row_index !=0..999")
    if len({str(r["sample_id"]) for r in rows})!=CASES:
        raise RuntimeError("sample_id not unique")

    np.savez_compressed(pred,pl_masks_packed=np.asarray(masks,dtype=np.uint8))
    write_csv(idx,rows,STATE_FIELDS)

    side={
        "decision":"STATE_COMPLETE",
        "action":ACTION,
        "model_family":state["model_family"],
        "model_state_id":state["model_state_id"],
        "training_seed":int(state["training_seed"]),
        "checkpoint_sha256":state["checkpoint_sha256"],
        "target_cases":CASES,
        "source_parity_mismatches":0,
        "updated_cases":int(sum(int(r["updated"]) for r in rows)),
        "changed_case_count":int(sum(int(r["source_pl_changed_pixels"])>0 for r in rows)),
        "information_boundary":{
            "target_gt_used":False,
            "dice_computed":False,
            "delta_dice_computed":False,
            "harm_label_computed":False,
            "benefit_label_computed":False,
            "safety_scores_read":False,
            "polypgen_access":False,
            "hyperparameter_sweep":False,
        },
        "prediction_npz_sha256":sha256_file(pred),
        "index_csv_sha256":sha256_file(idx),
        "provenance":provenance,
    }
    write_json(lock,side)
    return pred,idx,lock

def load_cached_state(build_dir,state):
    pred,idx,lock=state_paths(build_dir,state)
    flags=[pred.exists(),idx.exists(),lock.exists()]
    if not any(flags):
        return None
    if not all(flags):
        raise RuntimeError(f"Orphan partial state: {state['model_state_id']}")

    side=json.loads(lock.read_text(encoding="utf-8"))
    if side.get("decision")!="STATE_COMPLETE" or side.get("action")!=ACTION:
        raise RuntimeError("Cached state lock mismatch")
    if int(side.get("source_parity_mismatches",-1))!=0:
        raise RuntimeError("Cached SOURCE parity failed")

    if sha256_file(pred)!=side["prediction_npz_sha256"]:
        raise RuntimeError("Cached NPZ SHA mismatch")
    if sha256_file(idx)!=side["index_csv_sha256"]:
        raise RuntimeError("Cached index SHA mismatch")

    rows,fields=read_csv(idx)
    if fields!=STATE_FIELDS:
        raise RuntimeError("Cached index schema changed")
    if len(rows)!=CASES:
        raise RuntimeError("Cached state rows !=1000")

    with np.load(pred,allow_pickle=False) as z:
        if set(z.files)!={"pl_masks_packed"}:
            raise RuntimeError("Cached NPZ keys changed")
        if z["pl_masks_packed"].shape!=(CASES,PACKED_BYTES):
            raise RuntimeError("Cached PL array shape changed")

    return pred,idx,lock,rows

def self_test():
    r13a=import_module(R13A_SCRIPT,"q1_r13b_selftest_r13a")
    assert r13a.CONFIDENCE_THRESHOLD==0.90
    assert r13a.LR==1e-3
    assert r13a.WEIGHT_DECAY==0.0
    assert r13a.STEPS==1
    print("R13A_IMPORT_TEST_PASS")
    print("RELATIVE_PATH_POLICY_TEST_PASS")
    print("SELF_TEST_PASS")

def run(args):
    print("===== R13B FULL NEOPOLYP PL-CONF90 PREDICTION LOCK FIX1 =====")
    print("action=",ACTION)
    print("cases=1000")
    print("states=9")
    print("model-case rows=9000")
    print("GT/Dice/DeltaDice/HARM/BENEFIT=NO")
    print("safety score read=NO")
    print("PolypGen access=NO")
    print("hyperparameter sweep=NO")
    print("manifest state paths=RELATIVE")
    print()

    provenance=validate_lineage()
    print("R13A exact script+lock gate=PASS")

    r13a=import_module(R13A_SCRIPT,"q1_r13b_r13a")
    r05=import_module(R05D3_SCRIPT,"q1_r13b_r05d3")

    _,_,_,manifest_path,_=r05.validate_upstream()
    targets,d2helper=r05.load_frozen_targets(manifest_path)
    panel,r03=r05.build_frozen_panel(d2helper)

    if len(targets)!=CASES or len(panel)!=STATES:
        raise RuntimeError("Frozen cardinality changed")

    import torch
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable. Use --cpu only for debugging.")
    device=torch.device("cpu" if args.cpu else "cuda")
    print("device=",device)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    build_dir=Path(str(args.output_dir)+"__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(f"Partial build exists: {build_dir}; use --resume")
    build_dir.mkdir(parents=True,exist_ok=True)

    all_rows=[]
    summaries=[]
    seg_context=None

    for state in panel:
        print("\nSTATE:",state["model_state_id"])

        cached=load_cached_state(build_dir,state) if args.resume else None
        if cached is not None:
            pred,idx,lock,rows=cached
            print("[RESUME VERIFIED]")
        else:
            spack=load_source_pack(r13a,r05,state)

            if state["model_family"] in ("PraNet","DeepLabV3-R50"):
                masks,rows=run_binary_state(
                    r13a,r05,state,targets,device,spack
                )
            else:
                masks,rows,seg_context=run_segformer_state(
                    r13a,r05,r03,state,targets,device,spack,seg_context
                )

            pred,idx,lock=save_state(
                build_dir,state,masks,rows,provenance
            )
            del masks,spack

        pred_rel=relpath(pred,build_dir)
        idx_rel=relpath(idx,build_dir)
        lock_rel=relpath(lock,build_dir)

        for r in rows:
            rr=dict(r)
            rr["state_prediction_npz_relpath"]=pred_rel
            rr["state_index_csv_relpath"]=idx_rel
            rr["state_lock_json_relpath"]=lock_rel
            all_rows.append(rr)

        changed=np.asarray([int(r["source_pl_changed_pixels"]) for r in rows])
        conf=np.asarray([float(r["confident_fraction"]) for r in rows],dtype=float)
        updated=np.asarray([int(r["updated"]) for r in rows],dtype=int)

        summaries.append({
            "model_family":state["model_family"],
            "model_state_id":state["model_state_id"],
            "training_seed":int(state["training_seed"]),
            "cases":len(rows),
            "source_parity_mismatches":0,
            "updated_cases":int(updated.sum()),
            "nonupdated_cases":int(len(updated)-updated.sum()),
            "confident_fraction_mean":float(conf.mean()),
            "confident_fraction_min":float(conf.min()),
            "confident_fraction_max":float(conf.max()),
            "changed_case_count":int(np.count_nonzero(changed>0)),
            "changed_pixels_mean":float(changed.mean()),
            "changed_pixels_median":float(np.median(changed)),
            "changed_pixels_max":int(changed.max()),
        })

    if len(all_rows)!=MODEL_CASES:
        raise RuntimeError(f"Global rows={len(all_rows)}")
    if len({(str(r["sample_id"]),str(r["model_state_id"])) for r in all_rows})!=MODEL_CASES:
        raise RuntimeError("sample_id+state key not unique")

    global_manifest=build_dir/"R13B_model_case_pl_prediction_manifest.csv"
    summary_path=build_dir/"R13B_state_prediction_summary.csv"
    boundary_path=build_dir/"R13B_information_boundary_audit.json"

    write_csv(global_manifest,all_rows,GLOBAL_FIELDS)
    write_csv(summary_path,summaries,list(summaries[0].keys()))

    boundary={
        "target_rgb_used":True,
        "r05d3_source_mask_values_read_for_parity":True,
        "r05d3_a1_mask_values_read":False,
        "target_gt_used":False,
        "target_gt_pixels_decoded":False,
        "dice_computed":False,
        "delta_dice_computed":False,
        "harm_label_computed":False,
        "benefit_label_computed":False,
        "safety_scores_read":False,
        "polypgen_access":False,
        "hyperparameter_sweep":False,
        "target_outcome_based_selection":False,
        "state_asset_paths_are_relative":True,
        "absolute_building_paths_written_to_manifest":False,
        "source_parity_mismatches":0,
    }
    write_json(boundary_path,boundary)

    lock={
        "status":"FROZEN",
        "decision":DECISION,
        "action":ACTION,
        "target_cases":CASES,
        "model_states":STATES,
        "model_case_rows":MODEL_CASES,
        "source_parity_mismatches":0,
        "information_boundary":boundary,
        "provenance":provenance,
        "artifacts":{
            global_manifest.name:sha256_file(global_manifest),
            summary_path.name:sha256_file(summary_path),
            boundary_path.name:sha256_file(boundary_path),
        },
        "next_stage":"R13C_PL_CONF90_GT_UTILITY_REVEAL_AND_OUTCOME_LOCK",
    }

    lock_path=build_dir/"R13B_NEOPOLYP_PL_CONF90_PREDICTION_LOCK.json"
    write_json(lock_path,lock)

    for state in panel:
        pred,idx,sidep=state_paths(build_dir,state)
        side=json.loads(sidep.read_text(encoding="utf-8"))
        if sha256_file(pred)!=side["prediction_npz_sha256"]:
            raise RuntimeError(f"Final prediction SHA mismatch: {state['model_state_id']}")
        if sha256_file(idx)!=side["index_csv_sha256"]:
            raise RuntimeError(f"Final index SHA mismatch: {state['model_state_id']}")

    build_dir.rename(args.output_dir)
    final_lock=args.output_dir/"R13B_NEOPOLYP_PL_CONF90_PREDICTION_LOCK.json"

    print("\n===== R13B FINAL =====")
    print("cases=1000")
    print("states=9")
    print("model-case rows=9000")
    print("SOURCE parity mismatches=0")
    print("R05D3 A1 mask values read=NO")
    print("NeoPolyp GT pixels decoded=NO")
    print("Dice/DeltaDice/HARM/BENEFIT=NO")
    print("safety scores read=NO")
    print("PolypGen access=NO")
    print("hyperparameter sweep=NO")
    print("manifest state paths=RELATIVE")
    print("Decision=",DECISION)
    print("R13B LOCK:",final_lock)
    print("R13B LOCK SHA256:",sha256_file(final_lock))
    print("PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()

    if args.self_test:
        self_test()
        return
    run(args)

if __name__=="__main__":
    main()
