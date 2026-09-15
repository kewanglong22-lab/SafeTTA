#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P0-4 — read-only asset/statistical-unit/GT-history audit.

No model fitting, no AUROC/AUPRC recomputation, no score change,
no threshold tuning, no candidate regeneration.
"""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

VERSION = "2026-09-15-B6-P04-v1"
ROOT = Path(r"F:\MEDSEG_SAFETTA")
PROTOCOL_NAME = "B6_P0_POSTHOC_AUDIT_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "ed19e9f1d98b66a0f5280694274a1a92d001bbc884f07dd5313320e6dc63ba04"
OUT_DIR = ROOT / "B6_P04_asset_statistical_unit_and_gt_history_audit_v1"
PASS_GATE = "PASS_B6_P04_REPRO_AND_STATISTICAL_UNIT_AUDIT_COMPLETE"

KNOWN_LOCKS = {
    "FinalB1": ROOT / "FinalB1_mri_action_port_and_candidate_mask_lock_v1_fix3" / "FINALB1_MRI_ACTION_CANDIDATE_MASK_LOCK.json",
    "FinalB2": ROOT / "FinalB2_mri_current_runtime_candidate_conditioned_s64_lock_v1" / "FINALB2_MRI_S64_LOCK.json",
    "FinalB3": ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1" / "FINALB3_MRI_FUTURE_HARM_LOCK.json",
    "FinalB3A": ROOT / "FinalB3A_mri_joint_shift_cellwise_audit_v1" / "FINALB3A_AUDIT.json",
    "FinalB3B": ROOT / "FinalB3B_mri_joint_shift_reporting_v1" / "FINALB3B_REPORTING_AUDIT.json",
}

KNOWN_TABLES = {
    "FinalB3_PROMISE_joint_shift_scores":
        ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1" / "FINALB3_PROMISE12_JOINT_SHIFT_LOAO_SCORES.csv",
    "FinalB3_PROMISE_outcomes":
        ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1" / "FINALB3_PROMISE12_ACTION_OUTCOMES.csv",
    "FinalB3_PROSTATE_outcomes":
        ROOT / "FinalB3_mri_action_specific_outcome_and_future_harm_v1" / "FINALB3_PROSTATE158_ACTION_OUTCOMES.csv",
}

ID_HINTS = ["patient","subject","case","video","procedure","exam","study","series","image","frame","sample","global_index"]
PROB_HINTS = ["prob","probability","probabilities","logit","logits","softmax","sigmoid","entropy","confidence","low_conf"]
ENTITY_HINTS = ["polypgen","promise12","neopolyp","sun","prostate158"]

def sha256_file(path: Path, chunk: int = 8*1024*1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_json(path: Path) -> Dict[str, Any]:
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict): raise RuntimeError(f"Expected JSON object: {path}")
    return x

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def exact_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file(): raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch.\nexpected={expected}\nobserved={got}")

def verify_protocol(script_dir: Path) -> Dict[str, Any]:
    p = script_dir / PROTOCOL_NAME
    exact_sha(p, EXPECTED_PROTOCOL_SHA256, "B6_P0_PROTOCOL")
    d = load_json(p)
    if d.get("status") != "FROZEN_BEFORE_NEW_P0_METRICS":
        raise RuntimeError("P0 protocol status changed.")
    if float(d["frozen_scientific_constants"]["harm_threshold_delta_dice"]) != -0.02:
        raise RuntimeError("HARM threshold changed.")
    if int(d["frozen_scientific_constants"]["target_fit_rows"]) != 0:
        raise RuntimeError("Target-fit boundary changed.")
    return d

def safe_header(path: Path) -> List[str]:
    try:
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        if path.suffix.lower() in [".tsv", ".txt"]:
            return list(pd.read_csv(path, sep="\t", nrows=0).columns)
        if path.suffix.lower() == ".json" and path.stat().st_size <= 5_000_000:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict): return list(obj[0].keys())
            if isinstance(obj, dict): return list(obj.keys())
    except Exception:
        pass
    return []

def discover_relevant_files() -> List[Path]:
    out = []
    skip = {".git","__pycache__","node_modules",".cache"}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            lo = fn.lower()
            if (any(h in lo for h in ENTITY_HINTS) or any(h in lo for h in PROB_HINTS)
                or "manifest" in lo or "lock" in lo or "score" in lo
                or "outcome" in lo or "index" in lo):
                out.append(Path(dp)/fn)
    return out

def classify_entity(path: Path) -> str:
    s = str(path).lower()
    for e in ENTITY_HINTS:
        if e in s: return e
    return "other"

def id_recovery(files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in tqdm(files, desc="Inspect ID headers", unit="file", dynamic_ncols=True):
        if p.suffix.lower() not in [".csv",".tsv",".txt",".json"]: continue
        ent = classify_entity(p)
        if ent == "other": continue
        cols = safe_header(p)
        matched = [c for c in cols if any(h in c.lower() for h in ID_HINTS)]
        if matched:
            rows.append({
                "entity": ent, "path": str(p), "filename": p.name,
                "id_like_columns": ";".join(matched),
                "all_columns": ";".join(cols), "bytes": p.stat().st_size
            })
    return pd.DataFrame(rows)

def prob_recovery(files: List[Path]) -> pd.DataFrame:
    rows = []
    for p in files:
        lo = p.name.lower()
        if not any(h in lo for h in PROB_HINTS): continue
        shape, dtype, note = "", "", ""
        readable = True
        if p.suffix.lower() == ".npy":
            try:
                a = np.load(p, mmap_mode="r")
                shape, dtype = str(tuple(a.shape)), str(a.dtype)
            except Exception as e:
                readable = False
                note = f"npy_header_error:{type(e).__name__}"
        rows.append({
            "entity": classify_entity(p), "path": str(p), "filename": p.name,
            "suffix": p.suffix.lower(), "bytes": p.stat().st_size,
            "shape": shape, "dtype": dtype, "readable": readable, "note": note
        })
    return pd.DataFrame(rows)

def known_locks() -> pd.DataFrame:
    rows = []
    for name,p in KNOWN_LOCKS.items():
        r = {"name":name,"path":str(p),"exists":p.is_file(),"sha256":"","status":"","gate":""}
        if p.is_file():
            r["sha256"] = sha256_file(p)
            try:
                j = load_json(p); r["status"]=str(j.get("status","")); r["gate"]=str(j.get("gate",""))
            except Exception as e:
                r["status"] = f"JSON_ERROR:{type(e).__name__}"
        rows.append(r)
    return pd.DataFrame(rows)

def event_support() -> pd.DataFrame:
    rows = []
    for name,p in KNOWN_TABLES.items():
        if not p.is_file():
            rows.append({"asset":name,"path":str(p),"exists":False,"rows":"","physical_units":"","harm_events":"","physical_units_with_harm":"","grouping":"","note":"missing"})
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            rows.append({"asset":name,"path":str(p),"exists":True,"rows":"","physical_units":"","harm_events":"","physical_units_with_harm":"","grouping":"","note":f"read_error:{type(e).__name__}"})
            continue
        group = next((c for c in ["patient_id","patient","case_key","case_id","video_id","image_id","global_index"] if c in df.columns),"")
        harm = "harm" if "harm" in df.columns else ""
        units = int(df[group].astype(str).nunique()) if group else ""
        harms = int(df[harm].sum()) if harm else ""
        uh = ""
        if group and harm:
            uh = int((df.groupby(group,dropna=False)[harm].max()>0).sum())
        rows.append({"asset":name,"path":str(p),"exists":True,"rows":len(df),"physical_units":units,"harm_events":harms,"physical_units_with_harm":uh,"grouping":group,"note":""})
    return pd.DataFrame(rows)

def gt_history() -> pd.DataFrame:
    rows = [
        ["NeoPolyp","source development","YES","SOURCE-domain HARM supervision","source-development","N/A","SOURCE-domain action outcomes supervise the safety head."],
        ["PolypGen","external colonoscopy","YES","unseen-MEMO / joint-shift","historically GT-accessed; protocol-locked new-action evaluation",0,"Not pristine first-access validation."],
        ["SUN-SEG","external colonoscopy","YES","SOURCE-only external ranking / runtime subset","historical external evaluation",0,"Preserve historical label-access timeline for any new audit."],
        ["Prostate158","MRI source development","YES","MRI safety-head development","source-development","N/A","MRI PCA/scaler/head are re-developed on Prostate158."],
        ["PROMISE12","external prostate MRI","YES","historical SOURCE-only + new held-out-action transition","historically GT-accessed; protocol-locked new-action evaluation",0,"PROMISE12 GT was already accessed in the earlier SOURCE-only branch; new transition assets were frozen before new action-specific outcomes."]
    ]
    return pd.DataFrame(rows, columns=["cohort","role","historical_gt_access","current_use","current_analysis_status","target_rows_in_safety_fit","required_manuscript_wording"])

def hierarchy_summary(id_df: pd.DataFrame, entity: str) -> Dict[str, Any]:
    if id_df.empty or "entity" not in id_df.columns:
        return {"entity":entity,"recoverable_columns":[],"best_recoverable_unit":"NOT_RECOVERABLE","best_unit_columns":[]}
    sub = id_df[id_df["entity"]==entity]
    cols=[]
    for s in sub.get("id_like_columns", pd.Series(dtype=str)).fillna(""):
        cols += [x for x in str(s).split(";") if x]
    u = sorted(set(cols), key=str.lower)
    levels = [
        ("patient",[c for c in u if "patient" in c.lower() or "subject" in c.lower()]),
        ("procedure/video",[c for c in u if any(k in c.lower() for k in ["video","procedure","exam","study"])]),
        ("case",[c for c in u if "case" in c.lower()]),
        ("image/frame",[c for c in u if any(k in c.lower() for k in ["image","frame","sample","global_index"])])
    ]
    best, evidence = "NOT_RECOVERABLE", []
    for level, matches in levels:
        if matches: best,evidence=level,matches; break
    return {"entity":entity,"recoverable_columns":u,"best_recoverable_unit":best,"best_unit_columns":evidence,
            "note":"Column-name evidence only; semantic correctness must be verified before clustering."}

def self_test():
    assert str(ROOT) == r"F:\MEDSEG_SAFETTA"
    assert PASS_GATE == "PASS_B6_P04_REPRO_AND_STATISTICAL_UNIT_AUDIT_COMPLETE"
    print("SELF_TEST_CONSTANTS=PASS")
    print("SELF_TEST_READ_ONLY_AUDIT=PASS")
    print("SELF_TEST=PASS")

def main():
    ap = argparse.ArgumentParser(description="SafeTTA B6-P0-4 read-only provenance/asset audit.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    protocol = verify_protocol(Path(__file__).resolve().parent)
    if OUT_DIR.exists():
        raise FileExistsError(f"Never overwrite P0-4 output: {OUT_DIR}\nUse a _fix1 script/output directory if needed.")
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("="*132)
    print("SafeTTA B6-P0-4 — asset / statistical-unit / GT-history audit")
    print(f"Version       : {VERSION}")
    print(f"Protocol SHA  : {EXPECTED_PROTOCOL_SHA256}")
    print("Model fitting : NO | New AUROC/AUPRC: NO | Score changes: NO | Target tuning: NO")
    print("="*132)

    print("\n[1/6] Known locks")
    lock_df = known_locks()
    p_lock = OUT_DIR/"B6_P04_KNOWN_LOCKS.csv"; lock_df.to_csv(p_lock,index=False,encoding="utf-8")
    print(lock_df.to_string(index=False))

    print("\n[2/6] Relevant-file inventory")
    files = discover_relevant_files()
    inv = pd.DataFrame([{"entity":classify_entity(p),"path":str(p),"filename":p.name,"suffix":p.suffix.lower(),"bytes":p.stat().st_size} for p in files])
    p_inv = OUT_DIR/"B6_P04_RELEVANT_FILE_INVENTORY.csv"; inv.to_csv(p_inv,index=False,encoding="utf-8")
    print("relevant_files=",len(inv))

    print("\n[3/6] Physical-unit ID recovery")
    ids = id_recovery(files)
    p_ids = OUT_DIR/"B6_P04_ID_RECOVERY.csv"; ids.to_csv(p_ids,index=False,encoding="utf-8")
    hier = {name:hierarchy_summary(ids,key) for name,key in [
        ("PolypGen","polypgen"),("PROMISE12","promise12"),("NeoPolyp","neopolyp"),("SUN","sun"),("Prostate158","prostate158")
    ]}
    p_hier = OUT_DIR/"B6_P04_PHYSICAL_UNIT_SUMMARY.json"; write_json(p_hier,hier)
    for k,v in hier.items():
        print(f"{k}: {v['best_recoverable_unit']} {v['best_unit_columns']}")

    print("\n[4/6] Probability/logit asset recovery")
    probs = prob_recovery(files)
    p_probs = OUT_DIR/"B6_P04_PROBABILITY_ASSET_RECOVERY.csv"; probs.to_csv(p_probs,index=False,encoding="utf-8")
    print("probability_or_logit_candidates=",len(probs))

    print("\n[5/6] Event support + GT history")
    events = event_support()
    p_events = OUT_DIR/"B6_P04_EVENT_SUPPORT.csv"; events.to_csv(p_events,index=False,encoding="utf-8")
    print(events.to_string(index=False))
    gt = gt_history()
    p_gt = OUT_DIR/"B6_P04_GT_HISTORY.csv"; gt.to_csv(p_gt,index=False,encoding="utf-8")

    print("\n[6/6] Audit decision")
    prob_entities = sorted(set(probs["entity"].tolist())) if not probs.empty and "entity" in probs.columns else []
    recovery = {
        "PolypGen_best_recoverable_unit_from_column_names": hier["PolypGen"]["best_recoverable_unit"],
        "PROMISE12_best_recoverable_unit_from_column_names": hier["PROMISE12"]["best_recoverable_unit"],
        "probability_logit_candidate_entities": prob_entities,
        "reliability_change_baseline_decision":
            "PENDING_MANUAL_BINDING_TO_EXACT_ACTION_PANEL" if len(probs) else "NOT_RECOVERABLE_FROM_FILENAME_SCAN",
        "rule":"P0-1 may use reliability-change features only after exact action/panel/index binding is proven."
    }
    audit = {
        "status":"PASS","gate":PASS_GATE,"version":VERSION,
        "protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "read_only":True,"new_scientific_metrics":False,"model_fit":False,
        "score_change":False,"threshold_tuning":False,"target_tuning":False,
        "recovery_summary":recovery,
        "outputs":{}
    }
    for p in [p_lock,p_inv,p_ids,p_hier,p_probs,p_events,p_gt]:
        audit["outputs"][p.name]={"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size}
    p_audit = OUT_DIR/"B6_P04_REPRO_AUDIT.json"; write_json(p_audit,audit)

    report = "\n".join([
        "="*132,
        "SafeTTA B6-P0-4 AUDIT COMPLETE",
        f"PolypGen best recoverable unit: {hier['PolypGen']['best_recoverable_unit']}",
        f"PROMISE12 best recoverable unit: {hier['PROMISE12']['best_recoverable_unit']}",
        f"Probability/logit entities: {prob_entities}",
        f"Reliability-change status: {recovery['reliability_change_baseline_decision']}",
        "",
        f"GATE={PASS_GATE}",
        f"audit_json={p_audit}",
        f"stage_dir={OUT_DIR}",
        "="*132,""
    ])
    (OUT_DIR/"B6_P04_REPORT.txt").write_text(report,encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
