#!/usr/bin/env python
"""Post-freeze HARM-margin sensitivity audit for SafeTTA.

This script does NOT train, recalibrate, or change any frozen safety score. It scans
existing output CSVs for row-level tables containing delta_dice plus a candidate
pre-adaptation risk score, then recomputes HARM labels at margins 0.01, 0.02, 0.05.
It is intentionally read-only with respect to the experiment tree.

Default root: F:\\MEDSEG_SAFETTA\\outputs
Output: a new audit directory under F:\\MEDSEG_SAFETTA\\outputs unless --out is given.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

SCORE_PRIORITY = [
    "frozen_safety_probability", "final_safety_probability", "safety_probability",
    "harm_probability", "risk_probability", "risk_score", "safety_score",
]
MARGINS = (0.01, 0.02, 0.05)
FAMILY_COLS = ("model_family", "architecture_family", "family", "architecture")
CLUSTER_COLS = ("patient_id", "sample_id", "case_id", "video_id", "physical_case_id")

def pick(cols, options):
    low={c.lower():c for c in cols}
    for o in options:
        if o.lower() in low: return low[o.lower()]
    return None

def auc_safe(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    ok=np.isfinite(s)
    y=y[ok]; s=s[ok]
    if len(y)==0 or len(np.unique(y))<2: return math.nan, math.nan
    return float(roc_auc_score(y,s)), float(average_precision_score(y,s))

def dataset_hint(path: Path):
    t=str(path).lower()
    for key,label in [("promise","PROMISE12"),("prostate","Prostate158"),("polypgen","PolypGen"),("sunseg","SUN-SEG"),("sun_","SUN-SEG"),("neopolyp","NeoPolyp")]:
        if key in t: return label
    return "UNRESOLVED"

def inspect_csv(path: Path):
    try:
        cols=list(pd.read_csv(path,nrows=0).columns)
    except Exception:
        return None
    delta=pick(cols,["delta_dice","deltaDice","dice_delta"])
    score=pick(cols,SCORE_PRIORITY)
    fam=pick(cols,FAMILY_COLS)
    cluster=pick(cols,CLUSTER_COLS)
    if delta and score:
        return dict(path=str(path),delta=delta,score=score,family=fam,cluster=cluster,dataset=dataset_hint(path))
    return None

def summarize_candidate(meta):
    use=[meta['delta'],meta['score']]
    if meta['family']: use.append(meta['family'])
    if meta['cluster'] and meta['cluster'] not in use: use.append(meta['cluster'])
    df=pd.read_csv(meta['path'],usecols=use,low_memory=False)
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=[meta['delta'],meta['score']])
    rows=[]
    fam_col=meta['family']
    for margin in MARGINS:
        y=(df[meta['delta']].to_numpy(float) <= -margin).astype(int)
        auc,ap=auc_safe(y,df[meta['score']].to_numpy(float))
        fam_aucs=[]; fam_aps=[]
        if fam_col:
            for _,g in df.groupby(fam_col,sort=True):
                yy=(g[meta['delta']].to_numpy(float) <= -margin).astype(int)
                a,p=auc_safe(yy,g[meta['score']].to_numpy(float))
                if np.isfinite(a): fam_aucs.append(a)
                if np.isfinite(p): fam_aps.append(p)
        rows.append({
            "dataset_hint":meta['dataset'], "csv":meta['path'], "score_column":meta['score'],
            "delta_column":meta['delta'], "family_column":fam_col or "", "cluster_column":meta['cluster'] or "",
            "rows":len(df), "harm_margin":margin, "harm_prevalence":float(y.mean()),
            "pooled_auroc":auc, "pooled_auprc":ap,
            "macro_family_auroc":float(np.mean(fam_aucs)) if fam_aucs else math.nan,
            "macro_family_auprc":float(np.mean(fam_aps)) if fam_aps else math.nan,
        })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(r'F:\MEDSEG_SAFETTA\outputs'))
    ap.add_argument('--out',type=Path,default=Path(r'F:\MEDSEG_SAFETTA\outputs\Q1_SAFETTA_harm_margin_sensitivity_v1'))
    args=ap.parse_args()
    if not args.root.exists(): raise SystemExit(f'Root not found: {args.root}')
    paths=list(args.root.rglob('*.csv'))
    metas=[]
    for p in tqdm(paths,desc='Discover CSVs',unit='csv',dynamic_ncols=True):
        m=inspect_csv(p)
        if m: metas.append(m)
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metas).to_csv(args.out/'candidate_row_level_tables.csv',index=False)
    all_rows=[]
    for m in tqdm(metas,desc='Sensitivity',unit='table',dynamic_ncols=True):
        try: all_rows.extend(summarize_candidate(m))
        except Exception as e:
            all_rows.append({"dataset_hint":m['dataset'],"csv":m['path'],"error":repr(e)})
    out=pd.DataFrame(all_rows)
    out.to_csv(args.out/'harm_margin_sensitivity_all_candidates.csv',index=False)
    # Compact canonical-looking view: prefer explicit frozen/final score names and resolved dataset hints.
    if not out.empty and 'error' not in out.columns:
        pass
    if not out.empty:
        good=out.copy()
        if 'error' in good.columns: good=good[good['error'].isna()]
        if len(good):
            good['priority']=good['score_column'].map({n:i for i,n in enumerate(SCORE_PRIORITY)}).fillna(999)
            good=good.sort_values(['dataset_hint','priority','csv','harm_margin'])
            good.to_csv(args.out/'harm_margin_sensitivity_ranked.csv',index=False)
    print('\n===== SAFETTA HARM-MARGIN SENSITIVITY =====')
    print('CSV files scanned:',len(paths))
    print('Candidate row-level tables:',len(metas))
    print('Margins:',MARGINS)
    print('Output:',args.out)
    print('Training/recalibration: NONE')
    print('Frozen risk scores changed: NO')

if __name__=='__main__': main()
