#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, hashlib, json, re, tarfile, zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from tqdm import tqdm

ROOT = Path(r"F:\MEDSEG_SAFETTA")
DATA = ROOT / "data" / "external" / "SUN_SEG"
ANN = DATA / "SUN-SEG-Annotation-v2.zip"
RAW = DATA / "SUN-SEG-FinalData-v20251212.tar.gz"
OUT = ROOT / "outputs" / "Q1_R14B2_sunseg_deep_schema_pairing_identity_audit_fix1_v1"

ANN_SHA = "4ce0324c38743aeb47188dc18eded33258b2da8c07b7392a457dc55e77a1c7e7"
RAW_SHA = "d9a00fada04782937a144e9d1bc3c2a3b7f8e321e37ba1cf83bebdd490563016"
DECISION = "SUNSEG_DEEP_SCHEMA_AND_FRAME_GT_PAIRING_AUDITED_PRE_EXTRACTION"
CATS = {"frame","gt","edge","scribble","polygon","classification","detection"}
TEXT_EXTS = {".txt",".json",".md",".csv"}

def sha256_file(p):
    h = hashlib.sha256(); total = p.stat().st_size
    with p.open("rb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                 desc=f"SHA256 {p.name}", dynamic_ncols=True) as bar:
        while True:
            b = f.read(16*1024*1024)
            if not b: break
            h.update(b); bar.update(len(b))
    return h.hexdigest()

def norm(s):
    s = s.replace("\\","/")
    while s.startswith("./"): s = s[2:]
    return s

def parts(s): return [x for x in norm(s).split("/") if x]

def classify(name):
    ps = parts(name); lo = [x.lower() for x in ps]
    ds = next((ps[i] for i,x in enumerate(lo) if x in
               {"traindataset","testeasydataset","testharddataset"}), "<UNKNOWN>")
    vis = next((ps[i] for i,x in enumerate(lo) if x in {"seen","unseen"}), "<NA>")
    cat = next((ps[i] for i,x in enumerate(lo) if x in CATS), "<UNKNOWN>")
    return ds, vis, cat

def canonical_key(name, cat):
    ps = parts(name); lo = [x.lower() for x in ps]
    try: i = lo.index(cat.lower())
    except ValueError: return None
    after = ps[i+1:]
    if not after: return None
    after[-1] = str(PurePosixPath(after[-1]).with_suffix(""))
    return "/".join(ps[:i] + after).lower()

def rel_after(name, cat):
    ps = parts(name); lo = [x.lower() for x in ps]
    try: i = lo.index(cat.lower())
    except ValueError: return ""
    return "/".join(ps[i+1:])

def tokens(name):
    return [x for x in re.split(r"[_\-\s]+", PurePosixPath(name).stem) if x][:4]

def write_csv(p, rows, fields):
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def inventory_tar(p):
    rows, text_names = [], []
    with tarfile.open(p, "r:gz") as tf:
        members = tf.getmembers()
        for i,m in enumerate(tqdm(members, desc="Deep inventory TAR",
                                  unit="member", dynamic_ncols=True)):
            if not m.isfile(): continue
            n = norm(m.name); ds,vis,cat = classify(n); ext = PurePosixPath(n).suffix.lower()
            rows.append({"member_index":i,"member_name":n,"extension":ext,
                          "dataset":ds,"visibility":vis,"category":cat,"file_size":int(m.size)})
            if ext in TEXT_EXTS and m.size <= 2_000_000: text_names.append(n)
    return rows, text_names

def inventory_zip(p):
    rows, text_names = [], []
    with zipfile.ZipFile(p,"r") as zf:
        infos = zf.infolist()
        for i,m in enumerate(tqdm(infos, desc="Deep inventory ZIP",
                                  unit="member", dynamic_ncols=True)):
            if m.is_dir(): continue
            n = norm(m.filename); ds,vis,cat = classify(n); ext = PurePosixPath(n).suffix.lower()
            rows.append({"member_index":i,"member_name":n,"extension":ext,
                          "dataset":ds,"visibility":vis,"category":cat,"file_size":int(m.file_size)})
            if ext in TEXT_EXTS and m.file_size <= 2_000_000: text_names.append(n)
    return rows, text_names

def pair_audit(rows):
    frames, gts = {}, {}
    fdup, gdup = Counter(), Counter()
    for r in rows:
        c = r["category"].lower()
        if c == "frame":
            k = canonical_key(r["member_name"], "frame")
            if k: fdup[k]+=1; frames.setdefault(k,r)
        elif c == "gt":
            k = canonical_key(r["member_name"], "gt")
            if k: gdup[k]+=1; gts.setdefault(k,r)
    fk,gk = set(frames), set(gts)
    common = sorted(fk & gk)
    by = Counter(); pairs = []
    for k in common:
        f,g = frames[k], gts[k]; by[(f["dataset"],f["visibility"])] += 1
        pairs.append({"pair_key":k,"dataset":f["dataset"],"visibility":f["visibility"],
                       "frame_member":f["member_name"],"gt_member":g["member_name"],
                       "frame_relative":rel_after(f["member_name"],"frame"),
                       "gt_relative":rel_after(g["member_name"],"gt"),
                       "filename_tokens":"|".join(tokens(f["member_name"]))})
    return {"frame_unique":len(fk),"gt_unique":len(gk),"paired":len(common),
             "frame_only":len(fk-gk),"gt_only":len(gk-fk),
             "duplicate_frame_keys":sum(v>1 for v in fdup.values()),
             "duplicate_gt_keys":sum(v>1 for v in gdup.values()),
             "by_split":by,"pairs":pairs}

def deep_counts(rows, depth):
    c = Counter()
    for r in rows:
        ps = parts(r["member_name"])
        c[tuple(ps[i] if i < len(ps) else "<NONE>" for i in range(depth))] += 1
    return c

def dump_counter(p, c, depth):
    fields=[f"level{i}" for i in range(1,depth+1)]+["count"]; rows=[]
    for key,n in c.most_common():
        row={f"level{i+1}":key[i] for i in range(depth)}; row["count"]=n; rows.append(row)
    write_csv(p,rows,fields)

def decode(b):
    for e in ("utf-8","utf-8-sig","latin-1"):
        try: return b.decode(e)
        except UnicodeDecodeError: pass
    return b.decode("utf-8",errors="replace")

def read_tar_texts(p,names):
    out=[]
    with tarfile.open(p,"r:gz") as tf:
        mp={norm(m.name):m for m in tf.getmembers() if m.isfile()}
        for n in names:
            f=tf.extractfile(mp[n]); out.append((n,decode(f.read() if f else b"")))
    return out

def read_zip_texts(p,names):
    out=[]
    with zipfile.ZipFile(p,"r") as zf:
        for n in names: out.append((n,decode(zf.read(n))))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotation-zip",type=Path,default=ANN)
    ap.add_argument("--raw-tar",type=Path,default=RAW)
    ap.add_argument("--output-dir",type=Path,default=OUT)
    a=ap.parse_args()

    print("===== Q1 R14B2 SUN-SEG DEEP SCHEMA / PAIRING / IDENTITY AUDIT =====")
    print("EXTRACTION=NO"); print("GT_PIXEL_DECODE=NO"); print("MODEL_INFERENCE=NO")
    print("SUBSET_SELECTION=NO"); print("CONTAMINATION_CLAIM=NO")

    if sha256_file(a.annotation_zip) != ANN_SHA: raise RuntimeError("Annotation SHA mismatch")
    if sha256_file(a.raw_tar) != RAW_SHA: raise RuntimeError("Raw SHA mismatch")
    print("R14B1 SHA GATE=PASS")

    raw_rows, raw_text_names = inventory_tar(a.raw_tar)
    ann_rows, ann_text_names = inventory_zip(a.annotation_zip)
    raw_pair = pair_audit(raw_rows); ann_pair = pair_audit(ann_rows)

    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=False)

    fields=["member_index","member_name","extension","dataset","visibility","category","file_size"]
    write_csv(a.output_dir/"R14B2_raw_file_members.csv",raw_rows,fields)
    write_csv(a.output_dir/"R14B2_annotation_file_members.csv",ann_rows,fields)
    for d in (4,5,6):
        dump_counter(a.output_dir/f"R14B2_raw_level{d}_summary.csv",deep_counts(raw_rows,d),d)
        dump_counter(a.output_dir/f"R14B2_annotation_level{d}_summary.csv",deep_counts(ann_rows,d),d)

    pfields=["pair_key","dataset","visibility","frame_member","gt_member",
             "frame_relative","gt_relative","filename_tokens"]
    write_csv(a.output_dir/"R14B2_raw_frame_gt_pairs.csv",raw_pair["pairs"],pfields)

    parent_counter=Counter(); token_counter=Counter()
    for r in raw_pair["pairs"]:
        parent=str(PurePosixPath(r["frame_relative"]).parent)
        parent_counter[(r["dataset"],r["visibility"],parent)] += 1
        ts=[x for x in r["filename_tokens"].split("|") if x]
        if ts: token_counter[(r["dataset"],r["visibility"],ts[0])] += 1

    write_csv(a.output_dir/"R14B2_candidate_parent_identity_summary.csv",
              [{"dataset":k[0],"visibility":k[1],"parent_after_frame":k[2],"count":n}
               for k,n in parent_counter.most_common()],
              ["dataset","visibility","parent_after_frame","count"])
    write_csv(a.output_dir/"R14B2_candidate_filename_token1_summary.csv",
              [{"dataset":k[0],"visibility":k[1],"token1":k[2],"count":n}
               for k,n in token_counter.most_common()],
              ["dataset","visibility","token1","count"])

    chunks=["===== RAW ARCHIVE TEXT METADATA ====="]
    for n,t in read_tar_texts(a.raw_tar,raw_text_names): chunks.append(f"\n--- {n} ---\n{t[:200000]}")
    chunks.append("\n===== ANNOTATION ARCHIVE TEXT METADATA =====")
    for n,t in read_zip_texts(a.annotation_zip,ann_text_names): chunks.append(f"\n--- {n} ---\n{t[:200000]}")
    metadata="\n".join(chunks)
    (a.output_dir/"R14B2_text_metadata_dump.txt").write_text(metadata,encoding="utf-8")

    print("\n===== RAW FRAME <-> GT PAIRING =====")
    for k in ("frame_unique","gt_unique","paired","frame_only","gt_only",
              "duplicate_frame_keys","duplicate_gt_keys"): print(f"{k}: {raw_pair[k]}")

    print("\n===== RAW PAIRED COUNTS BY SPLIT =====")
    for (ds,vis),n in raw_pair["by_split"].most_common(): print(f"{ds}/{vis}: {n}")

    print("\n===== PAIRED MEMBER EXAMPLES =====")
    for r in raw_pair["pairs"][:30]:
        print(f"{r['dataset']}/{r['visibility']} | FRAME={r['frame_member']} | GT={r['gt_member']}")

    print("\n===== CANDIDATE PARENT IDENTITY TOP 80 =====")
    for k,n in parent_counter.most_common(80): print("/".join(k),":",n)

    print("\n===== CANDIDATE FILENAME TOKEN1 TOP 80 =====")
    for k,n in token_counter.most_common(80): print("/".join(k),":",n)

    print("\n===== RAW README / TEXT METADATA PREVIEW =====")
    print(metadata[:30000])

    print("\n===== ANNOTATION ARCHIVE PAIRING (DESCRIPTIVE) =====")
    for k in ("frame_unique","gt_unique","paired","frame_only","gt_only"): print(f"{k}: {ann_pair[k]}")

    lock={
        "status":"PASS","decision":DECISION,
        "input_sha256":{"annotation_zip":ANN_SHA,"raw_tar_gz":RAW_SHA},
        "raw_pairing":{k:raw_pair[k] for k in
                       ("frame_unique","gt_unique","paired","frame_only","gt_only",
                        "duplicate_frame_keys","duplicate_gt_keys")},
        "paired_counts_by_split":[{"dataset":k[0],"visibility":k[1],"paired_count":n}
                                  for k,n in raw_pair["by_split"].most_common()],
        "information_boundary":{"archive_extracted":False,"gt_pixels_decoded":False,
                                "model_inference":False,"tta_execution":False,
                                "subset_selection":False,"contamination_claim":False,
                                "identity_mapping_frozen":False},
        "next_stage":"R14B3_FREEZE_PHYSICAL_CASE_VIDEO_IDENTITY_AND_DETERMINISTIC_COHORT_SAMPLING_PLAN"
    }
    lp=a.output_dir/"R14B2_SUNSEG_DEEP_SCHEMA_PAIRING_IDENTITY_LOCK.json"
    lp.write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nDecision=",DECISION); print("LOCK=",lp); print("PASS")

if __name__ == "__main__":
    main()
