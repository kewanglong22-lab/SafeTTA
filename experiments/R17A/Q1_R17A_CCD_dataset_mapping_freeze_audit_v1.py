#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1_R17A_CCD_dataset_mapping_freeze_audit_v1.py

SafeTTA R17A — dataset-specific SOURCE-probability mapping freeze audit.

READ-ONLY:
- no training
- no inference
- no GT loading
- no file deletion/move

Purpose
-------
The previous R17A audit showed:
  PolypGen      -> direct soft-asset candidates exist
  Prostate158   -> direct soft-asset candidates exist
  NeoPolyp      -> exact integration unresolved
  SUN-SEG       -> exact integration unresolved
  PROMISE12     -> exact integration unresolved

This script now resolves *which exact frozen files* can be used to compute
official-code-faithful SicTTA CCD and which datasets require GT-free SOURCE
re-inference.

It produces compact mapping evidence, not CCD scores.

Key checks
----------
1. Inspect .npz/.npy headers without fully materializing large arrays.
2. Classify candidate arrays as logits/probabilities/hard masks/unknown.
3. Search nearby manifests / CSVs for sample identifiers and row counts.
4. Search current code for exact references to candidate asset paths/basenames.
5. Search current code for SOURCE checkpoint + inference paths.
6. Build one dataset-level mapping decision:
      DIRECT_SOFT_ASSET_READY
      GT_FREE_SOURCE_REINFERENCE_READY
      BLOCKED_AMBIGUOUS_MAPPING

No target GT is opened.

Output
------
F:\\MEDSEG_SAFETTA\\outputs\\Q1_R17A_CCD_dataset_mapping_freeze_audit_v1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "2026-09-08-Q1-R17A-CCD-DATASET-MAPPING-FREEZE-AUDIT-v1"
DEFAULT_ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT_REL = Path(r"outputs\Q1_R17A_CCD_dataset_mapping_freeze_audit_v1")

SEARCH_ROOTS = [
    Path("outputs"),
    Path("code"),
    Path("release/SafeTTA_clean_test_v102"),
    Path("release/public_v1_fix4"),
    Path("release/SafeTTA_repro_release_candidate_v1"),
    Path("data"),
    Path("cross_modality_prostate_mri"),
]

CODE_ROOTS = [
    Path("code"),
    Path("release/SafeTTA_clean_test_v102/experiments"),
    Path("release/SafeTTA_clean_test_v102/method_core"),
]

DATASET_ALIASES = {
    "NeoPolyp": [
        "neopolyp", "neo_polyp",
        "s01_b_pranet", "s05_b_deeplab",
        "q1_r02_segformer",
    ],
    "PolypGen": [
        "polypgen", "s06_c_polypgen",
        "q1_r10l3a", "q1_r10l3b",
    ],
    "SUNSEG": [
        "sunseg", "sun_seg", "sun-seg",
        "q1_r14b", "q1_r14c",
    ],
    "Prostate158": [
        "prostate158", "prostate_158",
        "q1x_cm3", "q1x_cm4",
    ],
    "PROMISE12": [
        "promise12",
        "q1x_cm5", "q1x_cm6", "q1x_cm7",
    ],
}

SOFT_NAME_TOKENS = (
    "logit", "logits", "prob", "probs", "probability", "softmax"
)
HARD_NAME_TOKENS = (
    "mask", "masks", "packbits", "binary", "label", "labels"
)

ARRAY_SUFFIXES = {".npy", ".npz"}
CKPT_SUFFIXES = {".pt", ".pth", ".ckpt"}
MANIFEST_SUFFIXES = {".csv", ".json", ".txt", ".jsonl"}
TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".csv", ".yaml", ".yml"}

GT_NAME_TOKENS = (
    "gt", "ground_truth", "groundtruth", "label", "labels",
    "mask_gt", "gt_mask", "dice", "outcome", "harm"
)

# Frozen candidate route priorities. These do NOT authorize use by themselves;
# they only rank likely current assets for the audit report.
PREFERRED_ROUTE_TOKENS = {
    "NeoPolyp": (
        "s01_b_pranet_source_only",
        "s05_b_deeplabv3_r50_source_only",
        "q1_r02_segformer_b0_three_seed_source_training",
        "q1_r03_nine_state",
        "q1_r05d3",
    ),
    "PolypGen": (
        "s06_c_polypgen_dual_backbone_no_label_lock",
        "q1_r10l3a_polypgen_source_a1_prediction_lock",
        "q1_r10l3b_polypgen_frozen_safety_score_lock",
    ),
    "SUNSEG": (
        "q1_r14c1_sunseg_source_tent1_pl_prediction_lock_pregt",
        "q1_r14c2b_sunseg_frozen_safety_score_lock_pregt",
    ),
    "Prostate158": (
        "q1x_cm3_source_oof_segmentation_training_and_prediction_lock_fix3",
        "q1x_cm4a_source_oof_tent1_outcome_asset_lock_fix4",
        "q1x_cm4d_final_all_source_segmentation_models_lock_fix2",
    ),
    "PROMISE12": (
        "q1x_cm5a_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1",
        "q1x_cm6_promise12_gt_reveal_locked_policy_evaluation_fix1",
    ),
}


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def norm(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def tags_for(rel: str) -> List[str]:
    low = rel.lower()
    out = []
    for ds, aliases in DATASET_ALIASES.items():
        if any(a in low for a in aliases):
            out.append(ds)
    return sorted(set(out))


def route_priority(ds: str, rel: str) -> int:
    low = rel.lower()
    toks = PREFERRED_ROUTE_TOKENS.get(ds, ())
    for i, tok in enumerate(toks):
        if tok in low:
            return 100 - i
    return 0


def likely_gt_path(rel: str) -> bool:
    low = rel.lower()
    parts = [x for x in re.split(r"[/\\._-]+", low) if x]
    # Exact-ish token logic; avoid flagging "target".
    return any(tok in parts for tok in GT_NAME_TOKENS)


def read_npy_header_from_file(path: Path) -> Dict[str, object]:
    try:
        with path.open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        return {
            "shape": list(shape),
            "dtype": str(dtype),
            "fortran_order": bool(fortran),
            "status": "OK",
        }
    except Exception as e:
        return {
            "shape": "",
            "dtype": "",
            "fortran_order": "",
            "status": f"ERROR:{type(e).__name__}:{e}",
        }


def read_npy_header_from_zip_member(zf: zipfile.ZipFile, member: str) -> Dict[str, object]:
    try:
        with zf.open(member, "r") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        return {
            "shape": list(shape),
            "dtype": str(dtype),
            "fortran_order": bool(fortran),
            "status": "OK",
        }
    except Exception as e:
        return {
            "shape": "",
            "dtype": "",
            "fortran_order": "",
            "status": f"ERROR:{type(e).__name__}:{e}",
        }


def array_metadata(path: Path) -> List[Dict[str, object]]:
    """
    Return one row per NPY file or one row per NPY member of NPZ.
    Reads array headers only; does not materialize full arrays.
    """
    if path.suffix.lower() == ".npy":
        meta = read_npy_header_from_file(path)
        return [{
            "member": "",
            **meta,
        }]

    rows = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".npy"):
                    continue
                meta = read_npy_header_from_zip_member(zf, member)
                rows.append({
                    "member": member,
                    **meta,
                })
    except Exception as e:
        rows.append({
            "member": "",
            "shape": "",
            "dtype": "",
            "fortran_order": "",
            "status": f"ERROR:{type(e).__name__}:{e}",
        })
    return rows


def classify_member(path_rel: str, member: str, shape) -> str:
    text = f"{path_rel} {member}".lower()

    if any(tok in text for tok in HARD_NAME_TOKENS):
        return "HARD_OR_MASK_LIKE"

    if any(tok in text for tok in SOFT_NAME_TOKENS):
        # Shape evidence is informative but not mandatory.
        return "SOFT_LOGIT_OR_PROB_CANDIDATE"

    if isinstance(shape, list) and len(shape) >= 3:
        # Could still be image/mask/cache; keep conservative.
        return "UNKNOWN_HIGH_DIM_ARRAY"

    return "UNKNOWN_ARRAY"


def scan_arrays(root: Path) -> List[dict]:
    files = []
    seen = set()
    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".pytest_cache"}
                and not (dp / d).is_symlink()
            ]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in ARRAY_SUFFIXES:
                    continue
                rel = norm(p, root)
                if rel in seen:
                    continue
                seen.add(rel)

                ds_tags = tags_for(rel)
                if not ds_tags:
                    continue

                # Never inspect obvious post-GT arrays as CCD source candidates.
                if likely_gt_path(rel):
                    continue

                try:
                    size = int(p.stat().st_size)
                except OSError:
                    size = 0

                files.append((p, rel, ds_tags, size))

    files.sort(key=lambda x: -x[3])

    rows = []
    iterator = files
    if tqdm is not None:
        iterator = tqdm(
            files,
            desc="Inspect candidate array headers",
            unit="file",
            dynamic_ncols=True,
        )

    for p, rel, ds_tags, size in iterator:
        metas = array_metadata(p)
        for meta in metas:
            cls = classify_member(rel, str(meta["member"]), meta["shape"])
            for ds in ds_tags:
                rows.append({
                    "dataset": ds,
                    "path": rel,
                    "member": meta["member"],
                    "size_bytes": size,
                    "size_human": fmt_bytes(size),
                    "shape": json.dumps(meta["shape"]) if isinstance(meta["shape"], list) else meta["shape"],
                    "dtype": meta["dtype"],
                    "header_status": meta["status"],
                    "signal_class": cls,
                    "route_priority": route_priority(ds, rel),
                })

    return rows


def scan_manifests(root: Path) -> List[dict]:
    rows = []
    seen = set()

    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".pytest_cache"}
                and not (dp / d).is_symlink()
            ]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in MANIFEST_SUFFIXES:
                    continue
                rel = norm(p, root)
                if rel in seen:
                    continue
                seen.add(rel)

                ds_tags = tags_for(rel)
                if not ds_tags:
                    continue

                # Exclude obvious post-GT outcome tables from alignment candidates.
                if likely_gt_path(rel):
                    continue

                try:
                    size = int(p.stat().st_size)
                except OSError:
                    continue
                if size > 30 * 1024 * 1024:
                    continue

                row_count = ""
                columns = ""
                status = "OK"

                if p.suffix.lower() == ".csv":
                    try:
                        with p.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
                            reader = csv.reader(f)
                            header = next(reader, [])
                            columns = "|".join(header[:80])
                            n = 0
                            for _ in reader:
                                n += 1
                            row_count = n
                    except Exception as e:
                        status = f"ERROR:{type(e).__name__}:{e}"

                elif p.suffix.lower() in {".json", ".jsonl"}:
                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        if p.suffix.lower() == ".jsonl":
                            row_count = sum(1 for x in txt.splitlines() if x.strip())
                        else:
                            obj = json.loads(txt)
                            if isinstance(obj, list):
                                row_count = len(obj)
                            elif isinstance(obj, dict):
                                columns = "|".join(list(obj.keys())[:80])
                    except Exception as e:
                        status = f"ERROR:{type(e).__name__}:{e}"

                for ds in ds_tags:
                    rows.append({
                        "dataset": ds,
                        "path": rel,
                        "size_bytes": size,
                        "size_human": fmt_bytes(size),
                        "row_count": row_count,
                        "columns_or_keys": columns,
                        "status": status,
                        "route_priority": route_priority(ds, rel),
                    })

    return rows


def scan_checkpoints(root: Path) -> List[dict]:
    rows = []
    seen = set()

    for rel_root in SEARCH_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".pytest_cache"}
                and not (dp / d).is_symlink()
            ]
            for name in filenames:
                p = dp / name
                if p.suffix.lower() not in CKPT_SUFFIXES:
                    continue
                rel = norm(p, root)
                if rel in seen:
                    continue
                seen.add(rel)

                low = rel.lower()
                ds_tags = tags_for(rel)

                # NeoPolyp source checkpoints often do not include "NeoPolyp"
                # in every basename; explicitly tag known SOURCE routes.
                if any(tok in low for tok in [
                    "s01_b_pranet_source_only",
                    "s05_b_deeplabv3_r50_source_only",
                    "q1_r02_segformer_b0_three_seed_source_training",
                ]):
                    if "NeoPolyp" not in ds_tags:
                        ds_tags.append("NeoPolyp")

                if not ds_tags:
                    continue

                try:
                    size = int(p.stat().st_size)
                except OSError:
                    size = 0

                for ds in sorted(set(ds_tags)):
                    rows.append({
                        "dataset": ds,
                        "path": rel,
                        "size_bytes": size,
                        "size_human": fmt_bytes(size),
                        "route_priority": route_priority(ds, rel),
                    })

    rows.sort(key=lambda r: (r["dataset"], -r["route_priority"], -r["size_bytes"]))
    return rows


def code_reference_audit(root: Path, candidate_paths: List[str]) -> List[dict]:
    basenames = defaultdict(list)
    for rel in candidate_paths:
        basenames[Path(rel).name].append(rel)

    rows = []
    code_files = []
    for rel_root in CODE_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            code_files.append((p, txt))

    iterator = candidate_paths
    if tqdm is not None:
        iterator = tqdm(
            candidate_paths,
            desc="Map candidate assets to code",
            unit="asset",
            dynamic_ncols=True,
        )

    for rel in iterator:
        rel_win = rel.replace("/", "\\")
        basename = Path(rel).name
        hits = []
        for p, txt in code_files:
            if rel in txt or rel_win in txt or basename in txt:
                hits.append(norm(p, root))
        rows.append({
            "candidate_path": rel,
            "code_reference_count": len(set(hits)),
            "code_reference_examples": " | ".join(sorted(set(hits))[:12]),
        })

    return rows


def inference_code_audit(root: Path) -> List[dict]:
    rows = []
    SIGNALS = (
        "torch.sigmoid", "sigmoid(", "torch.softmax", "softmax(",
        "load_state_dict", "checkpoint", "source", "model(",
    )

    for rel_root in CODE_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue

        for p in base.rglob("*.py"):
            rel = norm(p, root)
            ds_tags = tags_for(rel)

            lowrel = rel.lower()
            if not ds_tags:
                # Tag known generic source-training/inference families.
                if any(tok in lowrel for tok in [
                    "pranet", "deeplab", "segformer",
                    "source_only", "prediction_lock",
                ]):
                    ds_tags = ["NeoPolyp"]

            if not ds_tags:
                continue

            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            snippets = []
            for i, line in enumerate(lines):
                low = line.lower()
                if any(sig.lower() in low for sig in SIGNALS):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    snippets.append(" || ".join(
                        f"L{j+1}:{lines[j].strip()[:160]}" for j in range(start, end)
                    ))

            if snippets:
                for ds in sorted(set(ds_tags)):
                    rows.append({
                        "dataset": ds,
                        "path": rel,
                        "signal_hit_count": len(snippets),
                        "examples": " ### ".join(snippets[:8]),
                    })

    return rows


def build_readiness(
    arrays: List[dict],
    manifests: List[dict],
    ckpts: List[dict],
    inference_code: List[dict],
    code_refs: Dict[str, int],
) -> Dict[str, dict]:
    out = {}

    for ds in DATASET_ALIASES:
        ds_arrays = [r for r in arrays if r["dataset"] == ds]
        ds_soft = [
            r for r in ds_arrays
            if r["signal_class"] == "SOFT_LOGIT_OR_PROB_CANDIDATE"
            and r["header_status"] == "OK"
        ]
        ds_man = [r for r in manifests if r["dataset"] == ds and r["status"] == "OK"]
        ds_ck = [r for r in ckpts if r["dataset"] == ds]
        ds_code = [r for r in inference_code if r["dataset"] == ds]

        # Strong direct asset = soft candidate with at least one code reference.
        strong_soft = [
            r for r in ds_soft
            if code_refs.get(r["path"], 0) > 0
        ]

        if strong_soft:
            route = "DIRECT_SOFT_ASSET_READY"
            reason = "soft/logit candidate has array-header evidence and is referenced by retained code"
        elif ds_ck and ds_code:
            route = "GT_FREE_SOURCE_REINFERENCE_READY"
            reason = "no strong direct soft asset; frozen checkpoint and SOURCE inference code are available"
        else:
            route = "BLOCKED_AMBIGUOUS_MAPPING"
            reason = "cannot yet establish direct soft asset or checkpoint+inference path"

        out[ds] = {
            "decision": route,
            "reason": reason,
            "soft_candidate_count": len(ds_soft),
            "strong_soft_candidate_count": len(strong_soft),
            "manifest_candidate_count": len(ds_man),
            "checkpoint_count": len(ds_ck),
            "inference_code_count": len(ds_code),
            "top_soft_candidates": [
                {
                    "path": r["path"],
                    "member": r["member"],
                    "shape": r["shape"],
                    "dtype": r["dtype"],
                    "code_reference_count": code_refs.get(r["path"], 0),
                    "route_priority": r["route_priority"],
                }
                for r in sorted(
                    strong_soft or ds_soft,
                    key=lambda x: (-x["route_priority"], -x["size_bytes"])
                )[:12]
            ],
            "top_checkpoints": [
                r["path"] for r in ds_ck[:12]
            ],
            "top_manifests": [
                {
                    "path": r["path"],
                    "row_count": r["row_count"],
                    "columns_or_keys": r["columns_or_keys"],
                }
                for r in sorted(
                    ds_man,
                    key=lambda x: (-x["route_priority"], x["path"])
                )[:12]
            ],
        }

    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== R17A CCD DATASET MAPPING FREEZE AUDIT =====")
    print("Version:", VERSION)
    print("Root:", root)
    print("Read-only: YES")
    print("Training: NO")
    print("Inference: NO")
    print("GT loading: NO")
    print("Deletion/move: NO")
    print()

    print("[1/6] Inspect candidate soft/logit arrays...")
    arrays = scan_arrays(root)

    print("[2/6] Inspect non-GT alignment manifests...")
    manifests = scan_manifests(root)

    print("[3/6] Inventory frozen SOURCE checkpoints...")
    ckpts = scan_checkpoints(root)

    print("[4/6] Audit SOURCE inference code...")
    inf_code = inference_code_audit(root)

    print("[5/6] Map candidate assets to retained code...")
    candidate_paths = sorted(set(
        r["path"] for r in arrays
        if r["signal_class"] == "SOFT_LOGIT_OR_PROB_CANDIDATE"
    ))
    ref_rows = code_reference_audit(root, candidate_paths)
    ref_map = {
        r["candidate_path"]: int(r["code_reference_count"])
        for r in ref_rows
    }

    print("[6/6] Freeze dataset-level mapping decisions...")
    readiness = build_readiness(
        arrays, manifests, ckpts, inf_code, ref_map
    )

    write_csv(out_dir / "R17A_CCD_ARRAY_HEADER_AUDIT.csv", arrays)
    write_csv(out_dir / "R17A_CCD_ALIGNMENT_MANIFEST_AUDIT.csv", manifests)
    write_csv(out_dir / "R17A_CCD_SOURCE_CHECKPOINT_AUDIT.csv", ckpts)
    write_csv(out_dir / "R17A_CCD_SOURCE_INFERENCE_CODE_AUDIT.csv", inf_code)
    write_csv(out_dir / "R17A_CCD_ASSET_CODE_REFERENCE_AUDIT.csv", ref_rows)

    (out_dir / "R17A_CCD_DATASET_MAPPING.json").write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "===== R17A CCD DATASET MAPPING FREEZE AUDIT REPORT =====",
        f"Version: {VERSION}",
        f"Root: {root}",
        "",
        "DATASET DECISIONS:",
    ]

    for ds, r in readiness.items():
        lines.append(
            f"{ds}: {r['decision']} | "
            f"soft={r['soft_candidate_count']} "
            f"strong_soft={r['strong_soft_candidate_count']} "
            f"manifests={r['manifest_candidate_count']} "
            f"ckpt={r['checkpoint_count']} "
            f"inference_code={r['inference_code_count']}"
        )
        for c in r["top_soft_candidates"][:5]:
            lines.append(
                f"  SOFT: {c['path']} :: member={c['member']} "
                f"shape={c['shape']} dtype={c['dtype']} "
                f"code_refs={c['code_reference_count']} priority={c['route_priority']}"
            )
        for c in r["top_checkpoints"][:5]:
            lines.append(f"  CKPT: {c}")
        for m in r["top_manifests"][:5]:
            lines.append(
                f"  MANIFEST: {m['path']} :: rows={m['row_count']} "
                f"cols={m['columns_or_keys'][:300]}"
            )

    blocked = [
        ds for ds, r in readiness.items()
        if r["decision"] == "BLOCKED_AMBIGUOUS_MAPPING"
    ]

    lines += [
        "",
        "IMPORTANT:",
        "- This report does not compute CCD.",
        "- Direct soft assets still need exact model-state/sample-axis confirmation in the next score-lock script.",
        "- GT-free re-inference is allowed only with frozen SOURCE checkpoints and pre-existing inference code.",
        "- Hard masks are not accepted as SicTTA CCD inputs.",
        "- No target GT was opened.",
        "",
    ]

    if blocked:
        lines.append(
            "GATE=BLOCKED_R17A_CCD_MAPPING_FOR_" + "_".join(blocked)
        )
        lines.append(
            "NEXT=RESOLVE_ONLY_BLOCKED_DATASETS_BEFORE_SCORE_GENERATION"
        )
    else:
        lines.append("GATE=PASS_R17A_CCD_DATASET_MAPPING_FREEZE")
        lines.append(
            "NEXT=GENERATE_PRE_GT_CCD_SCORE_LOCK_WITH_FROZEN_MAPPING"
        )

    report = out_dir / "R17A_CCD_DATASET_MAPPING_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    for line in lines[-8:]:
        print(line)
    print("Report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
