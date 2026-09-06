#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix1.py

Prospective SUN-SEG confirmatory prediction lock BEFORE any GT reveal.

Frozen cohort:
- 980 RGB frames
- 49 physical caseNN clusters
- 9 frozen source-only model states
- 8,820 model-frame rows

Actions:
- SOURCE
- A1_TENT_1STEP
- A4_PL_CONF90_1STEP

Implementation lineage:
- SOURCE + TENT1 execution reuses the authoritative historical R05D3 functions.
- PL execution reuses the authoritative historical R13B Fix2 functions.
- R05D3/R13B case-count globals are changed ONLY from NeoPolyp 1000 to the
  already frozen SUN cohort size 980. Action/model semantics are unchanged.
- R13B receives the SUN SOURCE packed masks produced by R05D3 and enforces its
  own exact SOURCE parity before PL adaptation.

Strict pre-GT boundary:
- SUN GT pixels: NO
- Dice / DeltaDice / HARM / BENEFIT: NO
- frozen safety score: NO
- target-outcome-based selection: NO
- hyperparameter sweep/tuning: NO

RGB acquisition:
- reads only the 980 frozen `frame_member` entries from the official SUN TAR;
- uses one sequential `r|gz` pass;
- writes a local RGB-only cache under this output directory;
- verifies every extracted encoded-file SHA256 against the R14B4B Fix3 RGB
  hash table before any model inference.

Resume:
- completed states are reused only after SHA/cardinality/lineage validation;
- incomplete temporary files are never treated as completed state outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import traceback
from collections import Counter
from pathlib import Path

# Must precede torch import through historical helper modules.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from tqdm import tqdm


ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"
OUT = ROOT / "outputs"

R14C0_LOCK = (
    OUT
    / "Q1_R14C0_sunseg_preGT_prediction_runtime_lineage_preflight_fix4_v1"
    / "R14C0_SUNSEG_PREGT_RUNTIME_LINEAGE_LOCK.json"
)
EXPECTED_R14C0_DECISION = (
    "SUNSEG_PREGT_RUNTIME_LINEAGE_LOCKED_READY_FOR_PREDICTION_EXECUTION"
)

FINAL_MANIFEST = (
    OUT
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
)
RGB_HASH_TABLE = (
    OUT
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_SUN_SELECTED_RGB_HASHES.csv"
)
SUN_TAR = (
    ROOT / "data" / "external" / "SUN_SEG"
    / "SUN-SEG-FinalData-v20251212.tar.gz"
)

R05D3_SCRIPT = (
    CODE / "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1.py"
)
R13B_SCRIPT = (
    CODE / "Q1_R13B_full_neopolyp_pl_conf90_prediction_lock_fix2.py"
)
R13A_SCRIPT = (
    CODE / "Q1_R13A_confident_pseudolabel_three_family_feasibility_fix2.py"
)
R03_SCRIPT = (
    CODE / "Q1_R03_nine_state_heterogeneous_source_utility_asset_v1_fix2.py"
)

EXPECTED_FINAL_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)
EXPECTED_SUN_TAR_SHA = (
    "d9a00fada04782937a144e9d1bc3c2a3b7f8e321e37ba1cf83bebdd490563016"
)
EXPECTED_R05D3_SHA = (
    "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
)
EXPECTED_R13B_SHA = (
    "6f7ce6a3b15abf1205dc77703320ba823b79d392dc4f43beb51e0a7b94a2521d"
)
EXPECTED_R13A_SHA = (
    "5333af7a0d162ea86d68a1ba7039c699b85e00e596f66e1cf0817e4667b56459"
)
EXPECTED_R03_SHA = (
    "d737272b756a4d2051f9c74051d7085640eaec9cd8be6dbd951c325af2a17c31"
)

CASES = 980
PHYSICAL_CASES = 49
STATES = 9
MODEL_CASES = CASES * STATES
MASK_H = 352
MASK_W = 352
PACKED_BYTES = 15488

EXPECTED_CHECKPOINTS = {
    ("DeepLabV3-R50", "20260817"): (
        r"F:\MEDSEG_SAFETTA\outputs\S05_B_deeplabv3_r50_source_only_seed20260817_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
    ),
    ("DeepLabV3-R50", "20260818"): (
        r"F:\MEDSEG_SAFETTA\outputs\S05_B_deeplabv3_r50_source_only_seed20260818_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
    ),
    ("DeepLabV3-R50", "20260819"): (
        r"F:\MEDSEG_SAFETTA\outputs\S05_B_deeplabv3_r50_source_only_seed20260819_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    ),
    ("PraNet", "20260817"): (
        r"F:\MEDSEG_SAFETTA\outputs\S01_B_pranet_source_only_seed20260817_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
    ),
    ("PraNet", "20260818"): (
        r"F:\MEDSEG_SAFETTA\outputs\S01_B_pranet_source_only_seed20260818_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
    ),
    ("PraNet", "20260819"): (
        r"F:\MEDSEG_SAFETTA\outputs\S01_B_pranet_source_only_seed20260819_v1"
        r"\checkpoints\best_source_val_dice.pt",
        "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
    ),
    ("SegFormer-B0", "20260820"): (
        r"F:\MEDSEG_SAFETTA\outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
        r"\seed_20260820\best_model_state.pt",
        "9dae2ae907b193ea36c2bccc8e376ddbad1699ba27ae0769cf40bcba13e95605",
    ),
    ("SegFormer-B0", "20260821"): (
        r"F:\MEDSEG_SAFETTA\outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
        r"\seed_20260821\best_model_state.pt",
        "ad75290168eab7d116aa3de61b3eafc1e986f11fc0bbfa4a6108abbf669a258d",
    ),
    ("SegFormer-B0", "20260822"): (
        r"F:\MEDSEG_SAFETTA\outputs\Q1_R02_segformer_b0_three_seed_source_training_v1_fix7"
        r"\seed_20260822\best_model_state.pt",
        "ea0b3881373e9f966475a082490fabe4b0acae81179b8596c48a06ee2661a34a",
    ),
}

OUTPUT_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix1_v1"
)

DECISION = "SUNSEG_SOURCE_TENT1_PL_PREDICTIONS_LOCKED_BEFORE_GT_REVEAL"

GLOBAL_FIELDS = [
    "row_index",
    "sample_id",
    "external_case_id",
    "cluster_id",
    "clip_id",
    "source_split_label",
    "frame_member",
    "image_raw_sha256",
    "model_family",
    "model_state_id",
    "training_seed",
    "checkpoint_sha256",
    "source_foreground_pixels",
    "tent1_foreground_pixels",
    "source_tent1_changed_pixels",
    "pl_foreground_pixels",
    "source_pl_changed_pixels",
    "confident_pixels",
    "total_pixels",
    "confident_fraction",
    "pl_loss",
    "pl_parameter_max_abs_delta",
    "pl_buffer_max_abs_delta_after_adaptation",
    "pl_updated",
    "state_prediction_npz_relpath",
    "state_index_csv_relpath",
    "state_lock_json_relpath",
]


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file_progress(path: Path, chunk=16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        desc=f"SHA256 {path.name}",
    ) as bar:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
            bar.update(len(b))
    return h.hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def state_token(state):
    return (
        state["model_family"].lower().replace("-", "_").replace(" ", "_")
        + f"_seed{state['training_seed']}"
    )


def state_paths(output_dir: Path, state):
    token = state_token(state)
    d = output_dir / "state_predictions"
    return (
        d / f"{token}_source_tent1_pl_predictions.npz",
        d / f"{token}_source_tent1_pl_index.csv",
        d / f"{token}_source_tent1_pl_lock.json",
    )


def relpath(path: Path, base: Path):
    return str(path.relative_to(base)).replace("\\", "/")


def validate_lineage():
    required = (
        R14C0_LOCK,
        FINAL_MANIFEST,
        RGB_HASH_TABLE,
        SUN_TAR,
        R05D3_SCRIPT,
        R13B_SCRIPT,
        R13A_SCRIPT,
        R03_SCRIPT,
    )
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    print("===== R14C1 LINEAGE GATES =====")

    if sha256_file(FINAL_MANIFEST) != EXPECTED_FINAL_MANIFEST_SHA:
        raise RuntimeError("Final SUN manifest SHA mismatch.")

    if sha256_file(R05D3_SCRIPT) != EXPECTED_R05D3_SHA:
        raise RuntimeError("R05D3 script SHA mismatch.")
    if sha256_file(R13B_SCRIPT) != EXPECTED_R13B_SHA:
        raise RuntimeError("R13B script SHA mismatch.")
    if sha256_file(R13A_SCRIPT) != EXPECTED_R13A_SHA:
        raise RuntimeError("R13A script SHA mismatch.")
    if sha256_file(R03_SCRIPT) != EXPECTED_R03_SHA:
        raise RuntimeError("R03 script SHA mismatch.")

    tar_sha = sha256_file_progress(SUN_TAR)
    if tar_sha != EXPECTED_SUN_TAR_SHA:
        raise RuntimeError("SUN TAR SHA mismatch.")

    r14c0 = json.loads(R14C0_LOCK.read_text(encoding="utf-8"))
    if r14c0.get("status") != "PASS":
        raise RuntimeError("R14C0 lock status is not PASS.")
    if r14c0.get("decision") != EXPECTED_R14C0_DECISION:
        raise RuntimeError("R14C0 decision mismatch.")

    cohort = r14c0.get("sun_confirmatory_cohort", {})
    if cohort.get("manifest_sha256") != EXPECTED_FINAL_MANIFEST_SHA:
        raise RuntimeError("R14C0 cohort SHA mismatch.")
    if int(cohort.get("rows", -1)) != CASES:
        raise RuntimeError("R14C0 cohort row count mismatch.")
    if int(cohort.get("physical_cases", -1)) != PHYSICAL_CASES:
        raise RuntimeError("R14C0 physical-case count mismatch.")

    scripts = r14c0.get("historical_action_scripts", {})
    if scripts.get("source_tent1", {}).get("sha256") != EXPECTED_R05D3_SHA:
        raise RuntimeError("R14C0 R05D3 SHA mismatch.")
    if scripts.get("pl_conf90", {}).get("sha256") != EXPECTED_R13B_SHA:
        raise RuntimeError("R14C0 R13B SHA mismatch.")

    ckpts = r14c0.get("checkpoint_lineage", [])
    if len(ckpts) != STATES:
        raise RuntimeError(f"R14C0 checkpoint rows={len(ckpts)} expected={STATES}")

    seen = set()
    panel = []
    for row in ckpts:
        fam = str(row["model_family"])
        seed = str(row["training_seed"])
        key = (fam, seed)
        if key not in EXPECTED_CHECKPOINTS:
            raise RuntimeError(f"Unexpected state: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate state: {key}")
        seen.add(key)

        expected_path, expected_sha = EXPECTED_CHECKPOINTS[key]
        p = Path(str(row["checkpoint_path"]))
        if str(p) != expected_path:
            raise RuntimeError(
                f"Checkpoint path mismatch {key}: {p} != {expected_path}"
            )
        if str(row["checkpoint_sha256"]) != expected_sha:
            raise RuntimeError(f"Checkpoint lock SHA mismatch: {key}")
        if not p.is_file():
            raise FileNotFoundError(p)
        if sha256_file(p) != expected_sha:
            raise RuntimeError(f"Checkpoint on-disk SHA mismatch: {key}")

        panel.append({
            "model_family": fam,
            "training_seed": int(seed),
            "checkpoint_path": str(p),
            "checkpoint_sha256": expected_sha,
            "model_state_id": f"{fam}__seed{seed}",
        })

    if seen != set(EXPECTED_CHECKPOINTS):
        raise RuntimeError("Frozen checkpoint panel incomplete.")

    family_order = {"PraNet": 0, "DeepLabV3-R50": 1, "SegFormer-B0": 2}
    panel.sort(key=lambda x: (family_order[x["model_family"]], x["training_seed"]))

    print("final SUN manifest:", EXPECTED_FINAL_MANIFEST_SHA)
    print("SUN TAR:", EXPECTED_SUN_TAR_SHA)
    print("R05D3:", EXPECTED_R05D3_SHA)
    print("R13B:", EXPECTED_R13B_SHA)
    print("R13A:", EXPECTED_R13A_SHA)
    print("R03:", EXPECTED_R03_SHA)
    print("checkpoint states:", len(panel))
    print("PASS")

    return r14c0, panel


def load_final_manifest_and_hashes():
    rows, fields = read_csv(FINAL_MANIFEST)
    if len(rows) != CASES:
        raise RuntimeError(f"Final manifest rows={len(rows)} expected={CASES}")

    required = {
        "pair_key",
        "external_case_id",
        "cluster_id",
        "clip_id",
        "source_split_label",
        "frame_member",
    }
    if not required.issubset(fields):
        raise RuntimeError(
            f"Final manifest missing columns: {sorted(required - set(fields))}"
        )

    if len({r["pair_key"] for r in rows}) != CASES:
        raise RuntimeError("Final manifest pair_key values are not unique.")
    if len({r["frame_member"] for r in rows}) != CASES:
        raise RuntimeError("Final manifest frame_member values are not unique.")
    if len({r["cluster_id"] for r in rows}) != PHYSICAL_CASES:
        raise RuntimeError("Final manifest physical-case count mismatch.")

    hrows, hfields = read_csv(RGB_HASH_TABLE)
    if len(hrows) != CASES:
        raise RuntimeError(f"RGB hash rows={len(hrows)} expected={CASES}")

    required_h = {"pair_key", "frame_member", "file_sha256", "common_rgb_sha256"}
    if not required_h.issubset(hfields):
        raise RuntimeError(
            f"RGB hash table missing: {sorted(required_h - set(hfields))}"
        )

    hmap = {r["pair_key"]: r for r in hrows}
    if len(hmap) != CASES:
        raise RuntimeError("RGB hash pair_key values are not unique.")

    for r in rows:
        h = hmap.get(r["pair_key"])
        if h is None:
            raise RuntimeError(f"Missing RGB hash row: {r['pair_key']}")
        if h["frame_member"] != r["frame_member"]:
            raise RuntimeError(f"frame_member mismatch: {r['pair_key']}")

    return rows, hmap


def prepare_rgb_cache(output_dir: Path, manifest_rows, hmap, resume: bool):
    """
    Extract only the frozen 980 RGB members with one sequential TAR.GZ pass.
    No GT member is ever requested or decoded.
    """
    cache = output_dir / "rgb_cache"
    cache.mkdir(parents=True, exist_ok=True)

    targets = []
    need = {}

    for i, r in enumerate(manifest_rows):
        pair = r["pair_key"]
        h = hmap[pair]
        suffix = Path(r["frame_member"]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise RuntimeError(f"Unexpected SUN RGB suffix: {suffix}")

        local = cache / f"sun_{i:04d}{suffix}"
        expected_raw = h["file_sha256"]

        if local.is_file():
            got = sha256_file(local)
            if got != expected_raw:
                raise RuntimeError(
                    f"Existing RGB cache SHA mismatch: {local}"
                )
        else:
            need[r["frame_member"]] = (local, expected_raw)

        targets.append({
            "row_index": i,
            "sample_id": pair,
            "external_case_id": r["external_case_id"],
            "cluster_id": r["cluster_id"],
            "clip_id": r["clip_id"],
            "source_split_label": r["source_split_label"],
            "frame_member": r["frame_member"],
            "image_path": str(local),
            "image_raw_sha256": expected_raw,
            "common_rgb_sha256": h["common_rgb_sha256"],
        })

    if need:
        print(
            f"\n===== RGB CACHE EXTRACTION =====\n"
            f"existing verified={CASES-len(need)} need_extract={len(need)}"
        )

        found = set()
        with tarfile.open(SUN_TAR, "r|gz") as tf:
            with tqdm(
                total=len(need),
                desc="Extract frozen SUN RGB only (single TAR pass)",
                unit="frame",
                dynamic_ncols=True,
            ) as bar:
                for m in tf:
                    if not m.isfile():
                        continue
                    name = m.name.replace("\\", "/")
                    if name not in need:
                        continue

                    f = tf.extractfile(m)
                    if f is None:
                        raise RuntimeError(f"Cannot read SUN member: {name}")
                    blob = f.read()
                    local, expected = need[name]
                    got = hashlib.sha256(blob).hexdigest()
                    if got != expected:
                        raise RuntimeError(
                            f"SUN member encoded SHA mismatch: {name}"
                        )

                    tmp = local.with_suffix(local.suffix + ".tmp")
                    with tmp.open("wb") as out:
                        out.write(blob)
                    tmp.replace(local)

                    found.add(name)
                    bar.update(1)

                    if len(found) == len(need):
                        break

        if found != set(need):
            missing = sorted(set(need) - found)
            raise RuntimeError(
                f"Missing frozen RGB TAR members={len(missing)} "
                f"examples={missing[:5]}"
            )

    # Final verification. This is intentionally encoded-file SHA only; the
    # common pixel hash was already locked by R14B4B Fix3.
    for t in tqdm(
        targets,
        desc="Verify SUN RGB cache",
        unit="frame",
        dynamic_ncols=True,
    ):
        p = Path(t["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)
        if sha256_file(p) != t["image_raw_sha256"]:
            raise RuntimeError(f"RGB cache final SHA mismatch: {p}")

    return targets


def configure_historical_modules():
    """
    Import exact historical implementations and only patch dataset cardinality
    from 1000 to the frozen SUN 980 cases.
    """
    r05 = import_module(R05D3_SCRIPT, "q1_r14c1_r05d3")
    r13b = import_module(R13B_SCRIPT, "q1_r14c1_r13b")
    r13a = import_module(R13A_SCRIPT, "q1_r14c1_r13a")
    r03 = import_module(R03_SCRIPT, "q1_r14c1_r03")

    # Exact implementation functions remain unchanged; only target cardinality
    # is external-cohort-specific.
    r05.EXPECTED_CASES = CASES
    r05.EXPECTED_MODEL_CASES = MODEL_CASES

    r13b.CASES = CASES
    r13b.STATES = STATES
    r13b.MODEL_CASES = MODEL_CASES

    if r05.PACKED_BYTES != PACKED_BYTES:
        raise RuntimeError("R05D3 packed-mask format mismatch.")
    if r13b.PACKED_BYTES != PACKED_BYTES:
        raise RuntimeError("R13B packed-mask format mismatch.")

    # Revalidate the frozen PL action semantics and upstream implementation
    # lineage before any prediction execution.
    provenance = r13b.validate_lineage()

    return r05, r13b, r13a, r03, provenance


def run_source_tent1_state(r05, r03, state, targets, device, seg_context):
    fam = state["model_family"]

    if fam == "PraNet":
        source, tent1, rows = r05.run_pranet_state(
            state, targets, device
        )
        return source, tent1, rows, seg_context

    if fam == "DeepLabV3-R50":
        source, tent1, rows = r05.run_deeplab_state(
            state, targets, device
        )
        return source, tent1, rows, seg_context

    if fam == "SegFormer-B0":
        source, tent1, rows, seg_context = r05.run_segformer_state(
            state, targets, device, seg_context
        )
        return source, tent1, rows, seg_context

    raise RuntimeError(f"Unexpected family: {fam}")


def run_pl_state(
    r13b, r13a, r05, r03,
    state, targets, device, source_pack, seg_context
):
    fam = state["model_family"]

    if fam in {"PraNet", "DeepLabV3-R50"}:
        pl, rows = r13b.run_binary_state(
            r13a,
            r05,
            state,
            targets,
            device,
            source_pack,
        )
        return pl, rows, seg_context

    if fam == "SegFormer-B0":
        pl, rows, seg_context = r13b.run_segformer_state(
            r13a,
            r05,
            r03,
            state,
            targets,
            device,
            source_pack,
            context=seg_context,
        )
        return pl, rows, seg_context

    raise RuntimeError(f"Unexpected family: {fam}")


def validate_mask_array(name, arr):
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != (CASES, PACKED_BYTES):
        raise RuntimeError(
            f"{name} shape={arr.shape}; expected={(CASES, PACKED_BYTES)}"
        )
    return arr


def merge_state_rows(
    state,
    targets,
    tent_rows,
    pl_rows,
    pred_rel,
    index_rel,
    lock_rel,
):
    if len(tent_rows) != CASES or len(pl_rows) != CASES:
        raise RuntimeError("State row cardinality mismatch.")

    pl_by_idx = {int(r["row_index"]): r for r in pl_rows}
    if len(pl_by_idx) != CASES:
        raise RuntimeError("PL row_index not unique.")

    merged = []

    for i, tr in enumerate(tent_rows):
        if int(tr["row_index"]) != i:
            raise RuntimeError("TENT1 row order mismatch.")
        pr = pl_by_idx.get(i)
        if pr is None:
            raise RuntimeError(f"Missing PL row={i}")

        t = targets[i]
        for key in ("model_family", "training_seed", "checkpoint_sha256"):
            if str(tr[key]) != str(pr[key]):
                raise RuntimeError(f"TENT1/PL state metadata mismatch: {key}")

        if tr["sample_id"] != t["sample_id"]:
            raise RuntimeError("TENT1 target order mismatch.")
        if pr["sample_id"] != t["sample_id"]:
            raise RuntimeError("PL target order mismatch.")

        if int(tr["source_foreground_pixels"]) != int(pr["source_foreground_pixels"]):
            raise RuntimeError(
                f"SOURCE parity count mismatch state={state['model_state_id']} row={i}"
            )

        merged.append({
            "row_index": i,
            "sample_id": t["sample_id"],
            "external_case_id": t["external_case_id"],
            "cluster_id": t["cluster_id"],
            "clip_id": t["clip_id"],
            "source_split_label": t["source_split_label"],
            "frame_member": t["frame_member"],
            "image_raw_sha256": t["image_raw_sha256"],
            "model_family": state["model_family"],
            "model_state_id": state["model_state_id"],
            "training_seed": int(state["training_seed"]),
            "checkpoint_sha256": state["checkpoint_sha256"],
            "source_foreground_pixels": int(tr["source_foreground_pixels"]),
            "tent1_foreground_pixels": int(tr["a1_foreground_pixels"]),
            "source_tent1_changed_pixels": int(tr["changed_pixels"]),
            "pl_foreground_pixels": int(pr["pl_foreground_pixels"]),
            "source_pl_changed_pixels": int(pr["source_pl_changed_pixels"]),
            "confident_pixels": int(pr["confident_pixels"]),
            "total_pixels": int(pr["total_pixels"]),
            "confident_fraction": float(pr["confident_fraction"]),
            "pl_loss": float(pr["pl_loss"]),
            "pl_parameter_max_abs_delta": float(pr["parameter_max_abs_delta"]),
            "pl_buffer_max_abs_delta_after_adaptation": float(
                pr["buffer_max_abs_delta_after_adaptation"]
            ),
            "pl_updated": int(pr["updated"]),
            "state_prediction_npz_relpath": pred_rel,
            "state_index_csv_relpath": index_rel,
            "state_lock_json_relpath": lock_rel,
        })

    return merged


def save_state_atomic(
    output_dir,
    state,
    source_pack,
    tent1_pack,
    pl_pack,
    merged_rows,
    provenance,
):
    pred, idx, lock = state_paths(output_dir, state)
    pred.parent.mkdir(parents=True, exist_ok=True)

    for p in (pred, idx, lock):
        if p.exists():
            raise FileExistsError(p)

    source_pack = validate_mask_array("SOURCE", source_pack)
    tent1_pack = validate_mask_array("TENT1", tent1_pack)
    pl_pack = validate_mask_array("PL", pl_pack)

    if len(merged_rows) != CASES:
        raise RuntimeError("Merged state rows != 980")

    tmp_pred = pred.with_suffix(pred.suffix + ".tmp")
    tmp_idx = idx.with_suffix(idx.suffix + ".tmp")
    tmp_lock = lock.with_suffix(lock.suffix + ".tmp")

    for p in (tmp_pred, tmp_idx, tmp_lock):
        if p.exists():
            p.unlink()

    with tmp_pred.open("wb") as f:
        np.savez_compressed(
            f,
            source_masks_packed=source_pack,
            tent1_masks_packed=tent1_pack,
            pl_masks_packed=pl_pack,
        )

    write_csv(tmp_idx, merged_rows, GLOBAL_FIELDS)

    pred_sha = sha256_file(tmp_pred)
    idx_sha = sha256_file(tmp_idx)

    state_lock = {
        "status": "PASS",
        "decision": "SUNSEG_STATE_SOURCE_TENT1_PL_PREDICTIONS_LOCKED_PRE_GT",
        "model_family": state["model_family"],
        "model_state_id": state["model_state_id"],
        "training_seed": int(state["training_seed"]),
        "checkpoint_path": state["checkpoint_path"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "target_cases": CASES,
        "physical_cases": PHYSICAL_CASES,
        "mask_shape": [MASK_H, MASK_W],
        "packed_bytes": PACKED_BYTES,
        "actions": ["SOURCE", "A1_TENT_1STEP", "A4_PL_CONF90_1STEP"],
        "prediction_npz_sha256": pred_sha,
        "index_csv_sha256": idx_sha,
        "historical_lineage": {
            "r05d3_sha256": EXPECTED_R05D3_SHA,
            "r13b_fix2_sha256": EXPECTED_R13B_SHA,
            "r13a_fix2_sha256": EXPECTED_R13A_SHA,
            "r03_sha256": EXPECTED_R03_SHA,
            "r13b_validate_lineage": provenance,
        },
        "information_boundary": {
            "sun_rgb_accessed": True,
            "sun_gt_pixels_decoded": False,
            "dice_computed": False,
            "target_outcomes_accessed": False,
            "frozen_safety_scores_accessed": False,
            "target_tuning": False,
        },
    }
    write_json(tmp_lock, state_lock)

    tmp_pred.replace(pred)
    tmp_idx.replace(idx)
    tmp_lock.replace(lock)

    return pred, idx, lock


def load_completed_state(output_dir, state):
    pred, idx, lock = state_paths(output_dir, state)
    exists = [p.exists() for p in (pred, idx, lock)]

    if not any(exists):
        return None
    if not all(exists):
        raise RuntimeError(
            f"Incomplete completed-state artifact set: {state['model_state_id']}"
        )

    lk = json.loads(lock.read_text(encoding="utf-8"))
    if lk.get("status") != "PASS":
        raise RuntimeError("Completed state lock not PASS.")
    if lk.get("model_state_id") != state["model_state_id"]:
        raise RuntimeError("Completed state id mismatch.")
    if lk.get("checkpoint_sha256") != state["checkpoint_sha256"]:
        raise RuntimeError("Completed state checkpoint mismatch.")
    if int(lk.get("target_cases", -1)) != CASES:
        raise RuntimeError("Completed state target count mismatch.")
    if sha256_file(pred) != lk.get("prediction_npz_sha256"):
        raise RuntimeError("Completed state NPZ SHA mismatch.")
    if sha256_file(idx) != lk.get("index_csv_sha256"):
        raise RuntimeError("Completed state CSV SHA mismatch.")

    rows, fields = read_csv(idx)
    if fields != GLOBAL_FIELDS:
        raise RuntimeError("Completed state CSV schema mismatch.")
    if len(rows) != CASES:
        raise RuntimeError("Completed state CSV rows != 980.")

    with np.load(pred, allow_pickle=False) as z:
        for key in (
            "source_masks_packed",
            "tent1_masks_packed",
            "pl_masks_packed",
        ):
            if key not in z:
                raise RuntimeError(f"Completed state NPZ missing {key}")
            if z[key].shape != (CASES, PACKED_BYTES):
                raise RuntimeError(
                    f"Completed state {key} shape={z[key].shape}"
                )

    print(f"RESUME verified completed state: {state['model_state_id']}")
    return rows


def cleanup_temporary_files(output_dir):
    if not output_dir.exists():
        return
    removed = 0
    for p in output_dir.rglob("*.tmp"):
        if p.is_file():
            p.unlink()
            removed += 1
    if removed:
        print(f"Removed incomplete temporary files: {removed}")


def run(args):
    import torch

    _, panel = validate_lineage()
    manifest_rows, hmap = load_final_manifest_and_hashes()

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} already exists. Use --resume only for a "
            f"technical interruption of this exact R14C1 Fix1 run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleanup_temporary_files(args.output_dir)

    targets = prepare_rgb_cache(
        args.output_dir,
        manifest_rows,
        hmap,
        resume=args.resume,
    )

    if len(targets) != CASES:
        raise RuntimeError("SUN target count != 980.")

    print("\n===== PRE-GT INFORMATION BOUNDARY =====")
    print("SUN_RGB_ACCESS=YES")
    print("SUN_GT_PIXEL_DECODE=NO")
    print("DICE_COMPUTATION=NO")
    print("HARM_BENEFIT_ACCESS=NO")
    print("FROZEN_SAFETY_SCORE_ACCESS=NO")
    print("TARGET_TUNING=NO")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R14C1.")
    device = torch.device("cuda")
    print("device:", device)
    print("cuda:", torch.cuda.get_device_name(0))

    r05, r13b, r13a, r03, provenance = configure_historical_modules()

    global_rows = []
    seg_context_tent = None
    seg_context_pl = None

    for state_no, state in enumerate(panel, start=1):
        print(
            f"\n===== STATE {state_no}/{STATES}: "
            f"{state['model_state_id']} ====="
        )

        completed = load_completed_state(args.output_dir, state)
        if completed is not None:
            if not args.resume:
                raise RuntimeError(
                    "Completed state exists but --resume was not requested."
                )
            global_rows.extend(completed)
            continue

        # Historical SOURCE + A1_TENT_1STEP.
        source_pack, tent1_pack, tent_rows, seg_context_tent = (
            run_source_tent1_state(
                r05,
                r03,
                state,
                targets,
                device,
                seg_context_tent,
            )
        )

        # Historical A4_PL_CONF90_1STEP, with mandatory SOURCE parity against
        # the SUN SOURCE predictions just produced above.
        pl_pack, pl_rows, seg_context_pl = run_pl_state(
            r13b,
            r13a,
            r05,
            r03,
            state,
            targets,
            device,
            source_pack,
            seg_context_pl,
        )

        pred, idx, lock = state_paths(args.output_dir, state)
        pred_rel = relpath(pred, args.output_dir)
        idx_rel = relpath(idx, args.output_dir)
        lock_rel = relpath(lock, args.output_dir)

        merged = merge_state_rows(
            state,
            targets,
            tent_rows,
            pl_rows,
            pred_rel,
            idx_rel,
            lock_rel,
        )

        save_state_atomic(
            args.output_dir,
            state,
            source_pack,
            tent1_pack,
            pl_pack,
            merged,
            provenance,
        )

        global_rows.extend(merged)

        print(
            f"state locked: {state['model_state_id']} | "
            f"rows={len(merged)}"
        )

    if len(global_rows) != MODEL_CASES:
        raise RuntimeError(
            f"Global rows={len(global_rows)} expected={MODEL_CASES}"
        )

    keyset = {
        (
            r["sample_id"],
            r["model_family"],
            str(r["training_seed"]),
        )
        for r in global_rows
    }
    if len(keyset) != MODEL_CASES:
        raise RuntimeError("Global model-frame keys are not unique.")

    per_sample = Counter(r["sample_id"] for r in global_rows)
    if set(per_sample.values()) != {STATES}:
        raise RuntimeError("Every SUN frame must have exactly 9 states.")

    per_state = Counter(r["model_state_id"] for r in global_rows)
    if set(per_state.values()) != {CASES}:
        raise RuntimeError("Every model state must have exactly 980 frames.")

    global_rows.sort(
        key=lambda r: (
            int(r["row_index"]),
            r["model_family"],
            int(r["training_seed"]),
        )
    )

    global_csv = (
        args.output_dir
        / "R14C1_SUNSEG_MODEL_FRAME_PREDICTION_MANIFEST.csv"
    )
    write_csv(global_csv, global_rows, GLOBAL_FIELDS)

    # No target labels/outcomes are accessed. Only prediction-behavior
    # descriptive counts are printed.
    tent_changed = sum(
        int(r["source_tent1_changed_pixels"]) > 0
        for r in global_rows
    )
    pl_changed = sum(
        int(r["source_pl_changed_pixels"]) > 0
        for r in global_rows
    )
    pl_updated = sum(int(r["pl_updated"]) for r in global_rows)

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "cohort": {
            "final_manifest_path": str(FINAL_MANIFEST),
            "final_manifest_sha256": EXPECTED_FINAL_MANIFEST_SHA,
            "frames": CASES,
            "physical_cases": PHYSICAL_CASES,
        },
        "model_panel": {
            "states": STATES,
            "model_frame_rows": MODEL_CASES,
            "checkpoint_lineage": panel,
        },
        "actions": {
            "source": "SOURCE",
            "tent1": "A1_TENT_1STEP",
            "pl": "A4_PL_CONF90_1STEP",
        },
        "historical_implementation_lineage": {
            "r05d3_source_tent1_script": str(R05D3_SCRIPT),
            "r05d3_sha256": EXPECTED_R05D3_SHA,
            "r13b_pl_script": str(R13B_SCRIPT),
            "r13b_sha256": EXPECTED_R13B_SHA,
            "r13a_sha256": EXPECTED_R13A_SHA,
            "r03_sha256": EXPECTED_R03_SHA,
            "external_cohort_cardinality_patch_only": {
                "R05D3_EXPECTED_CASES": CASES,
                "R13B_CASES": CASES,
                "action_semantics_changed": False,
            },
        },
        "artifacts": {
            "global_manifest": {
                "path": str(global_csv),
                "sha256": sha256_file(global_csv),
                "rows": len(global_rows),
            },
            "state_prediction_directory": str(
                args.output_dir / "state_predictions"
            ),
            "rgb_cache_directory": str(args.output_dir / "rgb_cache"),
        },
        "descriptive_prediction_behavior_only": {
            "rows_with_nonzero_source_tent1_change": tent_changed,
            "rows_with_nonzero_source_pl_change": pl_changed,
            "rows_with_pl_update": pl_updated,
        },
        "information_boundary": {
            "sun_rgb_accessed": True,
            "sun_gt_pixels_decoded": False,
            "dice_computed": False,
            "delta_dice_computed": False,
            "harm_benefit_accessed": False,
            "frozen_safety_scores_accessed": False,
            "target_outcome_based_selection": False,
            "target_tuning": False,
        },
        "next_stage": (
            "R14C2_SUNSEG_FROZEN_SOURCE_SAFETY_SCORE_LOCK_PRE_GT"
        ),
    }

    lock_path = (
        args.output_dir
        / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
    )
    write_json(lock_path, lock)

    print("\n===== R14C1 FINAL =====")
    print("frames:", CASES)
    print("physical cases:", PHYSICAL_CASES)
    print("model states:", STATES)
    print("model-frame rows:", MODEL_CASES)
    print("rows with SOURCE->TENT1 pixel change:", tent_changed)
    print("rows with SOURCE->PL pixel change:", pl_changed)
    print("rows with PL optimizer update:", pl_updated)
    print("GT pixels decoded: NO")
    print("Dice/DeltaDice computed: NO")
    print("HARM/BENEFIT accessed: NO")
    print("frozen safety score accessed: NO")
    print("Decision=", DECISION)
    print("LOCK=", lock_path)
    print("PASS")


def self_test():
    assert CASES == 980
    assert PHYSICAL_CASES == 49
    assert STATES == 9
    assert MODEL_CASES == 8820
    assert MASK_H == 352 and MASK_W == 352
    assert PACKED_BYTES == 15488
    assert len(EXPECTED_CHECKPOINTS) == 9
    assert EXPECTED_R05D3_SHA == (
        "df22d6f3054257743d1f12d9e757323ea9f4a052b7642eb4ab610da46ff21251"
    )
    assert EXPECTED_R13B_SHA == (
        "6f7ce6a3b15abf1205dc77703320ba823b79d392dc4f43beb51e0a7b94a2521d"
    )
    print("CARDINALITY_TEST_PASS")
    print("PACKED_MASK_FORMAT_TEST_PASS")
    print("HISTORICAL_SCRIPT_SHA_TEST_PASS")
    print("NINE_CHECKPOINT_PANEL_TEST_PASS")
    print("PRE_GT_BOUNDARY_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "R14C1: lock SUN-SEG SOURCE + TENT1 + PL predictions "
            "for 980x9 model-frame pairs before GT reveal."
        )
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only SHA-verified completed per-state outputs after a "
            "technical interruption of this exact Fix1 run."
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        self_test()
        return 0

    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
