#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1.py

SafeTTA Q1 enhancement — CM6.

This is the FIRST stage allowed to reveal PROMISE12 segmentation GT.

It evaluates only policies that were already locked before GT:
1) fixed SOURCE deployment threshold;
2) naive unlabeled target-quantile baseline;
3) support-aware transport -- here prospectively locked as ABSTAIN;
4) oracle target threshold at HARM recall >= 0.90, reference only.

No post-GT threshold/model/representation changes are allowed.

Primary external ranking endpoint:
- macro-family target HARM AUROC;
- 2000x patient-cluster bootstrap 95% CI.

HARM remains exactly:
    DeltaDice = TENT1 slice Dice - SOURCE slice Dice
    HARM iff DeltaDice <= -0.02

Dice convention:
- binary whole gland;
- both-empty slice Dice = 1;
- patient-level 3D Dice from pooled voxel TP/FP/FN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-05-Q1X-CM6-v1-fix1"
BUILD = "Q1X_CM6_PROMISE12_GT_REVEAL_LOCKED_POLICY_EVALUATION_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

PROMISE_ZIP = (
    ROOT
    / "cross_modality_prostate_mri"
    / "data"
    / "raw"
    / "PROMISE12"
    / "training_data.zip"
)
EXPECTED_PROMISE_ZIP_SHA256 = (
    "150287d0c74cd0105d8b70b43af4a7bf4f1fd3e1748829779177a0ccaf948f45"
)

CM5A_LOCK = (
    OUT
    / "Q1X_CM5A_promise12_gt_free_prediction_tent1_safety_scoring_lock_fix1_v1"
    / "CM5A_PROMISE12_GT_FREE_PREDICTION_SAFETY_LOCK.json"
)
EXPECTED_CM5A_LOCK_SHA = (
    "18198bf417ec30434fedcab39f8f50bff270fade676e47fccfa1e397fc2c93fd"
)

CM5B_LOCK = (
    OUT
    / "Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix2_v1"
    / "CM5B_UNLABELED_OPERATING_POINT_TRANSPORT_LOCK.json"
)
EXPECTED_CM5B_LOCK_SHA = (
    "174e1c80869a2c559e6b8ffccef3798b191cfd29ee4f40f62072d2ea9fc2d9c5"
)

CM5B_HELPER = (
    CODE
    / "Q1X_CM5B_unlabeled_support_aware_operating_point_transport_lock_fix2.py"
)
EXPECTED_CM5B_HELPER_SHA = (
    "3ef0aa921a69fa2db877ac39369bed676edbb52e5c18f99a2d123870eeb3a908"
)

DEFAULT_OUTPUT = (
    OUT
    / "Q1X_CM6_promise12_gt_reveal_locked_policy_evaluation_fix1_v1"
)

FAMILIES = ["UNet", "DeepLabV3_R50", "SegFormer_B0"]
N_PATIENTS = 50
N_SLICES = 1377
SEG_SIZE = 352
PACKED_BYTES = (SEG_SIZE * SEG_SIZE + 7) // 8

HARM_THRESHOLD = -0.02
BENEFIT_THRESHOLD = +0.02
ORACLE_RECALL_TARGET = 0.90

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260906

SEG_MHD_RE = re.compile(
    r"^Case(\d{2})_segmentation\.mhd$",
    re.IGNORECASE,
)

PASS_RANKING_DECISION = (
    "PROMISE12_INDEPENDENT_MRI_HARM_RANKING_CONFIRMED"
)
FAIL_RANKING_DECISION = (
    "PROMISE12_INDEPENDENT_MRI_HARM_RANKING_NOT_CONFIRMED"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_lineage():
    for label, path, expected in [
        ("CM5A_LOCK", CM5A_LOCK, EXPECTED_CM5A_LOCK_SHA),
        ("CM5B_LOCK", CM5B_LOCK, EXPECTED_CM5B_LOCK_SHA),
        ("CM5B_HELPER", CM5B_HELPER, EXPECTED_CM5B_HELPER_SHA),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        print(label, got, "PASS" if got == expected else "FAIL")
        if got != expected:
            raise RuntimeError(f"{label} SHA mismatch.")

    if sha256_file(PROMISE_ZIP) != EXPECTED_PROMISE_ZIP_SHA256:
        raise RuntimeError("PROMISE12 archive SHA mismatch.")

    cm5a = load_json(CM5A_LOCK)
    cm5b = load_json(CM5B_LOCK)

    if cm5a.get("status") != "PASS":
        raise RuntimeError("CM5A status changed.")
    if cm5b.get("status") != "ABSTAIN":
        raise RuntimeError(
            "Expected prospectively locked CM5B ABSTAIN status."
        )

    expected_decision = (
        "UNLABELED_OPERATING_POINT_TRANSPORT_ABSTAIN_SUPPORT_INADEQUATE_"
        "READY_FOR_CM6_GT_REVEAL_AS_ABSTENTION_ANALYSIS"
    )
    if cm5b.get("decision") != expected_decision:
        raise RuntimeError("Unexpected CM5B decision.")

    if cm5b.get("information_boundary", {}).get(
        "promises12_gt_access"
    ) is not False:
        raise RuntimeError("CM5B GT boundary changed.")
    if cm5b.get("information_boundary", {}).get(
        "target_harm_labels"
    ) is not False:
        raise RuntimeError("CM5B target HARM boundary changed.")
    if cm5b.get("information_boundary", {}).get(
        "target_delta_dice"
    ) is not False:
        raise RuntimeError("CM5B target DeltaDice boundary changed.")

    transport = cm5b.get("support_aware_transport", {})
    if transport.get("status") != "ABSTAIN":
        raise RuntimeError(
            "CM5B support-aware transport was not locked as ABSTAIN."
        )

    return cm5a, cm5b


def parse_mhd_header_bytes(data: bytes):
    text = data.decode("ascii")
    fields = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    required = [
        "NDims",
        "DimSize",
        "ElementSpacing",
        "ElementType",
        "ElementDataFile",
    ]
    missing = [k for k in required if k not in fields]
    if missing:
        raise RuntimeError(f"MHD header missing fields: {missing}")

    if int(fields["NDims"]) != 3:
        raise RuntimeError("Expected 3D PROMISE12 GT.")

    dims = [int(v) for v in fields["DimSize"].split()]
    spacing = [float(v) for v in fields["ElementSpacing"].split()]

    if len(dims) != 3 or len(spacing) != 3:
        raise RuntimeError("Malformed GT DimSize/ElementSpacing.")

    element_type = fields["ElementType"].upper()
    if element_type not in {"MET_CHAR", "MET_UCHAR"}:
        raise RuntimeError(
            f"Unexpected PROMISE GT ElementType={element_type}"
        )

    compressed = fields.get("CompressedData", "False").lower() == "true"
    if compressed:
        raise RuntimeError("Compressed GT RAW not expected.")

    return {
        "fields": fields,
        "dims_xyz": dims,
        "spacing_xyz": spacing,
        "element_type": element_type,
        "raw_ref": fields["ElementDataFile"],
    }


def resolve_raw_member(zf, mhd_member, raw_ref):
    parent = Path(mhd_member).parent
    candidate = (parent / raw_ref).as_posix()

    names = set(zf.namelist())
    if candidate in names:
        return candidate

    base = Path(raw_ref).name.lower()
    matches = [
        name
        for name in names
        if Path(name).name.lower() == base
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Unable to uniquely resolve GT RAW {raw_ref}"
        )
    return matches[0]


def load_cm5a_target_slice_manifest(cm5a):
    cache_meta_path = Path(
        cm5a["artifacts"]["target_cache_meta"]
    )
    if sha256_file(cache_meta_path) != cm5a[
        "artifacts"
    ]["target_cache_meta_sha256"]:
        raise RuntimeError("CM5A target cache-meta SHA mismatch.")

    cache_meta = load_json(cache_meta_path)

    slices_path = Path(cache_meta["slices"])
    volumes_path = Path(cache_meta["volumes"])

    if sha256_file(slices_path) != cache_meta["slices_sha256"]:
        raise RuntimeError("CM5A slice-manifest SHA mismatch.")
    if sha256_file(volumes_path) != cache_meta["volumes_sha256"]:
        raise RuntimeError("CM5A volume-manifest SHA mismatch.")

    slices = pd.read_csv(slices_path)
    volumes = pd.read_csv(volumes_path)

    if len(slices) != N_SLICES:
        raise RuntimeError("Unexpected PROMISE slice count.")
    if slices["case_key"].nunique() != N_PATIENTS:
        raise RuntimeError("Unexpected PROMISE patient count.")
    if len(volumes) != N_PATIENTS:
        raise RuntimeError("Unexpected PROMISE volume count.")

    return slices, volumes, cache_meta


def build_segmentation_member_manifest():
    rows = []

    with zipfile.ZipFile(PROMISE_ZIP, "r") as zf:
        members = []

        for name in zf.namelist():
            base = Path(name).name
            m = SEG_MHD_RE.fullmatch(base)
            if m is not None:
                members.append((int(m.group(1)), name))

        members.sort(key=lambda x: x[0])

        if len(members) != N_PATIENTS:
            raise RuntimeError(
                f"Expected 50 GT MHD members, got {len(members)}"
            )

        if [x[0] for x in members] != list(range(N_PATIENTS)):
            raise RuntimeError("PROMISE GT case IDs changed.")

        for case_id, mhd_member in members:
            header = parse_mhd_header_bytes(zf.read(mhd_member))
            raw_member = resolve_raw_member(
                zf,
                mhd_member,
                header["raw_ref"],
            )

            rows.append({
                "case_id": case_id,
                "case_key": f"Case{case_id:02d}",
                "mhd_member": mhd_member,
                "raw_member": raw_member,
                "dim_x": header["dims_xyz"][0],
                "dim_y": header["dims_xyz"][1],
                "dim_z": header["dims_xyz"][2],
                "spacing_x": header["spacing_xyz"][0],
                "spacing_y": header["spacing_xyz"][1],
                "spacing_z": header["spacing_xyz"][2],
                "element_type": header["element_type"],
            })

    return pd.DataFrame(rows)


def audit_gt_geometry(gt_manifest, image_volumes):
    merged = gt_manifest.merge(
        image_volumes[
            [
                "case_id",
                "case_key",
                "dim_x",
                "dim_y",
                "dim_z",
                "spacing_x",
                "spacing_y",
                "spacing_z",
            ]
        ],
        on=["case_id", "case_key"],
        suffixes=("_gt", "_img"),
        validate="one_to_one",
    )

    for axis in ["x", "y", "z"]:
        if not np.array_equal(
            merged[f"dim_{axis}_gt"].to_numpy(dtype=np.int64),
            merged[f"dim_{axis}_img"].to_numpy(dtype=np.int64),
        ):
            raise RuntimeError(
                f"PROMISE GT/image dimension mismatch on axis {axis}."
            )

        diff = np.abs(
            merged[f"spacing_{axis}_gt"].to_numpy(dtype=np.float64)
            - merged[f"spacing_{axis}_img"].to_numpy(dtype=np.float64)
        )
        if float(diff.max()) > 1e-5:
            raise RuntimeError(
                f"PROMISE GT/image spacing mismatch on axis {axis}: "
                f"max={float(diff.max())}"
            )

    print("\n===== PROMISE12 GT GEOMETRY AUDIT =====")
    print("patients:", len(merged))
    print("image/GT DimSize exact:", "PASS")
    print("image/GT ElementSpacing tolerance 1e-5:", "PASS")
    print("GT ElementType counts:")
    print(
        merged["element_type"].value_counts().to_string()
    )


def center_crop_pad_batch(x, torch):
    _, _, h, w = x.shape

    if h > SEG_SIZE:
        top = (h - SEG_SIZE) // 2
        x = x[:, :, top:top + SEG_SIZE, :]
    elif h < SEG_SIZE:
        total = SEG_SIZE - h
        top = total // 2
        bottom = total - top
        x = torch.nn.functional.pad(
            x, (0, 0, top, bottom), value=0
        )

    w = x.shape[-1]
    if w > SEG_SIZE:
        left = (w - SEG_SIZE) // 2
        x = x[:, :, :, left:left + SEG_SIZE]
    elif w < SEG_SIZE:
        total = SEG_SIZE - w
        left = total // 2
        right = total - left
        x = torch.nn.functional.pad(
            x, (left, right, 0, 0), value=0
        )

    if tuple(x.shape[-2:]) != (SEG_SIZE, SEG_SIZE):
        raise RuntimeError(
            f"Unexpected GT crop/pad result {tuple(x.shape)}"
        )

    return x


def pack_masks(mask_batch):
    flat = np.asarray(mask_batch, dtype=np.uint8).reshape(
        len(mask_batch), -1
    )
    return np.packbits(flat, axis=1, bitorder="little")


def unpack_masks(packed_batch):
    p = np.asarray(packed_batch, dtype=np.uint8)
    flat = np.unpackbits(
        p,
        axis=1,
        count=SEG_SIZE * SEG_SIZE,
        bitorder="little",
    )
    return flat.reshape(-1, SEG_SIZE, SEG_SIZE).astype(bool)


def build_gt_cache(
    gt_manifest,
    image_volumes,
    target_slices,
    stage_dir,
    torch,
):
    gt_path = stage_dir / "PROMISE12_GT_MASKS_PACKBITS.npy"
    meta_path = stage_dir / "PROMISE12_GT_REVEAL_META.json"

    gt_packed = np.lib.format.open_memmap(
        gt_path,
        mode="w+",
        dtype=np.uint8,
        shape=(N_SLICES, PACKED_BYTES),
    )

    global_index = 0
    case_rows = []

    image_vol_map = {
        str(r.case_key): r
        for r in image_volumes.itertuples(index=False)
    }

    with zipfile.ZipFile(PROMISE_ZIP, "r") as zf:
        for row in tqdm(
            list(gt_manifest.itertuples(index=False)),
            desc="PROMISE12 GT reveal/preprocess",
            unit="patient",
            dynamic_ncols=True,
        ):
            raw = zf.read(str(row.raw_member))

            x = int(row.dim_x)
            y = int(row.dim_y)
            z = int(row.dim_z)

            expected_bytes = x * y * z
            if len(raw) != expected_bytes:
                raise RuntimeError(
                    f"{row.case_key} GT RAW bytes "
                    f"{len(raw)} != {expected_bytes}"
                )

            dtype = (
                np.int8
                if str(row.element_type).upper() == "MET_CHAR"
                else np.uint8
            )
            arr = np.frombuffer(raw, dtype=dtype).reshape(
                z, y, x
            )
            binary = (arr > 0).astype(np.float32)

            image_row = image_vol_map[str(row.case_key)]
            out_h = int(
                round(
                    y
                    * float(image_row.spacing_y)
                    / float(
                        load_json(
                            Path(
                                load_json(CM5A_LOCK)[
                                    "artifacts"
                                ]["target_cache_meta"]
                            )
                        )["target_spacing_xy"]
                    )
                )
            )
            out_w = int(
                round(
                    x
                    * float(image_row.spacing_x)
                    / float(
                        load_json(
                            Path(
                                load_json(CM5A_LOCK)[
                                    "artifacts"
                                ]["target_cache_meta"]
                            )
                        )["target_spacing_xy"]
                    )
                )
            )

            t = torch.from_numpy(binary).unsqueeze(1)
            t = torch.nn.functional.interpolate(
                t,
                size=(out_h, out_w),
                mode="nearest",
            )
            t = center_crop_pad_batch(t, torch)
            proc = (t[:, 0].numpy() > 0.5)

            if proc.shape != (z, SEG_SIZE, SEG_SIZE):
                raise RuntimeError(
                    f"{row.case_key} GT processed shape changed: "
                    f"{proc.shape}"
                )

            expected_case_slices = target_slices[
                target_slices["case_key"].astype(str)
                == str(row.case_key)
            ]
            if len(expected_case_slices) != z:
                raise RuntimeError(
                    f"{row.case_key} target/GT z mismatch."
                )

            expected_gi = expected_case_slices[
                "global_index"
            ].to_numpy(dtype=np.int64)
            if not np.array_equal(
                expected_gi,
                np.arange(global_index, global_index + z),
            ):
                raise RuntimeError(
                    f"{row.case_key} target slice ordering changed."
                )

            gt_packed[
                global_index:global_index + z
            ] = pack_masks(proc)

            case_rows.append({
                "case_key": str(row.case_key),
                "slices": z,
                "positive_slices": int(
                    np.sum(proc.reshape(z, -1).any(axis=1))
                ),
                "foreground_voxels": int(proc.sum()),
            })

            global_index += z

    gt_packed.flush()

    if global_index != N_SLICES:
        raise RuntimeError(
            f"GT cache slices {global_index} != {N_SLICES}"
        )

    case_df = pd.DataFrame(case_rows)
    case_path = stage_dir / "PROMISE12_GT_CASE_AUDIT.csv"
    case_df.to_csv(case_path, index=False)

    meta = {
        "status": "PASS",
        "version": VERSION,
        "first_gt_reveal_stage": "CM6",
        "patients": N_PATIENTS,
        "slices": N_SLICES,
        "gt_file": str(gt_path),
        "gt_sha256": sha256_file(gt_path),
        "case_audit": str(case_path),
        "case_audit_sha256": sha256_file(case_path),
        "label_rule": "segmentation > 0",
        "resampling": (
            "same image-derived in-plane spacing transform as CM5A; "
            "nearest-neighbor; no z resampling; center crop/pad 352"
        ),
        "post_gt_threshold_changes_allowed": False,
    }
    save_json(meta_path, meta)

    return np.load(gt_path, mmap_mode="r"), meta


def load_locked_target_assets(cm5a, cm5b):
    score_path = Path(cm5a["artifacts"]["target_scores"])
    if sha256_file(score_path) != cm5a["artifacts"]["target_scores_sha256"]:
        raise RuntimeError("CM5A target-score SHA mismatch.")

    scores = pd.read_csv(score_path)

    if len(scores) != len(FAMILIES) * N_SLICES:
        raise RuntimeError("Unexpected target score rows.")

    forbidden = {
        "harmful",
        "beneficial",
        "delta_dice",
        "source_dice",
        "tent1_dice",
    }
    if forbidden.intersection(scores.columns):
        raise RuntimeError(
            "CM5A score asset unexpectedly already contains GT outcomes."
        )

    pred_sidecars = cm5a["artifacts"]["family_predictions"]

    masks = {}
    for family in FAMILIES:
        side = pred_sidecars[family]
        src_path = Path(side["source_masks"])
        tent_path = Path(side["tent1_masks"])

        if sha256_file(src_path) != side["source_masks_sha256"]:
            raise RuntimeError(
                f"{family} SOURCE target-mask SHA mismatch."
            )
        if sha256_file(tent_path) != side["tent1_masks_sha256"]:
            raise RuntimeError(
                f"{family} TENT target-mask SHA mismatch."
            )

        src = np.load(src_path, mmap_mode="r")
        tent = np.load(tent_path, mmap_mode="r")

        if src.shape != (N_SLICES, PACKED_BYTES):
            raise RuntimeError(
                f"{family} SOURCE packed-mask shape changed."
            )
        if tent.shape != (N_SLICES, PACKED_BYTES):
            raise RuntimeError(
                f"{family} TENT packed-mask shape changed."
            )

        masks[family] = {
            "source": src,
            "tent": tent,
        }

    fixed_tau = float(
        cm5b["fixed_source_reference"]["threshold"]
    )
    naive_tau = float(
        cm5b["naive_target_quantile_baseline"]["threshold"]
    )

    if cm5b["support_aware_transport"]["status"] != "ABSTAIN":
        raise RuntimeError(
            "CM5B support-aware policy was not prospectively ABSTAIN."
        )

    return scores, masks, fixed_tau, naive_tau


def dice_binary(pred, gt):
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())

    den = 2 * tp + fp + fn
    dice = 1.0 if den == 0 else (2.0 * tp) / den

    return float(dice), tp, fp, fn


def build_outcome_table(
    target_slices,
    scores,
    masks,
    gt_packed,
):
    rows = []

    score_lookup = {
        family: (
            scores[scores["family"] == family]
            .sort_values("global_index")
            .reset_index(drop=True)
        )
        for family in FAMILIES
    }

    for family in FAMILIES:
        sdf = score_lookup[family]

        if not np.array_equal(
            sdf["global_index"].to_numpy(dtype=np.int64),
            np.arange(N_SLICES, dtype=np.int64),
        ):
            raise RuntimeError(
                f"{family} score global_index ordering changed."
            )

        for start in tqdm(
            range(0, N_SLICES, 16),
            total=(N_SLICES + 15) // 16,
            desc=f"{family} GT outcomes",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + 16, N_SLICES)

            gt = unpack_masks(gt_packed[start:end])
            src = unpack_masks(
                masks[family]["source"][start:end]
            )
            tent = unpack_masks(
                masks[family]["tent"][start:end]
            )

            for local, gi in enumerate(range(start, end)):
                sd, stp, sfp, sfn = dice_binary(
                    src[local], gt[local]
                )
                td, ttp, tfp, tfn = dice_binary(
                    tent[local], gt[local]
                )
                delta = td - sd

                if delta <= HARM_THRESHOLD:
                    state = "HARM"
                    harmful = 1
                    beneficial = 0
                elif delta >= BENEFIT_THRESHOLD:
                    state = "BENEFIT"
                    harmful = 0
                    beneficial = 1
                else:
                    state = "NEUTRAL"
                    harmful = 0
                    beneficial = 0

                sr = sdf.iloc[gi]
                tr = target_slices.iloc[gi]

                rows.append({
                    "family": family,
                    "global_index": gi,
                    "case_key": str(tr["case_key"]),
                    "slice_index": int(tr["slice_index"]),
                    "risk_score": float(sr["risk_score"]),
                    "source_dice": sd,
                    "tent1_dice": td,
                    "delta_dice": delta,
                    "state": state,
                    "harmful": harmful,
                    "beneficial": beneficial,
                    "source_tp": stp,
                    "source_fp": sfp,
                    "source_fn": sfn,
                    "tent_tp": ttp,
                    "tent_fp": tfp,
                    "tent_fn": tfn,
                })

    df = pd.DataFrame(rows)

    if len(df) != len(FAMILIES) * N_SLICES:
        raise RuntimeError("Outcome-table row count changed.")

    return df


def ranking_metrics(outcomes):
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    rows = []

    for family in FAMILIES:
        sub = outcomes[outcomes["family"] == family]
        y = sub["harmful"].to_numpy(dtype=np.int64)
        s = sub["risk_score"].to_numpy(dtype=np.float64)

        if len(np.unique(y)) != 2:
            raise RuntimeError(
                f"{family} target HARM has only one class."
            )

        rows.append({
            "scope": family,
            "rows": len(sub),
            "patients": sub["case_key"].nunique(),
            "harm_count": int(y.sum()),
            "harm_prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "spearman_score_vs_minus_delta": float(
                spearmanr(
                    s,
                    -sub["delta_dice"].to_numpy(dtype=np.float64),
                ).statistic
            ),
        })

    family_df = pd.DataFrame(rows)

    y = outcomes["harmful"].to_numpy(dtype=np.int64)
    s = outcomes["risk_score"].to_numpy(dtype=np.float64)

    pooled = {
        "scope": "POOLED",
        "rows": len(outcomes),
        "patients": outcomes["case_key"].nunique(),
        "harm_count": int(y.sum()),
        "harm_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "spearman_score_vs_minus_delta": float(
            spearmanr(
                s,
                -outcomes["delta_dice"].to_numpy(dtype=np.float64),
            ).statistic
        ),
    }

    macro = {
        "scope": "MACRO_FAMILY",
        "rows": len(outcomes),
        "patients": outcomes["case_key"].nunique(),
        "harm_count": int(y.sum()),
        "harm_prevalence": float(y.mean()),
        "auroc": float(family_df["auroc"].mean()),
        "auprc": float(family_df["auprc"].mean()),
        "spearman_score_vs_minus_delta": float("nan"),
    }

    return pd.concat(
        [
            family_df,
            pd.DataFrame([macro, pooled]),
        ],
        ignore_index=True,
    )


def patient_cluster_bootstrap_macro_auc(outcomes):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.asarray(
        sorted(outcomes["case_key"].astype(str).unique())
    )

    if len(patients) != N_PATIENTS:
        raise RuntimeError("Unexpected target bootstrap patient count.")

    case = outcomes["case_key"].astype(str).to_numpy()
    family = outcomes["family"].astype(str).to_numpy()
    y = outcomes["harmful"].to_numpy(dtype=np.int64)
    s = outcomes["risk_score"].to_numpy(dtype=np.float64)

    patient_to_rows = {
        p: np.flatnonzero(case == p)
        for p in patients
    }

    vals = []

    for _ in tqdm(
        range(BOOTSTRAP_REPS),
        desc="PROMISE patient bootstrap macro AUROC",
        unit="rep",
        dynamic_ncols=True,
    ):
        sampled = rng.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        idx = np.concatenate(
            [patient_to_rows[p] for p in sampled]
        )

        aucs = []
        valid = True

        for fam in FAMILIES:
            m = family[idx] == fam
            yy = y[idx][m]
            ss = s[idx][m]

            if len(np.unique(yy)) != 2:
                valid = False
                break

            aucs.append(
                float(roc_auc_score(yy, ss))
            )

        if valid:
            vals.append(float(np.mean(aucs)))

    vals = np.asarray(vals, dtype=np.float64)

    if len(vals) < int(0.95 * BOOTSTRAP_REPS):
        raise RuntimeError(
            f"Too many invalid target bootstrap replicates: "
            f"{len(vals)}/{BOOTSTRAP_REPS}"
        )

    return {
        "reps_requested": BOOTSTRAP_REPS,
        "reps_valid": int(len(vals)),
        "seed": BOOTSTRAP_SEED,
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


def largest_threshold_recall90(scores, harm):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(harm, dtype=np.int64)

    candidates = np.unique(s)
    best = None

    for tau in candidates:
        high = s >= tau
        tp = int(np.sum((y == 1) & high))
        fn = int(np.sum((y == 1) & (~high)))
        recall = tp / (tp + fn)

        if recall + 1e-15 >= ORACLE_RECALL_TARGET:
            best = float(tau)

    if best is None:
        raise RuntimeError("Unable to find oracle recall90 threshold.")

    return best


def policy_row_metrics(sub, tau):
    y = sub["harmful"].to_numpy(dtype=np.int64)
    s = sub["risk_score"].to_numpy(dtype=np.float64)

    retain = s >= tau
    adapt = ~retain

    tp = int(np.sum((y == 1) & retain))
    fn = int(np.sum((y == 1) & adapt))
    fp = int(np.sum((y == 0) & retain))
    tn = int(np.sum((y == 0) & adapt))

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    deployed_slice_dice = np.where(
        adapt,
        sub["tent1_dice"].to_numpy(dtype=np.float64),
        sub["source_dice"].to_numpy(dtype=np.float64),
    )

    return {
        "threshold": float(tau),
        "tp_harm": tp,
        "fn_harm": fn,
        "fp_nonharm": fp,
        "tn_nonharm": tn,
        "harm_recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "adaptation_coverage": float(np.mean(adapt)),
        "prevented_harm_fraction": float(recall),
        "mean_slice_source_dice": float(
            sub["source_dice"].mean()
        ),
        "mean_slice_tent1_dice": float(
            sub["tent1_dice"].mean()
        ),
        "mean_slice_deployed_dice": float(
            deployed_slice_dice.mean()
        ),
    }


def patient_3d_policy_dice(sub, tau):
    rows = []

    for (family, case_key), g in sub.groupby(
        ["family", "case_key"],
        sort=False,
    ):
        s = g["risk_score"].to_numpy(dtype=np.float64)
        adapt = s < tau

        source_tp = g["source_tp"].to_numpy(dtype=np.int64)
        source_fp = g["source_fp"].to_numpy(dtype=np.int64)
        source_fn = g["source_fn"].to_numpy(dtype=np.int64)

        tent_tp = g["tent_tp"].to_numpy(dtype=np.int64)
        tent_fp = g["tent_fp"].to_numpy(dtype=np.int64)
        tent_fn = g["tent_fn"].to_numpy(dtype=np.int64)

        dep_tp = np.where(adapt, tent_tp, source_tp)
        dep_fp = np.where(adapt, tent_fp, source_fp)
        dep_fn = np.where(adapt, tent_fn, source_fn)

        def pooled_dice(tp_arr, fp_arr, fn_arr):
            tp = int(tp_arr.sum())
            fp = int(fp_arr.sum())
            fn = int(fn_arr.sum())
            den = 2 * tp + fp + fn
            return 1.0 if den == 0 else (2.0 * tp) / den

        rows.append({
            "family": family,
            "case_key": str(case_key),
            "source_dice_3d": pooled_dice(
                source_tp, source_fp, source_fn
            ),
            "tent1_dice_3d": pooled_dice(
                tent_tp, tent_fp, tent_fn
            ),
            "deployed_dice_3d": pooled_dice(
                dep_tp, dep_fp, dep_fn
            ),
            "adaptation_coverage": float(np.mean(adapt)),
        })

    return pd.DataFrame(rows)


def evaluate_policy(outcomes, policy_name, tau):
    rows = []

    for scope in FAMILIES + ["POOLED"]:
        sub = (
            outcomes
            if scope == "POOLED"
            else outcomes[outcomes["family"] == scope]
        )

        m = policy_row_metrics(sub, tau)

        patient_df = patient_3d_policy_dice(sub, tau)

        if scope == "POOLED":
            patient_mean_source = float(
                patient_df["source_dice_3d"].mean()
            )
            patient_mean_tent = float(
                patient_df["tent1_dice_3d"].mean()
            )
            patient_mean_deployed = float(
                patient_df["deployed_dice_3d"].mean()
            )
        else:
            pf = patient_df[patient_df["family"] == scope]
            patient_mean_source = float(
                pf["source_dice_3d"].mean()
            )
            patient_mean_tent = float(
                pf["tent1_dice_3d"].mean()
            )
            patient_mean_deployed = float(
                pf["deployed_dice_3d"].mean()
            )

        rows.append({
            "policy": policy_name,
            "scope": scope,
            **m,
            "mean_patient_3d_source_dice": patient_mean_source,
            "mean_patient_3d_tent1_dice": patient_mean_tent,
            "mean_patient_3d_deployed_dice": patient_mean_deployed,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1X CM6 PROMISE12 FIRST GT REVEAL + LOCKED POLICY EVALUATION ====="
    )
    print("PROMISE12_GT_ACCESS=YES_FIRST_TIME")
    print("POST_GT_THRESHOLD_CHANGES=NO")
    print("POST_GT_MODEL_CHANGES=NO")
    print("POST_GT_REPRESENTATION_CHANGES=NO")
    print("HARM_THRESHOLD=", HARM_THRESHOLD)
    print("BENEFIT_THRESHOLD=", BENEFIT_THRESHOLD)
    print("PRIMARY_RANKING=macro-family AUROC")
    print("PATIENT_BOOTSTRAP_REPS=", BOOTSTRAP_REPS)
    print("SUPPORT_AWARE_POLICY=LOCKED_ABSTAIN")

    cm5a, cm5b = verify_lineage()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    target_slices, image_volumes, cache_meta = (
        load_cm5a_target_slice_manifest(cm5a)
    )

    gt_manifest = build_segmentation_member_manifest()
    audit_gt_geometry(gt_manifest, image_volumes)

    import torch

    gt_packed, gt_meta = build_gt_cache(
        gt_manifest,
        image_volumes,
        target_slices,
        args.output_dir,
        torch,
    )

    scores, masks, fixed_tau, naive_tau = (
        load_locked_target_assets(cm5a, cm5b)
    )

    outcomes = build_outcome_table(
        target_slices,
        scores,
        masks,
        gt_packed,
    )

    outcome_path = args.output_dir / "CM6_PROMISE12_OUTCOMES.csv"
    outcomes.to_csv(outcome_path, index=False)

    print("\n===== TARGET HARM PREVALENCE =====")
    harm_summary = (
        outcomes.groupby("family", as_index=False)
        .agg(
            rows=("harmful", "size"),
            patients=("case_key", "nunique"),
            harm_count=("harmful", "sum"),
            harm_prevalence=("harmful", "mean"),
            benefit_count=("beneficial", "sum"),
            mean_source_dice=("source_dice", "mean"),
            mean_tent1_dice=("tent1_dice", "mean"),
            mean_delta_dice=("delta_dice", "mean"),
            median_delta_dice=("delta_dice", "median"),
        )
    )
    print(harm_summary.to_string(index=False))

    ranking = ranking_metrics(outcomes)
    ranking_path = args.output_dir / "CM6_TARGET_HARM_RANKING.csv"
    ranking.to_csv(ranking_path, index=False)

    print("\n===== TARGET HARM RANKING =====")
    print(ranking.to_string(index=False))

    bootstrap = patient_cluster_bootstrap_macro_auc(outcomes)
    bootstrap_path = (
        args.output_dir
        / "CM6_MACRO_FAMILY_AUROC_PATIENT_BOOTSTRAP.json"
    )
    save_json(bootstrap_path, bootstrap)

    print("\n===== TARGET MACRO AUROC BOOTSTRAP =====")
    print(json.dumps(bootstrap, indent=2))

    oracle_tau = largest_threshold_recall90(
        outcomes["risk_score"].to_numpy(dtype=np.float64),
        outcomes["harmful"].to_numpy(dtype=np.int64),
    )

    policy_tables = [
        evaluate_policy(
            outcomes,
            "FIXED_SOURCE_THRESHOLD",
            fixed_tau,
        ),
        evaluate_policy(
            outcomes,
            "NAIVE_UNLABELED_TARGET_QUANTILE",
            naive_tau,
        ),
        evaluate_policy(
            outcomes,
            "ORACLE_TARGET_RECALL90_REFERENCE_ONLY",
            oracle_tau,
        ),
    ]

    policy_df = pd.concat(
        policy_tables,
        ignore_index=True,
    )
    policy_path = args.output_dir / "CM6_LOCKED_POLICY_EVALUATION.csv"
    policy_df.to_csv(policy_path, index=False)

    print("\n===== LOCKED POLICY EVALUATION =====")
    print(policy_df.to_string(index=False))
    print(
        "\nSUPPORT_AWARE_TRANSPORT: ABSTAIN "
        "(locked before GT; no threshold evaluated)"
    )

    macro_auc = float(
        ranking.loc[
            ranking["scope"] == "MACRO_FAMILY",
            "auroc",
        ].iloc[0]
    )

    ranking_confirmed = bootstrap["ci95_low"] > 0.5
    decision = (
        PASS_RANKING_DECISION
        if ranking_confirmed
        else FAIL_RANKING_DECISION
    )

    fixed_pooled = policy_df[
        (policy_df["policy"] == "FIXED_SOURCE_THRESHOLD")
        & (policy_df["scope"] == "POOLED")
    ].iloc[0]
    naive_pooled = policy_df[
        (policy_df["policy"] == "NAIVE_UNLABELED_TARGET_QUANTILE")
        & (policy_df["scope"] == "POOLED")
    ].iloc[0]
    oracle_pooled = policy_df[
        (policy_df["policy"] == "ORACLE_TARGET_RECALL90_REFERENCE_ONLY")
        & (policy_df["scope"] == "POOLED")
    ].iloc[0]

    lock = {
        "status": "PASS",
        "decision": decision,
        "version": VERSION,
        "build": BUILD,
        "cm5a_lock_sha256": EXPECTED_CM5A_LOCK_SHA,
        "cm5b_lock_sha256": EXPECTED_CM5B_LOCK_SHA,
        "cm5b_helper_sha256": EXPECTED_CM5B_HELPER_SHA,
        "first_gt_reveal": True,
        "post_gt_changes": {
            "thresholds": False,
            "models": False,
            "representation": False,
        },
        "target": {
            "patients": N_PATIENTS,
            "slices": N_SLICES,
            "family_slice_rows": int(len(outcomes)),
        },
        "harm_definition": "TENT1 Dice - SOURCE Dice <= -0.02",
        "ranking": {
            "primary_metric": "macro-family AUROC",
            "point_estimate": macro_auc,
            "patient_cluster_bootstrap": bootstrap,
            "confirmed": bool(ranking_confirmed),
            "criterion": "95% CI lower bound > 0.50",
        },
        "operating_point_results": {
            "support_aware_transport": {
                "status": "ABSTAIN",
                "reason": cm5b["support_aware_transport"]["reason"],
                "prospectively_locked_before_gt": True,
            },
            "fixed_source_pooled": fixed_pooled.to_dict(),
            "naive_unlabeled_quantile_pooled": naive_pooled.to_dict(),
            "oracle_reference_pooled": oracle_pooled.to_dict(),
            "fixed_source_threshold": fixed_tau,
            "naive_quantile_threshold": naive_tau,
            "oracle_target_recall90_threshold": oracle_tau,
        },
        "interpretation_boundary": (
            "Support-aware unlabeled transport abstained because pre-GT "
            "representation support was inadequate. Therefore no claim of "
            "successful unlabeled operating-point transport is permitted. "
            "Ranking confirmation is evaluated independently."
        ),
        "artifacts": {
            "gt_reveal_meta": str(
                args.output_dir / "PROMISE12_GT_REVEAL_META.json"
            ),
            "gt_reveal_meta_sha256": sha256_file(
                args.output_dir / "PROMISE12_GT_REVEAL_META.json"
            ),
            "outcomes": str(outcome_path),
            "outcomes_sha256": sha256_file(outcome_path),
            "ranking": str(ranking_path),
            "ranking_sha256": sha256_file(ranking_path),
            "bootstrap": str(bootstrap_path),
            "bootstrap_sha256": sha256_file(bootstrap_path),
            "policy_evaluation": str(policy_path),
            "policy_evaluation_sha256": sha256_file(policy_path),
        },
    }

    lock_path = (
        args.output_dir
        / "CM6_PROMISE12_GT_REVEAL_LOCKED_EVALUATION_LOCK.json"
    )
    save_json(lock_path, lock)

    print("\n===== CM6 FINAL =====")
    print("Primary macro-family AUROC:", macro_auc)
    print(
        "Patient-cluster bootstrap 95% CI:",
        f"[{bootstrap['ci95_low']:.6f}, "
        f"{bootstrap['ci95_high']:.6f}]",
    )
    print(
        "Fixed SOURCE pooled HARM recall/FPR/coverage/deployed3D:",
        float(fixed_pooled["harm_recall"]),
        float(fixed_pooled["fpr"]),
        float(fixed_pooled["adaptation_coverage"]),
        float(fixed_pooled["mean_patient_3d_deployed_dice"]),
    )
    print(
        "Naive quantile pooled HARM recall/FPR/coverage/deployed3D:",
        float(naive_pooled["harm_recall"]),
        float(naive_pooled["fpr"]),
        float(naive_pooled["adaptation_coverage"]),
        float(naive_pooled["mean_patient_3d_deployed_dice"]),
    )
    print(
        "Oracle reference HARM recall/FPR/coverage/deployed3D:",
        float(oracle_pooled["harm_recall"]),
        float(oracle_pooled["fpr"]),
        float(oracle_pooled["adaptation_coverage"]),
        float(oracle_pooled["mean_patient_3d_deployed_dice"]),
    )
    print("Support-aware transport: ABSTAIN (prospectively locked)")
    print("Post-GT threshold changes: NO")
    print("Decision=", decision)
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
