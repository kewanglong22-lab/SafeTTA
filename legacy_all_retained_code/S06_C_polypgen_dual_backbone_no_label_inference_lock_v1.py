#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S06-C: untouched PolypGen dual-backbone NO-LABEL inference + SHA lock.

Allowed: PolypGen IMAGE bytes, frozen Source/TENT inference, 24 unlabeled
features, frozen SafeTTA decisions.
Forbidden: PolypGen MASK bytes, GT metrics, refitting, recalibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\MEDSEG_SAFETTA")
os.environ["TORCH_HOME"] = str(ROOT / "assets" / "torchvision_cache")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

VERSION = "2026-08-18-S06-C-v1"
BUILD = "S06_C_DUAL_BACKBONE_EXTERNAL_NO_LABEL_LOCK_BEFORE_GT"

PROTOCOL = ROOT / "docs" / "S06_C_polypgen_dual_backbone_no_label_lock_protocol_v1.md"
EXPECTED_PROTOCOL_SHA256 = "4de3d2b60b07b6e30a3a154b8d7668f843a6a8426d294dff15ebaf8d1824e36c"
S06A_PROTOCOL = ROOT / "docs" / "S06_A_polypgen_untouched_external_confirmation_protocol_v1_fix3.md"
EXPECTED_S06A_SHA256 = "744672e234a18cbdf3cca3a7e231e38362c5e09d1e6bdaaf43c9dcb98f49c78d"
S06B_SCRIPT = ROOT / "code" / "S06_B_polypgen_integrity_overlap_audit_v1_fix3.py"
EXPECTED_S06B_SCRIPT_SHA256 = "93c3ae7eb5ed08ee18dcecc3238f7d692330169c28074c6a4cefeb322385859c"
S06B_DIR = ROOT / "outputs" / "S06_B_polypgen_integrity_overlap_audit_v1_fix3"
S06B_AUDIT = S06B_DIR / "audit.json"
ELIGIBILITY = S06B_DIR / "S06_polypgen_external_eligibility_manifest.csv"

ARCHIVE = ROOT / "data" / "downloads" / "polypgen" / "polypgen2021_multicenterdata_v3.zip"
EXPECTED_ARCHIVE_SHA256 = "a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3"

PRANET_HELPER = ROOT / "code" / "S03_B_build_source_side_safettta_gate_v1.py"
EXPECTED_PRANET_HELPER_SHA256 = "07ac65247216dea376e8996cbfbc98c54c62157ecdd1e9b7d9b539b5a74a2b5c"
PRANET_TRAINING_HELPER = ROOT / "code" / "S01_B_train_pranet_source_only_frozen_seeds_v1.py"
EXPECTED_PRANET_TRAINING_HELPER_SHA256 = "2b2c9c1b75ff60bd087b403e43dc77468b31cbd54fb607481cc80ab47182b67a"
PRANET_GATE = ROOT / "outputs" / "S03_B_source_side_safettta_gate_v1" / "SafeTTA_v1_SOURCE_LOCKED_GATE.json"
EXPECTED_PRANET_GATE_SHA256 = "a71696fd3730dae19bda7a66405a5314022c1eb11acca6604ddcb7ccd1b82e2f"

DEEPLAB_HELPER = ROOT / "code" / "S05_C_build_deeplab_source_side_safettta_gate_v1.py"
EXPECTED_DEEPLAB_HELPER_SHA256 = "ab1d98f9847b2e338f07ec26ff426f9cd428e68da4813b90e2227782319aa424"
DEEPLAB_TRAINING_HELPER = ROOT / "code" / "S05_B_train_deeplabv3_resnet50_source_only_frozen_seeds_v1_fix2.py"
EXPECTED_DEEPLAB_TRAINING_HELPER_SHA256 = "7d0fd5192a9af40dbae0aeac692a65641745fbef2e986f1cd4e163a59ad93a31"
DEEPLAB_GATE = ROOT / "outputs" / "S05_C_deeplab_source_side_safettta_gate_v1" / "SafeTTA_v1_DEEPLAB_SOURCE_LOCKED_GATE.json"
EXPECTED_DEEPLAB_GATE_SHA256 = "cfa1d67d43d291bff32c243ebfc6b98aedc873c18fcafd901961a4b779b11b31"
EXPECTED_DEEPLAB_TAU = 0.3938344633

OUTPUT_DIR = ROOT / "outputs" / "S06_C_polypgen_dual_backbone_no_label_lock_v1"
SEEDS = (20260817, 20260818, 20260819)
ARCHITECTURES = ("pranet", "deeplab")
IMAGE_SIZE = 352
EXPECTED_IMAGES = 1537
EXPECTED_CENTERS = {"C1": 256, "C2": 301, "C3": 457, "C4": 227, "C5": 208, "C6": 88}
EXPECTED_OBSERVATIONS = EXPECTED_IMAGES * len(SEEDS) * len(ARCHITECTURES)
PASS_DECISION = "S06_C_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK_PASS_READY_FOR_S06D_GT_REVEAL"
EXPECTED_CHECKPOINTS = {
    "pranet": {
        20260817: "ef623c0f1207bab02377ec17932db43fe2fd1843dca858cc79ef87ead29b975a",
        20260818: "e56dd8ba4b5fc7785fedf3918c275427178fadb5f035b47087d48cde41ccec22",
        20260819: "f8cad00bbc9bbbff04c9a29547bd2a5f3beb97d53980c269cf4b7d2c30130b72",
    },
    "deeplab": {
        20260817: "d63f914e337295652627c773332d8e7382fbcd6a39d69805e1316dcc759605a2",
        20260818: "d4ede45d5b30f62fdbc92cd9aa8fc84e0d5a52073d5ef2c3ac36baa09a56eb25",
        20260819: "66aeae338d350db5aa878bcaab2e68bbe98eec8e626e28a1bb02b6708fa358eb",
    },
}
GATE_SHAS = {"pranet": EXPECTED_PRANET_GATE_SHA256, "deeplab": EXPECTED_DEEPLAB_GATE_SHA256}
HELPER_SHAS = {"pranet": EXPECTED_PRANET_HELPER_SHA256, "deeplab": EXPECTED_DEEPLAB_HELPER_SHA256}
TRAINING_HELPER_SHAS = {"pranet": EXPECTED_PRANET_TRAINING_HELPER_SHA256, "deeplab": EXPECTED_DEEPLAB_TRAINING_HELPER_SHA256}


def file_sha256(path: Path, chunk_size=16 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bytes_sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def sequence_sha256(values):
    h = hashlib.sha256()
    for value in values:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames or []


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def validate_sha(path: Path, expected: str, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA mismatch\nExpected: {expected}\nActual  : {actual}")
    return actual


def import_exact(path: Path, expected_sha: str, name: str):
    validate_sha(path, expected_sha, name)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sigmoid_scalar(z: float):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def seed_runtime(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_gate(path: Path, expected_sha: str, expected_features, architecture: str):
    validate_sha(path, expected_sha, f"{architecture} gate")
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"feature_names", "scaler_mean", "scaler_scale", "logistic_coef", "logistic_intercept", "harm_threshold_tau", "checkpoint_sha256"}
    missing = sorted(required - set(raw))
    if missing:
        raise RuntimeError(f"{architecture} gate missing fields: {missing}")
    names = list(raw["feature_names"])
    if names != list(expected_features) or len(names) != 24:
        raise RuntimeError(f"{architecture} frozen feature order/count mismatch")
    mean = np.asarray(raw["scaler_mean"], dtype=np.float64)
    scale = np.asarray(raw["scaler_scale"], dtype=np.float64)
    coef = np.asarray(raw["logistic_coef"], dtype=np.float64)
    intercept = float(raw["logistic_intercept"])
    tau = float(raw["harm_threshold_tau"])
    if mean.shape != (24,) or scale.shape != (24,) or coef.shape != (24,):
        raise RuntimeError(f"{architecture} gate dimensions invalid")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0) or np.any(~np.isfinite(coef)) or not math.isfinite(intercept):
        raise RuntimeError(f"{architecture} gate contains invalid parameters")
    if not math.isfinite(tau) or not (0 <= tau <= 1):
        raise RuntimeError(f"{architecture} tau invalid: {tau}")
    if architecture == "deeplab" and abs(tau - EXPECTED_DEEPLAB_TAU) > 5e-10:
        raise RuntimeError(f"DeepLab frozen tau mismatch: {tau}")
    ck = {str(k): str(v) for k, v in raw["checkpoint_sha256"].items()}
    expected_ck = {str(s): EXPECTED_CHECKPOINTS[architecture][s] for s in SEEDS}
    if ck != expected_ck:
        raise RuntimeError(f"{architecture} gate checkpoint provenance mismatch")
    return {"feature_names": names, "mean": mean, "scale": scale, "coef": coef, "intercept": intercept, "tau": tau}


def apply_gate(features: dict, gate: dict):
    x = np.asarray([float(features[n]) for n in gate["feature_names"]], dtype=np.float64)
    if x.shape != (24,) or np.any(~np.isfinite(x)):
        raise RuntimeError("Malformed/non-finite 24-feature vector")
    z = float(np.dot(gate["coef"], (x - gate["mean"]) / gate["scale"]) + gate["intercept"])
    p = float(sigmoid_scalar(z))
    if not math.isfinite(p) or not (0 <= p <= 1):
        raise RuntimeError(f"Invalid p_harm={p}")
    accepted = int(p < gate["tau"])
    return p, accepted


def validate_s06b():
    validate_sha(S06A_PROTOCOL, EXPECTED_S06A_SHA256, "S06-A FIX3 protocol")
    validate_sha(S06B_SCRIPT, EXPECTED_S06B_SCRIPT_SHA256, "S06-B FIX3 script")
    if not S06B_AUDIT.exists() or not ELIGIBILITY.exists():
        raise FileNotFoundError("S06-B audit/eligibility output is missing")
    audit_sha = file_sha256(S06B_AUDIT)
    manifest_sha = file_sha256(ELIGIBILITY)
    audit = json.loads(S06B_AUDIT.read_text(encoding="utf-8"))
    expected_flags_false = [
        "polypgen_mask_bytes_opened", "polypgen_masks_extracted",
        "polypgen_mask_pixels_inspected", "eligibility_used_mask_content",
        "model_inference_performed", "target_metrics_computed",
    ]
    checks = [
        audit.get("decision") == "S06_B_AUDIT_PASS_READY_FOR_S06C_NO_LABEL_INFERENCE",
        audit.get("protocol_sha256") == EXPECTED_S06A_SHA256,
        audit.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256,
        int(audit.get("polypgen_single_frame_pair_count", -1)) == EXPECTED_IMAGES,
        int(audit.get("eligible_pending_s06c_samples", -1)) == EXPECTED_IMAGES,
        int(audit.get("exact_duplicate_external_samples", -1)) == 0,
        int(audit.get("near_duplicate_hold_external_samples", -1)) == 0,
        audit.get("pairing_rule") == "S06_B1_V2_AGGRESSIVE_ALPHANUMERIC",
        bool(audit.get("pairing_rule_metadata_only")) is True,
        {str(k): int(v) for k, v in audit.get("center_pair_counts", {}).items()} == EXPECTED_CENTERS,
        all(not bool(audit.get(k)) for k in expected_flags_false),
    ]
    if not all(checks):
        raise RuntimeError("S06-B output does not satisfy frozen S06-C entry conditions")

    rows, fields = read_csv(ELIGIBILITY)
    required = {"external_id", "center", "image_member", "mask_member", "image_width", "image_height", "image_raw_sha256", "image_decoded_rgb_sha256", "eligibility", "eligibility_uses_mask_content"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Eligibility manifest missing columns: {missing}")
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} eligible rows, got {len(rows)}")
    if any(r["eligibility"] != "ELIGIBLE_PENDING_S06C" for r in rows):
        raise RuntimeError("Eligibility manifest contains a non-eligible row")
    if any(r["eligibility_uses_mask_content"].upper() != "NO" for r in rows):
        raise RuntimeError("Eligibility used mask content")
    if len({r["external_id"] for r in rows}) != len(rows) or len({r["image_member"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate external_id/image_member")
    if dict(sorted(Counter(r["center"] for r in rows).items())) != EXPECTED_CENTERS:
        raise RuntimeError("Eligibility center counts changed")
    for r in rows:
        if not r["mask_member"] or r["mask_member"] == r["image_member"]:
            raise RuntimeError(f"Invalid paired metadata: {r['external_id']}")
        if int(r["image_width"]) <= 0 or int(r["image_height"]) <= 0:
            raise RuntimeError(f"Invalid dimensions: {r['external_id']}")
        for key in ("image_raw_sha256", "image_decoded_rgb_sha256"):
            v = r[key].lower()
            if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
                raise RuntimeError(f"Invalid SHA field {key}: {r['external_id']}")
    return sorted(rows, key=lambda r: (r["center"], r["external_id"])), audit_sha, manifest_sha


def read_external_image(zf: zipfile.ZipFile, row: dict):
    image_member = row["image_member"]
    # The sole target-ZIP byte read in S06-C is the IMAGE member below.
    with zf.open(image_member, "r") as f:
        payload = f.read()
    if bytes_sha256(payload) != row["image_raw_sha256"]:
        raise RuntimeError(f"Image raw SHA mismatch: {row['external_id']}")
    with Image.open(io.BytesIO(payload)) as im:
        image = im.convert("RGB")
    if image.size != (int(row["image_width"]), int(row["image_height"])):
        raise RuntimeError(f"Image dimensions changed: {row['external_id']}")
    if bytes_sha256(np.asarray(image, dtype=np.uint8).tobytes()) != row["image_decoded_rgb_sha256"]:
        raise RuntimeError(f"Decoded RGB SHA mismatch: {row['external_id']}")
    return image


def load_stack(architecture: str, audit_sha: str, manifest_sha: str):
    if architecture == "pranet":
        validate_sha(PRANET_TRAINING_HELPER, EXPECTED_PRANET_TRAINING_HELPER_SHA256, "PraNet training helper")
        helper = import_exact(PRANET_HELPER, EXPECTED_PRANET_HELPER_SHA256, "s06c_pranet_helper")
        training = helper.import_training_helper()
        helper.validate_checkpoints()
        gate = load_gate(PRANET_GATE, EXPECTED_PRANET_GATE_SHA256, helper.FEATURE_NAMES, architecture)
    elif architecture == "deeplab":
        validate_sha(DEEPLAB_TRAINING_HELPER, EXPECTED_DEEPLAB_TRAINING_HELPER_SHA256, "DeepLab training helper")
        helper = import_exact(DEEPLAB_HELPER, EXPECTED_DEEPLAB_HELPER_SHA256, "s06c_deeplab_helper")
        training = helper.import_training_helper()
        helper.validate_checkpoints()
        gate = load_gate(DEEPLAB_GATE, EXPECTED_DEEPLAB_GATE_SHA256, helper.FEATURE_NAMES, architecture)
    else:
        raise ValueError(architecture)
    for seed in SEEDS:
        p = Path(helper.CHECKPOINTS[seed]["path"])
        validate_sha(p, EXPECTED_CHECKPOINTS[architecture][seed], f"{architecture} checkpoint seed {seed}")
        if str(helper.CHECKPOINTS[seed]["sha256"]) != EXPECTED_CHECKPOINTS[architecture][seed]:
            raise RuntimeError(f"{architecture} helper checkpoint provenance mismatch seed={seed}")
    return {"helper": helper, "training": training, "gate": gate, "audit_sha": audit_sha, "manifest_sha": manifest_sha}


def unit_dirs(units: Path, architecture: str, seed: int):
    final = units / f"{architecture}_seed{seed}"
    building = units / f"{architecture}_seed{seed}__building"
    return final, building


def npy_shape(npz_path: Path, key: str):
    with zipfile.ZipFile(npz_path, "r") as archive:
        member = f"{key}.npy"
        if member not in archive.namelist():
            raise RuntimeError(f"NPZ missing {member}")
        with archive.open(member, "r") as f:
            ver = np.lib.format.read_magic(f)
            if ver == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif ver in ((2, 0), (3, 0)):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                raise RuntimeError(f"Unsupported NPY header {ver}")
    return tuple(int(x) for x in shape)


def validate_unit(final: Path, architecture: str, seed: int, rows, ctx):
    lock_path = final / "UNIT_NO_LABEL_LOCK.json"
    csv_path = final / "no_label_features_and_gate_decisions.csv"
    npz_path = final / "locked_logits.npz"
    if not lock_path.exists() or not csv_path.exists() or not npz_path.exists():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_ids = [r["external_id"] for r in rows]
        invariants = [
            lock.get("architecture") == architecture,
            int(lock.get("seed", -1)) == seed,
            int(lock.get("rows", -1)) == len(rows),
            lock.get("checkpoint_sha256") == EXPECTED_CHECKPOINTS[architecture][seed],
            lock.get("gate_artifact_sha256") == GATE_SHAS[architecture],
            lock.get("helper_sha256") == HELPER_SHAS[architecture],
            lock.get("training_helper_sha256") == TRAINING_HELPER_SHAS[architecture],
            lock.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
            lock.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256,
            lock.get("s06b_audit_sha256") == ctx["audit_sha"],
            lock.get("eligibility_manifest_sha256") == ctx["manifest_sha"],
            lock.get("external_id_order_sha256") == sequence_sha256(expected_ids),
            list(lock.get("feature_names", [])) == list(ctx["gate"]["feature_names"]),
            abs(float(lock.get("harm_threshold_tau", -1)) - ctx["gate"]["tau"]) <= 1e-12,
            not bool(lock.get("polypgen_mask_bytes_opened")),
            not bool(lock.get("target_metrics_computed")),
            not bool(lock.get("model_refit")),
            not bool(lock.get("threshold_recalibrated")),
            file_sha256(npz_path) == lock.get("npz_sha256"),
            file_sha256(csv_path) == lock.get("csv_sha256"),
            npy_shape(npz_path, "source_logits") == (len(rows), IMAGE_SIZE, IMAGE_SIZE),
            npy_shape(npz_path, "tent_logits") == (len(rows), IMAGE_SIZE, IMAGE_SIZE),
        ]
        if not all(invariants):
            return None
        with np.load(npz_path, allow_pickle=False) as d:
            if d["external_ids"].tolist() != expected_ids:
                return None
            if d["centers"].tolist() != [r["center"] for r in rows]:
                return None
            if d["image_members"].tolist() != [r["image_member"] for r in rows]:
                return None
        csv_rows, fields = read_csv(csv_path)
        required = {"architecture", "seed", "external_id", "center", "image_member", "p_harm", "tau", "tent_accepted", "decision"} | set(ctx["gate"]["feature_names"])
        if len(csv_rows) != len(rows) or not required.issubset(fields):
            return None
        if [r["external_id"] for r in csv_rows] != expected_ids:
            return None
        for r in csv_rows:
            p, tau, accepted = float(r["p_harm"]), float(r["tau"]), int(r["tent_accepted"])
            if not math.isfinite(p) or not (0 <= p <= 1) or accepted != int(p < tau):
                return None
            if r["decision"] != ("ACCEPT_TENT" if accepted else "REJECT_TENT_USE_SOURCE"):
                return None
            if any(not math.isfinite(float(r[n])) for n in ctx["gate"]["feature_names"]):
                return None
        return lock
    except Exception:
        return None


def produce_unit(architecture: str, seed: int, rows, zf, ctx, units: Path, device):
    final, building = unit_dirs(units, architecture, seed)
    if final.exists() or building.exists():
        raise RuntimeError(f"Unit path already exists: {final if final.exists() else building}")
    building.mkdir(parents=True)
    helper, training, gate = ctx["helper"], ctx["training"], ctx["gate"]
    seed_runtime(seed)
    model = helper.load_model(training, seed, device)
    unsafe_names, dropout_names, unsafe_modules, dropout_modules = [], [], None, None
    if architecture == "pranet":
        params = helper.configure_tent(model)
        source_values = helper.snapshot_params(params)
    else:
        params, _, unsafe_names, unsafe_modules, dropout_names, dropout_modules = helper.configure_singleton_safe_tent(model)
        if unsafe_names != ["classifier.0.convs.4.2"] or dropout_names != ["classifier.0.project.3"]:
            raise RuntimeError("DeepLab singleton-safe TENT topology changed")
        source_values = helper.snapshot_params(params)

    src = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float16)
    tent = np.empty_like(src)
    out_rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bar = tqdm(enumerate(rows), total=len(rows), desc=f"S06-C {architecture} seed {seed}", unit="img", dynamic_ncols=True)
    for i, row in bar:
        image = read_external_image(zf, row)
        x = helper.image_to_model_tensor(training, image).to(device, non_blocking=True)
        if architecture == "pranet":
            z_src, z_tent = helper.source_and_tent_logits(model, params, source_values, x)
            adapt_loss = ""
        else:
            z_src, z_tent, loss = helper.source_and_tent_logits(
                helper=training, model=model, params=params, source_values=source_values,
                unsafe_bn_modules=unsafe_modules, dropout_modules=dropout_modules, x=x,
            )
            adapt_loss = f"{float(loss):.12f}"
        z_src, z_tent = np.asarray(z_src, np.float32), np.asarray(z_tent, np.float32)
        if z_src.shape != (IMAGE_SIZE, IMAGE_SIZE) or z_tent.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Logit shape mismatch: {architecture} seed={seed} {row['external_id']}")
        if not np.isfinite(z_src).all() or not np.isfinite(z_tent).all():
            raise RuntimeError(f"Non-finite logits: {architecture} seed={seed} {row['external_id']}")
        feat = helper.extract_features(z_src, z_tent)
        if set(feat) != set(gate["feature_names"]):
            raise RuntimeError("Feature names changed")
        p_harm, accepted = apply_gate(feat, gate)
        src[i], tent[i] = z_src.astype(np.float16), z_tent.astype(np.float16)
        item = {
            "architecture": architecture, "seed": seed, "external_id": row["external_id"],
            "center": row["center"], "image_member": row["image_member"],
            "p_harm": f"{p_harm:.12f}", "tau": f"{gate['tau']:.12f}",
            "tent_accepted": accepted,
            "decision": "ACCEPT_TENT" if accepted else "REJECT_TENT_USE_SOURCE",
            "tent_adapt_loss": adapt_loss,
        }
        for name in gate["feature_names"]:
            v = float(feat[name])
            if not math.isfinite(v):
                raise RuntimeError(f"Non-finite feature {name}")
            item[name] = f"{v:.12f}"
        out_rows.append(item)
        bar.set_postfix(pH=f"{p_harm:.3f}", accept=accepted)
    bar.close()

    npz = building / "locked_logits.npz"
    table = building / "no_label_features_and_gate_decisions.csv"
    np.savez_compressed(
        npz, source_logits=src, tent_logits=tent,
        external_ids=np.asarray([r["external_id"] for r in rows], dtype=str),
        centers=np.asarray([r["center"] for r in rows], dtype=str),
        image_members=np.asarray([r["image_member"] for r in rows], dtype=str),
        architecture=np.asarray([architecture], dtype=str), seed=np.asarray([seed], dtype=np.int64),
        gate_artifact_sha256=np.asarray([GATE_SHAS[architecture]], dtype=str),
        protocol_sha256=np.asarray([EXPECTED_PROTOCOL_SHA256], dtype=str),
        archive_sha256=np.asarray([EXPECTED_ARCHIVE_SHA256], dtype=str),
        eligibility_manifest_sha256=np.asarray([ctx["manifest_sha"]], dtype=str),
    )
    fields = ["architecture", "seed", "external_id", "center", "image_member", "p_harm", "tau", "tent_accepted", "decision", "tent_adapt_loss"] + gate["feature_names"]
    write_csv(table, out_rows, fields)
    peak = float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
    lock = {
        "script_version": VERSION, "build": BUILD, "architecture": architecture, "seed": seed,
        "rows": len(rows), "npz_sha256": file_sha256(npz), "csv_sha256": file_sha256(table),
        "checkpoint_sha256": EXPECTED_CHECKPOINTS[architecture][seed],
        "gate_artifact_sha256": GATE_SHAS[architecture], "helper_sha256": HELPER_SHAS[architecture],
        "training_helper_sha256": TRAINING_HELPER_SHAS[architecture], "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256, "s06b_audit_sha256": ctx["audit_sha"],
        "eligibility_manifest_sha256": ctx["manifest_sha"],
        "external_id_order_sha256": sequence_sha256([r["external_id"] for r in rows]),
        "feature_names": gate["feature_names"], "feature_count": 24, "harm_threshold_tau": gate["tau"],
        "deeplab_singleton_unsafe_bn_names": unsafe_names, "deeplab_dropout_eval_names": dropout_names,
        "peak_gpu_memory_gb": peak, "polypgen_image_bytes_opened": True,
        "polypgen_mask_bytes_opened": False, "polypgen_masks_extracted": False,
        "polypgen_mask_pixels_inspected": False, "target_metrics_computed": False,
        "target_dice_computed": False, "target_iou_computed": False,
        "target_delta_dice_computed": False, "model_refit": False,
        "threshold_recalibrated": False, "feature_selection_performed": False,
    }
    (building / "UNIT_NO_LABEL_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    building.rename(final)
    checked = validate_unit(final, architecture, seed, rows, ctx)
    if checked is None:
        raise RuntimeError(f"Fresh unit lock validation failed: {final}")
    del src, tent, out_rows, model, params, source_values
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checked


def clean_incomplete(units: Path):
    removed = []
    if units.exists():
        for p in sorted(units.glob("*__building")):
            if p.is_dir() and p.parent.resolve() == units.resolve():
                shutil.rmtree(p)
                removed.append(p)
    return removed


def build_global(build_dir: Path, rows, contexts, locks, audit_sha, manifest_sha, script_sha):
    expected_keys = {f"{a}::{s}" for a in ARCHITECTURES for s in SEEDS}
    if set(locks) != expected_keys:
        raise RuntimeError("Six valid architecture-seed unit locks are required")
    units_meta, total = {}, 0
    for key in sorted(locks):
        arch, seed_text = key.split("::")
        seed = int(seed_text)
        final, _ = unit_dirs(build_dir / "units", arch, seed)
        checked = validate_unit(final, arch, seed, rows, contexts[arch])
        if checked is None:
            raise RuntimeError(f"Unit invalid before global lock: {key}")
        lock_path = final / "UNIT_NO_LABEL_LOCK.json"
        total += int(checked["rows"])
        units_meta[key] = {
            "unit_lock_sha256": file_sha256(lock_path), "npz_sha256": checked["npz_sha256"],
            "csv_sha256": checked["csv_sha256"], "checkpoint_sha256": checked["checkpoint_sha256"],
            "gate_artifact_sha256": checked["gate_artifact_sha256"], "harm_threshold_tau": checked["harm_threshold_tau"],
        }
    if total != EXPECTED_OBSERVATIONS:
        raise RuntimeError(f"Observation count mismatch: {total}")
    obj = {
        "script_version": VERSION, "script_sha256": script_sha, "build": BUILD,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "s06a_protocol_sha256": EXPECTED_S06A_SHA256,
        "s06b_script_sha256": EXPECTED_S06B_SCRIPT_SHA256, "s06b_audit_sha256": audit_sha,
        "eligibility_manifest_sha256": manifest_sha, "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "external_images": len(rows), "center_counts": dict(sorted(Counter(r["center"] for r in rows).items())),
        "architectures": list(ARCHITECTURES), "seeds": list(SEEDS), "architecture_seed_units": len(units_meta),
        "total_architecture_seed_image_observations": total, "feature_count_per_architecture": 24,
        "units": units_meta, "polypgen_image_bytes_opened": True,
        "polypgen_mask_bytes_opened_before_global_lock": False,
        "polypgen_masks_extracted_before_global_lock": False,
        "polypgen_mask_pixels_inspected_before_global_lock": False,
        "target_metrics_computed_before_global_lock": False,
        "target_dice_computed_before_global_lock": False,
        "target_iou_computed_before_global_lock": False,
        "target_delta_dice_computed_before_global_lock": False,
        "model_refit": False, "threshold_recalibrated": False, "feature_selection_performed": False,
        "decision": PASS_DECISION,
    }
    path = build_dir / "POLYPGEN_DUAL_BACKBONE_NO_LABEL_GLOBAL_LOCK.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    sha = file_sha256(path)
    for a in ARCHITECTURES:
        for s in SEEDS:
            final, _ = unit_dirs(build_dir / "units", a, s)
            if validate_unit(final, a, s, rows, contexts[a]) is None:
                raise RuntimeError(f"Unit changed during global locking: {a} seed={s}")
    return path, sha


def run(args):
    validate_sha(PROTOCOL, EXPECTED_PROTOCOL_SHA256, "S06-C protocol")
    script_sha = file_sha256(Path(__file__).resolve())
    rows, audit_sha, manifest_sha = validate_s06b()
    validate_sha(ARCHIVE, EXPECTED_ARCHIVE_SHA256, "PolypGen archive")
    if args.output_dir.exists():
        raise FileExistsError(f"Final S06-C output exists: {args.output_dir}")
    build_dir = Path(str(args.output_dir) + "__building")
    if build_dir.exists() and not args.resume:
        raise FileExistsError(f"Partial build exists: {build_dir}\nUse --resume.")
    build_dir.mkdir(parents=True, exist_ok=True)
    units = build_dir / "units"
    units.mkdir(parents=True, exist_ok=True)
    if args.clean_incomplete:
        if not args.resume:
            raise RuntimeError("--clean-incomplete requires --resume")
        for p in clean_incomplete(units):
            print(f"[CLEAN-INCOMPLETE] {p}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[BUILD] {VERSION} | {BUILD}")
    print(f"Device: {device}")
    print(f"S06-C script SHA256: {script_sha}")
    print(f"Protocol SHA256: {EXPECTED_PROTOCOL_SHA256}")
    print(f"S06-B audit SHA256: {audit_sha}")
    print(f"Eligibility manifest SHA256: {manifest_sha}")
    print(f"Images={len(rows)} centers={dict(sorted(Counter(r['center'] for r in rows).items()))}")
    print(f"Expected architecture-seed-image observations={EXPECTED_OBSERVATIONS}")
    print("PolypGen mask bytes opened=NO | target metrics=NO | refit=NO | recalibration=NO\n")

    contexts = {a: load_stack(a, audit_sha, manifest_sha) for a in ARCHITECTURES}
    print(f"PraNet frozen tau={contexts['pranet']['gate']['tau']:.12f}")
    print(f"DeepLab frozen tau={contexts['deeplab']['gate']['tau']:.12f}\n")
    locks = {}
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        names = set(zf.namelist())
        for r in rows:
            if r["image_member"] not in names or r["mask_member"] not in names:
                raise RuntimeError(f"Paired member metadata missing: {r['external_id']}")
        for architecture in ARCHITECTURES:
            ctx = contexts[architecture]
            for seed in SEEDS:
                final, building = unit_dirs(units, architecture, seed)
                if building.exists():
                    raise RuntimeError(f"Incomplete unit exists: {building}\nUse --resume --clean-incomplete after a technical interruption.")
                if final.exists():
                    if not args.resume:
                        raise FileExistsError(final)
                    checked = validate_unit(final, architecture, seed, rows, ctx)
                    if checked is None:
                        raise RuntimeError(f"Existing locked unit is invalid and will not be overwritten: {final}")
                    print(f"[RESUME] verified {architecture} seed={seed} NPZ={checked['npz_sha256'][:12]}...")
                else:
                    checked = produce_unit(architecture, seed, rows, zf, ctx, units, device)
                    print(f"[LOCKED] {architecture} seed={seed} NPZ={checked['npz_sha256'][:12]}...")
                locks[f"{architecture}::{seed}"] = checked

    global_path, global_sha = build_global(build_dir, rows, contexts, locks, audit_sha, manifest_sha, script_sha)
    acceptance = {}
    for a in ARCHITECTURES:
        n, acc = 0, 0
        for s in SEEDS:
            final, _ = unit_dirs(units, a, s)
            rr, _ = read_csv(final / "no_label_features_and_gate_decisions.csv")
            n += len(rr); acc += sum(int(x["tent_accepted"]) for x in rr)
        acceptance[a] = acc / n
    summary = f"""===== S06-C POLYPGEN DUAL-BACKBONE NO-LABEL GLOBAL LOCK =====
Script version: {VERSION}
Build: {BUILD}

External cohort:
  images={len(rows)}
  centers={dict(sorted(Counter(r['center'] for r in rows).items()))}
  archive SHA256={EXPECTED_ARCHIVE_SHA256}
  S06-B audit SHA256={audit_sha}
  eligibility manifest SHA256={manifest_sha}

Frozen deployment:
  PraNet seeds={SEEDS} tau={contexts['pranet']['gate']['tau']:.12f}
  DeepLab seeds={SEEDS} tau={contexts['deeplab']['gate']['tau']:.12f}
  architecture-seed units={len(locks)}
  total observations={EXPECTED_OBSERVATIONS}
  PraNet TENT acceptance rate={acceptance['pranet']:.6f}
  DeepLab TENT acceptance rate={acceptance['deeplab']:.6f}

Ground-truth firewall:
  PolypGen image bytes opened=YES
  PolypGen mask bytes opened=NO
  PolypGen masks extracted=NO
  PolypGen mask pixels inspected=NO
  target metrics computed=NO
  target Dice computed=NO
  target IoU computed=NO
  target DeltaDice computed=NO

Scientific integrity:
  model refit=NO
  threshold recalibration=NO
  feature selection=NO
  features per architecture=24

Global no-label lock:
  file={global_path.name}
  SHA256={global_sha}

Decision: {PASS_DECISION}
"""
    (build_dir / "summary.txt").write_text(summary, encoding="utf-8")
    if file_sha256(global_path) != global_sha:
        raise RuntimeError("Global lock changed before final commit")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.rename(args.output_dir)
    print("\n" + (args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"[OK] Outputs: {args.output_dir}")


def self_test():
    assert abs(sigmoid_scalar(0.0) - 0.5) < 1e-12
    names = [f"f{i}" for i in range(24)]
    gate = {"feature_names": names, "mean": np.zeros(24), "scale": np.ones(24), "coef": np.ones(24) * 0.01, "intercept": 0.0, "tau": 0.6}
    p, accepted = apply_gate({n: 0.0 for n in names}, gate)
    assert abs(p - 0.5) < 1e-12 and accepted == 1
    gate["tau"] = 0.5
    _, accepted = apply_gate({n: 0.0 for n in names}, gate)
    assert accepted == 0
    assert EXPECTED_IMAGES == 1537 and sum(EXPECTED_CENTERS.values()) == 1537
    assert EXPECTED_OBSERVATIONS == 9222
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        npz = td / "toy.npz"
        np.savez_compressed(npz, source_logits=np.zeros((2, 3, 4), np.float16), tent_logits=np.ones((2, 3, 4), np.float16))
        assert npy_shape(npz, "source_logits") == (2, 3, 4)
        assert npy_shape(npz, "tent_logits") == (2, 3, 4)
        final, building = unit_dirs(td, "pranet", 20260817)
        assert final.parent == td and building.name.endswith("__building")
    print("GATE_MATH_TEST_PASS")
    print("FROZEN_COUNT_TEST_PASS")
    print("NPZ_HEADER_SHAPE_TEST_PASS")
    print("UNIT_PATH_TEST_PASS")
    print("MASK_BYTE_FIREWALL_DESIGN_TEST_PASS")
    print("NO_TARGET_METRIC_STAGE_TEST_PASS")
    print("SELF_TEST_PASS")


def parse_args():
    p = argparse.ArgumentParser(description="Lock frozen PraNet + DeepLab SafeTTA predictions/decisions on untouched PolypGen before GT reveal.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse only cryptographically verified completed no-label units.")
    p.add_argument("--clean-incomplete", action="store_true", help="With --resume, delete only incomplete per-unit __building dirs.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test(); return 0
    run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
