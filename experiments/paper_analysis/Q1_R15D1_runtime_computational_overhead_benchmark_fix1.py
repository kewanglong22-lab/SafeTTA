#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15D1_runtime_computational_overhead_benchmark_fix1.py

Final post-freeze runtime / computational-overhead benchmark for the frozen
SafeTTA safety module.

Upstream protocol:
  Q1_R15D0_runtime_computational_overhead_preflight_fix2_v1

Frozen runtime cohort:
- 256 SUN-SEG frames
- outcome-independent SHA256-based selection
- no GT / Dice / HARM / BENEFIT access

Frozen method:
1) RGB direct bicubic 224x224
2) frozen facebook/dinov2-base
3) 16x16 patch tokens
4) SOURCE binary prediction -> exact 22x22 block occupancy -> 16x16
5) foreground/background weighted semantic pooling
6) float32 pooling -> float16 lock -> float32 concat (1536-D)
7) frozen PCA64.transform()
8) M2 first (fg fraction, boundary density) + PCA64 = 66-D
9) frozen safety_head.predict_proba()

Benchmark reports:
A) exact historical end-to-end DINO semantic extraction latency
   (warm filesystem cache; includes image open/decode, RGB conversion,
    bicubic224, processor, host->device, model forward, device->host tokens)
B) GPU forward-only latency with exact preprocessed tensors already on GPU
C) SOURCE-mask feature latency (packed mask -> occupancy + M2)
D) conditioned pooling + PCA64 + safety-head latency
E) combined post-DINO safety-score latency
F) combined total safety-module latency estimates per DINO batch size
G) CUDA peak memory, parameter count, serialized safety-artifact sizes

No source-segmentation/TTA runtime is mixed into the main safety overhead.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


VERSION = "2026-09-04-Q1-R15D1-v1-fix1"
BUILD = "Q1_R15D1_RUNTIME_COMPUTATIONAL_OVERHEAD_BENCHMARK_FIX1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

R15D0_DIR = (
    OUT
    / "Q1_R15D0_runtime_computational_overhead_preflight_fix2_v1"
)
R15D0_LOCK = (
    R15D0_DIR
    / "R15D0_RUNTIME_OVERHEAD_PROTOCOL_LOCK.json"
)
RUNTIME_COHORT = (
    R15D0_DIR
    / "R15D0_RUNTIME_COHORT_256.csv"
)

EXPECTED_R15D0_LOCK_SHA = (
    "b2981c6530cab8df33781e6c55b1195e7836f5d2c95c639e2791e5c225226bef"
)
EXPECTED_RUNTIME_COHORT_SHA = (
    "783fe894cc6e660ca1e6a3bc075082255651b4bf3e53a9ce73cd6069a49931b2"
)
EXPECTED_R15D0_DECISION = (
    "RUNTIME_OVERHEAD_LINEAGE_AND_PROTOCOL_LOCKED_READY_FOR_R15D1"
)

R14C2B_SCRIPT = (
    CODE
    / "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1.py"
)
R10K2A_SCRIPT = (
    CODE
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py"
)
R10J1_SCRIPT = (
    CODE
    / "Q1_R10J1_source_mask_morphology_feature_builder_fix1.py"
)
R10L3B_SCRIPT = (
    CODE
    / "Q1_R10L3B_polypgen_frozen_safety_score_lock_fix1.py"
)

EXPECTED_R10K2A_SCRIPT_SHA = (
    "25853e849fecaf847f7ba04dcc2c8b520aa3e248f50af5ac21d19bb0ce4c4778"
)
EXPECTED_R10J1_SCRIPT_SHA = (
    "88555d55c33a57c76cf68e7605f267c500fe8b44921706300131879ca013dccf"
)
EXPECTED_R10L3B_SCRIPT_SHA = (
    "e19e802355dc36873af4bf78ac336ba24313eea8beb19c40c3f646eb4effc646"
)

R10L0_DIR = (
    OUT
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
PCA_PATH = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
HEAD_PATH = R10L0_DIR / "R10L0_final_safety_head.joblib"

EXPECTED_PCA_SHA = (
    "2c6c289ce395105a0c79f7b7751df2a195d76a248c22b43e817d71511c4a24f1"
)
EXPECTED_HEAD_SHA = (
    "602574418a309400bf7e7fc854267a702496c1bd47e4f07f6f60415ec9bd14ce"
)

MODEL_ID = "facebook/dinov2-base"
FROZEN_THRESHOLD = 0.300584763193734

BENCHMARK_FRAMES = 256
DINO_BATCH_SIZES = [1, 4, 8, 16]
WARMUP_STEPS = 20
TIMED_STEPS = 100
DOWNSTREAM_REPEATS = 500
DOWNSTREAM_WARMUP = 20

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15D1_runtime_computational_overhead_benchmark_fix1_v1"
)

DECISION = (
    "FROZEN_SAFETY_MODULE_RUNTIME_AND_COMPUTATIONAL_OVERHEAD_BENCHMARK_COMPLETE"
)


def sha256_file(path: Path, chunk=8 * 1024 * 1024) -> str:
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_sha(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: {got}"
        )
    print(label, got, "PASS")
    return got


def stats_ms(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {
            "mean_ms": math.nan,
            "median_ms": math.nan,
            "p95_ms": math.nan,
            "std_ms": math.nan,
        }
    return {
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "std_ms": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def verify_r15d0():
    print("===== R15D0 FROZEN PROTOCOL GATE =====")

    assert_sha(
        R15D0_LOCK,
        EXPECTED_R15D0_LOCK_SHA,
        "R15D0_LOCK",
    )
    assert_sha(
        RUNTIME_COHORT,
        EXPECTED_RUNTIME_COHORT_SHA,
        "RUNTIME_COHORT",
    )

    lock = load_json(R15D0_LOCK)
    if lock.get("status") != "PASS":
        raise RuntimeError("R15D0 status changed.")
    if lock.get("decision") != EXPECTED_R15D0_DECISION:
        raise RuntimeError("R15D0 decision changed.")

    protocol = lock.get("benchmark_protocol", {})
    if list(protocol.get("dino_batch_sizes", [])) != DINO_BATCH_SIZES:
        raise RuntimeError("DINO batch sizes changed.")
    if int(protocol.get("warmup_steps", -1)) != WARMUP_STEPS:
        raise RuntimeError("Warmup steps changed.")
    if int(protocol.get("timed_steps", -1)) != TIMED_STEPS:
        raise RuntimeError("Timed steps changed.")
    if int(protocol.get("downstream_repeats", -1)) != DOWNSTREAM_REPEATS:
        raise RuntimeError("Downstream repeats changed.")
    if not bool(protocol.get("cuda_synchronize_before_after_timer", False)):
        raise RuntimeError("CUDA synchronize timing rule changed.")

    cohort = lock.get("runtime_cohort", {})
    if int(cohort.get("actual_frames", -1)) != BENCHMARK_FRAMES:
        raise RuntimeError("Runtime cohort size changed.")
    if bool(cohort.get("gt_or_outcome_used", True)):
        raise RuntimeError("Runtime cohort used GT/outcome.")

    print("PASS")
    return lock


def verify_method_lineage():
    print("\n===== EXACT METHOD LINEAGE =====")
    assert_sha(
        R10K2A_SCRIPT,
        EXPECTED_R10K2A_SCRIPT_SHA,
        "R10K2A",
    )
    assert_sha(
        R10J1_SCRIPT,
        EXPECTED_R10J1_SCRIPT_SHA,
        "R10J1",
    )
    assert_sha(
        R10L3B_SCRIPT,
        EXPECTED_R10L3B_SCRIPT_SHA,
        "R10L3B",
    )
    assert_sha(
        PCA_PATH,
        EXPECTED_PCA_SHA,
        "PCA",
    )
    assert_sha(
        HEAD_PATH,
        EXPECTED_HEAD_SHA,
        "HEAD",
    )
    if not R14C2B_SCRIPT.is_file():
        raise FileNotFoundError(R14C2B_SCRIPT)
    print("R14C2B", sha256_file(R14C2B_SCRIPT), "PRESENT")
    print("PASS")


def resolve_runtime_rgb(cohort, rgb):
    """
    Join the R15D0 frozen runtime cohort to the no-GT RGB cache manifest.

    Prefer exact encoded image SHA because it is independent of filename or
    local-cache naming. Fall back only to another exact one-to-one key.
    """
    candidates = [
        ("image_raw_sha256", "image_raw_sha256"),
        ("pair_key", "sample_id"),
        ("frame_member", "frame_member"),
    ]

    for left, right in candidates:
        if left not in cohort.columns or right not in rgb.columns:
            continue

        a = cohort[left].astype(str)
        b = rgb[right].astype(str)

        if a.duplicated().any() or b.duplicated().any():
            continue

        temp = cohort.copy()
        temp["_join_key"] = a
        rr = rgb.copy()
        rr["_join_key"] = b

        joined = temp.merge(
            rr[
                [
                    "_join_key",
                    "sample_id",
                    "image_path",
                    "image_raw_sha256",
                ]
            ],
            on="_join_key",
            how="left",
            validate="one_to_one",
            suffixes=("_cohort", "_rgb"),
        )

        if len(joined) != BENCHMARK_FRAMES:
            continue
        if joined["image_path"].isna().any():
            continue

        print(
            "runtime RGB join:",
            f"{left} -> {right}",
            "PASS",
        )
        return joined

    raise RuntimeError(
        "Could not construct exact one-to-one runtime RGB mapping."
    )


def make_path_batches(paths, batch_size, steps):
    paths = list(paths)
    n = len(paths)
    batches = []

    for i in range(steps):
        start = (i * batch_size) % n
        idx = [
            (start + j) % n
            for j in range(batch_size)
        ]
        batches.append([paths[j] for j in idx])

    return batches


def exact_processor_inputs(
    paths,
    processor,
    k2a,
    device,
):
    pil_images = []
    for p in paths:
        with Image.open(Path(str(p))) as im:
            rgb = im.convert("RGB")
            rgb = rgb.resize(
                (k2a.IMAGE_SIZE, k2a.IMAGE_SIZE),
                resample=Image.Resampling.BICUBIC,
            )
            pil_images.append(rgb.copy())

    inputs = processor(
        images=pil_images,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return {
        k: v.to(device)
        for k, v in inputs.items()
    }


def benchmark_dino_full(
    paths,
    batch_size,
    processor,
    model,
    device,
    k2a,
    l3b,
):
    """
    Exact historical run_dino_batch latency:
    disk/cache read + PIL decode + RGB + bicubic224 + processor +
    H2D + model + D2H/NumPy.
    """
    warm_batches = make_path_batches(
        paths,
        batch_size,
        WARMUP_STEPS,
    )
    timed_batches = make_path_batches(
        paths,
        batch_size,
        TIMED_STEPS,
    )

    for batch in warm_batches:
        _ = l3b.run_dino_batch(
            batch,
            processor,
            model,
            device,
            k2a,
            torch,
        )

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    baseline_alloc = int(
        torch.cuda.memory_allocated()
    )
    torch.cuda.reset_peak_memory_stats()

    samples_ms = []

    for batch in tqdm(
        timed_batches,
        desc=f"DINO exact E2E bs={batch_size}",
        unit="step",
        dynamic_ncols=True,
    ):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()

        _ = l3b.run_dino_batch(
            batch,
            processor,
            model,
            device,
            k2a,
            torch,
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        samples_ms.append(
            (t1 - t0) / 1e6 / batch_size
        )

    peak_alloc = int(
        torch.cuda.max_memory_allocated()
    )

    s = stats_ms(samples_ms)
    s.update({
        "component": "DINO_EXACT_END_TO_END",
        "batch_size": batch_size,
        "timed_steps": TIMED_STEPS,
        "images_per_second_from_mean": (
            1000.0 / s["mean_ms"]
            if s["mean_ms"] > 0
            else math.nan
        ),
        "baseline_cuda_allocated_MiB": (
            baseline_alloc / (1024 ** 2)
        ),
        "peak_cuda_allocated_MiB": (
            peak_alloc / (1024 ** 2)
        ),
        "incremental_peak_cuda_MiB": (
            max(0, peak_alloc - baseline_alloc)
            / (1024 ** 2)
        ),
        "includes_disk_decode_resize_processor_h2d_d2h": True,
    })

    return s


def benchmark_dino_gpu_forward(
    paths,
    batch_size,
    processor,
    model,
    device,
    k2a,
):
    """
    GPU forward only. Exact preprocessed input is created before timing.
    No CPU preprocessing / H2D / D2H in the timed region.
    """
    batch = list(paths[:batch_size])
    inputs = exact_processor_inputs(
        batch,
        processor,
        k2a,
        device,
    )

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(**inputs)

    torch.cuda.synchronize()
    baseline_alloc = int(
        torch.cuda.memory_allocated()
    )
    torch.cuda.reset_peak_memory_stats()

    samples_ms = []

    for _ in tqdm(
        range(TIMED_STEPS),
        desc=f"DINO GPU forward bs={batch_size}",
        unit="step",
        dynamic_ncols=True,
    ):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()

        with torch.no_grad():
            outputs = model(**inputs)

        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        # Touch shape only; do not D2H in this forward-only benchmark.
        if (
            outputs.last_hidden_state.shape[1]
            != 1 + k2a.PATCH_GRID * k2a.PATCH_GRID
        ):
            raise RuntimeError(
                "DINO forward output shape changed."
            )

        samples_ms.append(
            (t1 - t0) / 1e6 / batch_size
        )

    peak_alloc = int(
        torch.cuda.max_memory_allocated()
    )

    s = stats_ms(samples_ms)
    s.update({
        "component": "DINO_GPU_FORWARD_ONLY",
        "batch_size": batch_size,
        "timed_steps": TIMED_STEPS,
        "images_per_second_from_mean": (
            1000.0 / s["mean_ms"]
            if s["mean_ms"] > 0
            else math.nan
        ),
        "baseline_cuda_allocated_MiB": (
            baseline_alloc / (1024 ** 2)
        ),
        "peak_cuda_allocated_MiB": (
            peak_alloc / (1024 ** 2)
        ),
        "incremental_peak_cuda_MiB": (
            max(0, peak_alloc - baseline_alloc)
            / (1024 ** 2)
        ),
        "includes_disk_decode_resize_processor_h2d_d2h": False,
    })

    del inputs
    del outputs
    torch.cuda.empty_cache()

    return s


def extract_runtime_patch_tokens(
    rgb_runtime,
    processor,
    model,
    device,
    k2a,
    l3b,
    batch_size=16,
):
    """
    Untimed exact frozen DINO extraction for downstream benchmark preparation.
    """
    patch_by_sid = {}

    for start in tqdm(
        range(0, len(rgb_runtime), batch_size),
        desc="Prepare frozen patch tokens",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + batch_size, len(rgb_runtime))
        b = rgb_runtime.iloc[start:stop]

        _, patch = l3b.run_dino_batch(
            b["image_path"].tolist(),
            processor,
            model,
            device,
            k2a,
            torch,
        )

        for j, row in enumerate(
            b.itertuples(index=False)
        ):
            patch_by_sid[str(row.sample_id)] = (
                np.asarray(
                    patch[j],
                    dtype=np.float32,
                )
            )

    if len(patch_by_sid) != BENCHMARK_FRAMES:
        raise RuntimeError(
            "Runtime patch-token cache cardinality failure."
        )

    return patch_by_sid


def build_packed_source_lookup(states_runtime):
    """
    Read SOURCE packed rows only. Adapted mask values are never touched.
    """
    packed = {}

    for state_id, g in states_runtime.groupby(
        "model_state_id",
        sort=True,
    ):
        npz_values = (
            g["state_prediction_npz"]
            .astype(str)
            .unique()
        )
        if len(npz_values) != 1:
            raise RuntimeError(
                f"{state_id}: expected one NPZ."
            )

        path = Path(npz_values[0])
        with np.load(path, allow_pickle=False) as z:
            if "source_masks_packed" not in z.files:
                raise RuntimeError(
                    f"{state_id}: SOURCE key missing."
                )
            source = z["source_masks_packed"]

            for row in g.itertuples(index=False):
                key = (
                    str(row.sample_id),
                    str(row.model_state_id),
                )
                packed[key] = np.asarray(
                    source[int(row.state_row_index)],
                    dtype=np.uint8,
                ).copy()

    if len(packed) != len(states_runtime):
        raise RuntimeError(
            "Packed runtime SOURCE lookup cardinality mismatch."
        )

    return packed


def deterministic_state_sequence(states_runtime, n):
    x = states_runtime.copy()
    x["_key"] = (
        x["sample_id"].astype(str)
        + "|"
        + x["model_state_id"].astype(str)
    )
    x["_hash"] = x["_key"].map(
        lambda s: hashlib.sha256(
            s.encode("utf-8")
        ).hexdigest()
    )
    x = x.sort_values(
        ["_hash", "_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    idx = [
        i % len(x)
        for i in range(n)
    ]
    return x.iloc[idx].reset_index(drop=True)


def mask_features_from_packed(
    packed,
    k2a,
    j1,
):
    occ = k2a.unpack_mask_occupancy16(
        packed
    )
    mask = j1.unpack_source_mask(
        packed
    )

    m2 = np.asarray(
        [
            float(mask.mean()),
            float(j1.boundary_density(mask)),
        ],
        dtype=np.float64,
    )

    return occ.astype(np.float32), m2


def conditioned_score_from_precomputed(
    patch,
    occupancy,
    m2,
    k2a,
    pca,
    head,
):
    fg, bg = k2a.pooled_region_features(
        patch,
        occupancy.reshape(1, -1),
    )

    # Exact historical precision lock.
    fg16 = fg.astype(np.float16)
    bg16 = bg.astype(np.float16)

    cond = np.concatenate(
        [
            fg16.astype(np.float32),
            bg16.astype(np.float32),
        ],
        axis=1,
    )

    z = pca.transform(cond)

    X = np.column_stack(
        [
            np.asarray(
                m2,
                dtype=np.float64,
            ).reshape(1, 2),
            np.asarray(
                z,
                dtype=np.float64,
            ),
        ]
    )

    prob = head.predict_proba(X)[:, 1]

    return float(prob[0])


def full_post_dino_score(
    packed,
    patch,
    k2a,
    j1,
    pca,
    head,
):
    occ, m2 = mask_features_from_packed(
        packed,
        k2a,
        j1,
    )
    return conditioned_score_from_precomputed(
        patch,
        occ,
        m2,
        k2a,
        pca,
        head,
    )


def benchmark_cpu_stage(
    name,
    sequence,
    fn,
    warmup=DOWNSTREAM_WARMUP,
):
    for i in range(warmup):
        row = sequence.iloc[
            i % len(sequence)
        ]
        fn(row)

    gc.collect()

    samples_ms = []

    for i in tqdm(
        range(DOWNSTREAM_REPEATS),
        desc=name,
        unit="call",
        dynamic_ncols=True,
    ):
        row = sequence.iloc[
            i % len(sequence)
        ]

        t0 = time.perf_counter_ns()
        _ = fn(row)
        t1 = time.perf_counter_ns()

        samples_ms.append(
            (t1 - t0) / 1e6
        )

    s = stats_ms(samples_ms)
    s.update({
        "component": name,
        "repeats": DOWNSTREAM_REPEATS,
        "calls_per_second_from_mean": (
            1000.0 / s["mean_ms"]
            if s["mean_ms"] > 0
            else math.nan
        ),
    })

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1 R15D1 RUNTIME / COMPUTATIONAL OVERHEAD BENCHMARK ====="
    )
    print("STATUS=POST_FREEZE_RUNTIME_BENCHMARK")
    print("MODEL_FITTING=NO")
    print("SCORE_RECALIBRATION=NO")
    print("GT_ACCESS=NO")
    print("OUTCOME_ACCESS=NO")
    print("METHOD_CHANGE=NO")
    print("SOURCE_SEGMENTATION_TIMED=NO")
    print("TTA_TIMED=NO")
    print("BENCHMARK_FRAMES=", BENCHMARK_FRAMES)
    print("DINO_BATCH_SIZES=", DINO_BATCH_SIZES)
    print("WARMUP_STEPS=", WARMUP_STEPS)
    print("TIMED_STEPS=", TIMED_STEPS)
    print("DOWNSTREAM_REPEATS=", DOWNSTREAM_REPEATS)
    print("CUDA_SYNCHRONIZE_BEFORE_AFTER_TIMER=YES")

    r15d0 = verify_r15d0()
    verify_method_lineage()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if args.output_dir.exists():
        raise FileExistsError(
            args.output_dir
        )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Import exact historical method modules.
    r14c2b = import_module(
        R14C2B_SCRIPT,
        "q1_r15d1_r14c2b",
    )
    k2a = import_module(
        R10K2A_SCRIPT,
        "q1_r15d1_k2a",
    )
    j1 = import_module(
        R10J1_SCRIPT,
        "q1_r15d1_j1",
    )
    l3b = import_module(
        R10L3B_SCRIPT,
        "q1_r15d1_l3b",
    )

    # Reuse exact frozen SUN pre-GT representation loader.
    # This accesses SOURCE masks only and no GT/outcomes.
    rgb, states, rgb_cache = (
        r14c2b.load_sun_no_gt_manifests()
    )
    occupancy_all, m2_all, fg_agree = (
        r14c2b.build_source_mask_features(
            states,
            k2a,
            j1,
        )
    )

    if fg_agree != len(states):
        raise RuntimeError(
            "SOURCE-mask foreground-pixel sentinel failed."
        )

    cohort = pd.read_csv(
        RUNTIME_COHORT,
        low_memory=False,
    )
    if len(cohort) != BENCHMARK_FRAMES:
        raise RuntimeError(
            "Runtime cohort row count changed."
        )

    joined = resolve_runtime_rgb(
        cohort,
        rgb,
    )

    # Normalize output runtime RGB table.
    rgb_runtime = pd.DataFrame({
        "sample_id": joined["sample_id"].astype(str),
        "image_path": joined["image_path"].astype(str),
        "image_raw_sha256": joined[
            "image_raw_sha256_rgb"
            if "image_raw_sha256_rgb" in joined.columns
            else "image_raw_sha256"
        ].astype(str),
    })

    if rgb_runtime["sample_id"].nunique() != BENCHMARK_FRAMES:
        raise RuntimeError(
            "Runtime RGB sample IDs are not unique."
        )

    missing_paths = [
        p
        for p in rgb_runtime["image_path"].tolist()
        if not Path(str(p)).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            missing_paths[0]
        )

    selected_sids = set(
        rgb_runtime["sample_id"].astype(str)
    )

    states_runtime = states[
        states["sample_id"]
        .astype(str)
        .isin(selected_sids)
    ].copy()

    if len(states_runtime) != BENCHMARK_FRAMES * 9:
        raise RuntimeError(
            f"Runtime state rows={len(states_runtime)} "
            f"expected={BENCHMARK_FRAMES * 9}"
        )

    print("\n===== RUNTIME COHORT RESOLUTION =====")
    print("RGB rows:", len(rgb_runtime))
    print("state rows:", len(states_runtime))
    print("RGB cache:", rgb_cache)
    print("GT/outcome access: NO")
    print("PASS")

    # Load exact frozen DINO via historical deployment function.
    torch_mod, processor, model, device = (
        l3b.load_frozen_dinov2(
            k2a,
            local_files_only=True,
            device_name="cuda",
        )
    )
    if torch_mod is not torch:
        # Module object identity normally matches, but exact identity is not
        # scientifically important; version must match.
        if torch_mod.__version__ != torch.__version__:
            raise RuntimeError(
                "Torch module version mismatch."
            )

    # Exact representation sentinel before timing.
    sentinel_diff = l3b.representation_sentinel(
        processor,
        model,
        device,
        k2a,
        torch,
    )

    parameter_count = int(
        sum(p.numel() for p in model.parameters())
    )
    trainable_parameter_count = int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    print("\n===== MODEL COMPLEXITY =====")
    print("DINO parameters:", parameter_count)
    print(
        "DINO trainable parameters:",
        trainable_parameter_count,
    )
    print(
        "Approx FP32 parameter memory MiB:",
        parameter_count * 4 / (1024 ** 2),
    )

    # ------------------------------------------------------------------
    # A/B. DINO benchmarks.
    # ------------------------------------------------------------------
    paths = rgb_runtime["image_path"].tolist()
    dino_rows = []

    for bs in DINO_BATCH_SIZES:
        full = benchmark_dino_full(
            paths=paths,
            batch_size=bs,
            processor=processor,
            model=model,
            device=device,
            k2a=k2a,
            l3b=l3b,
        )
        dino_rows.append(full)

        gpu = benchmark_dino_gpu_forward(
            paths=paths,
            batch_size=bs,
            processor=processor,
            model=model,
            device=device,
            k2a=k2a,
        )
        dino_rows.append(gpu)

        print(
            f"\nBATCH={bs} "
            f"ExactE2E median={full['median_ms']:.4f} ms/img "
            f"mean={full['mean_ms']:.4f} "
            f"p95={full['p95_ms']:.4f} "
            f"throughput={full['images_per_second_from_mean']:.2f} img/s "
            f"peak={full['peak_cuda_allocated_MiB']:.1f} MiB"
        )
        print(
            f"BATCH={bs} "
            f"GPUForward median={gpu['median_ms']:.4f} ms/img "
            f"mean={gpu['mean_ms']:.4f} "
            f"p95={gpu['p95_ms']:.4f} "
            f"throughput={gpu['images_per_second_from_mean']:.2f} img/s "
            f"peak={gpu['peak_cuda_allocated_MiB']:.1f} MiB"
        )

    dino_df = pd.DataFrame(
        dino_rows
    )

    # ------------------------------------------------------------------
    # Prepare untimed frozen patch tokens and SOURCE packed masks.
    # ------------------------------------------------------------------
    patch_by_sid = extract_runtime_patch_tokens(
        rgb_runtime,
        processor,
        model,
        device,
        k2a,
        l3b,
        batch_size=16,
    )

    packed_lookup = build_packed_source_lookup(
        states_runtime
    )

    # Align precomputed occupancy/M2 from the exact R14C2B builder to state
    # rows for sentinel and downstream timing.
    occ_map = {}
    m2_map = {}

    for idx, row in states_runtime.iterrows():
        key = (
            str(row["sample_id"]),
            str(row["model_state_id"]),
        )
        occ_map[key] = np.asarray(
            occupancy_all[int(idx)],
            dtype=np.float32,
        )
        m2_map[key] = np.asarray(
            m2_all[int(idx)],
            dtype=np.float64,
        )

    sequence = deterministic_state_sequence(
        states_runtime,
        DOWNSTREAM_REPEATS + DOWNSTREAM_WARMUP,
    )

    # Verify exact local reconstruction against R14C2B precomputed features.
    for row in sequence.iloc[:16].itertuples(index=False):
        key = (
            str(row.sample_id),
            str(row.model_state_id),
        )
        occ, m2 = mask_features_from_packed(
            packed_lookup[key],
            k2a,
            j1,
        )
        if not np.allclose(
            occ,
            occ_map[key],
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Occupancy runtime reconstruction mismatch."
            )
        if not np.allclose(
            m2,
            m2_map[key],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "M2 runtime reconstruction mismatch."
            )

    print("\nRuntime SOURCE-mask feature reconstruction sentinel: PASS")

    pca = joblib.load(PCA_PATH)
    head = joblib.load(HEAD_PATH)

    def row_key(row):
        return (
            str(row["sample_id"]),
            str(row["model_state_id"]),
        )

    def mask_fn(row):
        key = row_key(row)
        return mask_features_from_packed(
            packed_lookup[key],
            k2a,
            j1,
        )

    def score_fn(row):
        key = row_key(row)
        sid = str(row["sample_id"])
        return conditioned_score_from_precomputed(
            patch_by_sid[sid],
            occ_map[key],
            m2_map[key],
            k2a,
            pca,
            head,
        )

    def full_post_fn(row):
        key = row_key(row)
        sid = str(row["sample_id"])
        return full_post_dino_score(
            packed_lookup[key],
            patch_by_sid[sid],
            k2a,
            j1,
            pca,
            head,
        )

    mask_timing = benchmark_cpu_stage(
        "SOURCE_MASK_OCCUPANCY_PLUS_M2",
        sequence,
        mask_fn,
    )
    score_timing = benchmark_cpu_stage(
        "CONDITIONED_POOLING_PCA64_HEAD",
        sequence,
        score_fn,
    )
    post_timing = benchmark_cpu_stage(
        "FULL_POST_DINO_SAFETY_SCORE",
        sequence,
        full_post_fn,
    )

    downstream_df = pd.DataFrame(
        [
            mask_timing,
            score_timing,
            post_timing,
        ]
    )

    print("\n===== DOWNSTREAM CPU TIMING =====")
    print(
        downstream_df[
            [
                "component",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "calls_per_second_from_mean",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Combined total safety overhead.
    # Use exact E2E DINO + exact full post-DINO per-model-frame score.
    # This is the incremental safety cost after SOURCE segmentation has
    # already produced its binary mask.
    # ------------------------------------------------------------------
    total_rows = []

    post_mean = float(
        post_timing["mean_ms"]
    )
    post_median = float(
        post_timing["median_ms"]
    )

    for bs in DINO_BATCH_SIZES:
        d = dino_df[
            (dino_df["component"] == "DINO_EXACT_END_TO_END")
            & (dino_df["batch_size"] == bs)
        ].iloc[0]

        total_rows.append({
            "batch_size": bs,
            "dino_exact_e2e_mean_ms_per_image": float(
                d["mean_ms"]
            ),
            "dino_exact_e2e_median_ms_per_image": float(
                d["median_ms"]
            ),
            "post_dino_mean_ms_per_model_frame": post_mean,
            "post_dino_median_ms_per_model_frame": post_median,
            "estimated_total_mean_ms_per_image": float(
                d["mean_ms"] + post_mean
            ),
            "estimated_total_median_ms_per_image": float(
                d["median_ms"] + post_median
            ),
            "estimated_total_images_per_second_from_mean": float(
                1000.0
                / (
                    float(d["mean_ms"])
                    + post_mean
                )
            ),
            "peak_cuda_allocated_MiB": float(
                d["peak_cuda_allocated_MiB"]
            ),
            "incremental_peak_cuda_MiB": float(
                d["incremental_peak_cuda_MiB"]
            ),
        })

    total_df = pd.DataFrame(
        total_rows
    )

    print("\n===== ESTIMATED TOTAL FROZEN SAFETY OVERHEAD =====")
    print(
        total_df.to_string(index=False)
    )

    # Artifact sizes.
    r15d0_art = (
        r15d0
        .get("frozen_artifacts", {})
    )

    pca_bytes = int(
        PCA_PATH.stat().st_size
    )
    head_bytes = int(
        HEAD_PATH.stat().st_size
    )
    nondino_bytes = pca_bytes + head_bytes

    complexity = {
        "dino_model_id": MODEL_ID,
        "dino_parameters": parameter_count,
        "dino_trainable_parameters": trainable_parameter_count,
        "approx_fp32_parameter_memory_MiB": (
            parameter_count * 4 / (1024 ** 2)
        ),
        "pca_bytes": pca_bytes,
        "head_bytes": head_bytes,
        "non_dino_safety_artifact_bytes_pca_plus_head": nondino_bytes,
        "non_dino_safety_artifact_MiB_pca_plus_head": (
            nondino_bytes / (1024 ** 2)
        ),
        "frozen_representation_sentinel_max_abs_float16": float(
            sentinel_diff
        ),
        "r15d0_frozen_artifacts": r15d0_art,
    }

    # Save.
    dino_path = (
        args.output_dir
        / "R15D1_dino_runtime_by_batch.csv"
    )
    downstream_path = (
        args.output_dir
        / "R15D1_downstream_safety_runtime.csv"
    )
    total_path = (
        args.output_dir
        / "R15D1_total_safety_overhead.csv"
    )
    complexity_path = (
        args.output_dir
        / "R15D1_model_complexity_and_artifact_size.json"
    )

    dino_df.to_csv(
        dino_path,
        index=False,
    )
    downstream_df.to_csv(
        downstream_path,
        index=False,
    )
    total_df.to_csv(
        total_path,
        index=False,
    )
    write_json(
        complexity_path,
        complexity,
    )

    artifacts = {}
    for p in (
        dino_path,
        downstream_path,
        total_path,
        complexity_path,
    ):
        artifacts[p.name] = {
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    hw = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_MiB": (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 2)
        ),
    }

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": (
            "post-freeze runtime and computational-overhead benchmark"
        ),
        "hardware": hw,
        "runtime_cohort": {
            "frames": BENCHMARK_FRAMES,
            "r15d0_cohort_sha256": EXPECTED_RUNTIME_COHORT_SHA,
            "rgb_join_outcome_independent": True,
            "gt_accessed": False,
            "outcomes_accessed": False,
        },
        "method": {
            "model_id": MODEL_ID,
            "parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "frozen_threshold": FROZEN_THRESHOLD,
            "representation_sentinel_max_abs_float16": float(
                sentinel_diff
            ),
        },
        "benchmark_protocol": {
            "dino_batch_sizes": DINO_BATCH_SIZES,
            "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
            "downstream_warmup": DOWNSTREAM_WARMUP,
            "downstream_repeats": DOWNSTREAM_REPEATS,
            "cuda_synchronize_before_after_gpu_timer": True,
            "dino_exact_e2e_includes": [
                "warm-cache file open/decode",
                "RGB conversion",
                "bicubic resize 224x224",
                "historical processor",
                "host-to-device",
                "frozen DINO forward",
                "device-to-host float32 NumPy tokens",
            ],
            "gpu_forward_only_preprocessed_input": True,
            "source_segmentation_timed": False,
            "tta_timed": False,
            "total_overhead_interpretation": (
                "incremental safety-module cost after SOURCE prediction exists"
            ),
        },
        "information_boundary": {
            "model_fitting": False,
            "score_recalibration": False,
            "target_calibration": False,
            "threshold_tuning": False,
            "gt_access": False,
            "outcome_access": False,
            "method_change": False,
        },
        "upstream": {
            "r15d0_lock_sha256": EXPECTED_R15D0_LOCK_SHA,
            "runtime_cohort_sha256": EXPECTED_RUNTIME_COHORT_SHA,
            "r10k2a_script_sha256": EXPECTED_R10K2A_SCRIPT_SHA,
            "r10j1_script_sha256": EXPECTED_R10J1_SCRIPT_SHA,
            "r10l3b_script_sha256": EXPECTED_R10L3B_SCRIPT_SHA,
            "pca_sha256": EXPECTED_PCA_SHA,
            "head_sha256": EXPECTED_HEAD_SHA,
        },
        "artifacts": artifacts,
        "next_stage": (
            "R15E_FINAL_EXPERIMENT_EVIDENCE_SYNTHESIS_AND_PAPER_TABLES"
        ),
    }

    lock_path = (
        args.output_dir
        / "R15D1_RUNTIME_COMPUTATIONAL_OVERHEAD_LOCK.json"
    )
    write_json(
        lock_path,
        lock,
    )

    print("\n===== R15D1 FINAL =====")
    print("Decision=", DECISION)
    print("SOURCE_SEGMENTATION/TTA_RUNTIME_INCLUDED=NO")
    print("GT/OUTCOME_ACCESS=NO")
    print("LOCK=", lock_path)
    print("LOCK SHA256=", sha256_file(lock_path))
    print("PASS")


if __name__ == "__main__":
    main()
