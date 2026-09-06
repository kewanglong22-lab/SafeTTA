#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R15D0_runtime_computational_overhead_preflight_fix1.py

Post-freeze runtime/computational-overhead preflight.

This stage performs NO benchmark inference and NO model fitting.
It only freezes the exact runtime lineage, hardware/software environment,
benchmark cohort rule, benchmark batch sizes, timing protocol, and required
artifacts for the final SafeTTA overhead audit.

Planned R15D1 measurements:
1) frozen DINOv2-base forward latency / throughput;
2) source-mask occupancy + FG/BG pooling latency;
3) PCA64 + frozen safety head latency;
4) total pre-adaptation safety-score overhead;
5) peak CUDA memory;
6) parameter counts / serialized artifact sizes.

The runtime cohort is outcome-independent:
- use the already frozen SUN-SEG confirmatory manifest;
- select up to 256 frames by SHA256(sample_id) order;
- no GT, Dice, HARM, BENEFIT, or adaptation outcome is accessed;
- source masks come only from the already frozen pre-GT SOURCE prediction
  artifacts.

This preflight intentionally does not time SOURCE segmentation or TTA itself.
Those are adaptation/model-specific and would confound the cost of the safety
module. R15D1 will report the incremental safety overhead separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


VERSION = "2026-09-04-Q1-R15D0-v1-fix2"
BUILD = "Q1_R15D0_RUNTIME_COMPUTATIONAL_OVERHEAD_PREFLIGHT_FIX2"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
OUT = ROOT / "outputs"
CODE = ROOT / "code"

# ---------------- exact frozen method lineage ----------------
R10L0_DIR = (
    OUT
    / "Q1_R10L0_final_source_safety_estimator_lock_fix3_v1"
)
PCA_PATH = R10L0_DIR / "R10L0_final_condDINO_PCA64.joblib"
HEAD_PATH = R10L0_DIR / "R10L0_final_safety_head.joblib"
THRESH_PATH = R10L0_DIR / "R10L0_frozen_operating_threshold.json"
LOCK_R10L0 = R10L0_DIR / "R10L0_FINAL_SOURCE_ESTIMATOR_LOCK.json"

EXPECTED_PCA_SHA = (
    "2c6c289ce395105a0c79f7b7751df2a195d76a248c22b43e817d71511c4a24f1"
)
EXPECTED_HEAD_SHA = (
    "602574418a309400bf7e7fc854267a702496c1bd47e4f07f6f60415ec9bd14ce"
)
EXPECTED_THRESHOLD_SHA = (
    "4a37fb4f9a9b729a9aa6551b1417afea7348e2dfecdfcb6e0e2b5986e748e4d7"
)

R14C2B_DIR = (
    OUT
    / "Q1_R14C2B_sunseg_frozen_safety_score_lock_preGT_fix1_v1"
)
R14C2B_LOCK = (
    R14C2B_DIR
    / "R14C2B_SUNSEG_FROZEN_SAFETY_SCORE_LOCK.json"
)
EXPECTED_R14C2B_LOCK_SHA = (
    "29030a2301178575971c8f5fab2c131d7cc06ffa99dbbd98431c288a3343337d"
)

R14C1_DIR = (
    OUT
    / "Q1_R14C1_sunseg_source_tent1_pl_prediction_lock_preGT_fix3_v1"
)
R14C1_LOCK = (
    R14C1_DIR
    / "R14C1_SUNSEG_SOURCE_TENT1_PL_PREDICTION_LOCK.json"
)
EXPECTED_R14C1_LOCK_SHA = (
    "802e585dc075df23e0eeddf2b7d613d229dd1f7c6ae99aeaa89cf7dd7b358163"
)

SUN_MANIFEST = (
    OUT
    / "Q1_R14B4B_sunseg_selected_rgb_contamination_audit_fix3_v1"
    / "R14B4B_FIX3_FINAL_CONFIRMATORY_MANIFEST.csv"
)
EXPECTED_SUN_MANIFEST_SHA = (
    "f38cc8e5af4a007df3394ec95fa621c0b819f488bdd797ddad9338ac12491b4d"
)

AUTHORITATIVE_R10K2A_SCRIPT = (
    CODE
    / "Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py"
)
EXPECTED_R10K2A_SCRIPT_SHA = (
    "25853e849fecaf847f7ba04dcc2c8b520aa3e248f50af5ac21d19bb0ce4c4778"
)

MODEL_ID = "facebook/dinov2-base"
FROZEN_THRESHOLD = 0.300584763193734

# Exact historical R10K2A/R10L3B project-local caches.
DEFAULT_HF_CACHE = ROOT / "cache" / "huggingface"
DEFAULT_TORCH_CACHE = ROOT / "cache" / "torch"

# ---------------- benchmark protocol lock ----------------
BENCHMARK_FRAMES = 256
DINO_BATCH_SIZES = [1, 4, 8, 16]
WARMUP_STEPS = 20
TIMED_STEPS = 100
DOWNSTREAM_REPEATS = 500
CUDA_SYNCHRONIZE = True

DEFAULT_OUTPUT = (
    OUT
    / "Q1_R15D0_runtime_computational_overhead_preflight_fix2_v1"
)

DECISION = (
    "RUNTIME_OVERHEAD_LINEAGE_AND_PROTOCOL_LOCKED_READY_FOR_R15D1"
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


def stable_hash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_file_sha(path: Path, expected_sha: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    print(label, got, "PASS" if got == expected_sha else "FAIL")
    if got != expected_sha:
        raise RuntimeError(
            f"{label} SHA mismatch: {got}"
        )
    return got


def find_sample_id_column(df):
    candidates = [
        "sample_id",
        "frame_id",
        "pair_key",
        "image_id",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        "SUN manifest has no recognized stable sample identifier column. "
        f"Columns={list(df.columns)}"
    )


def choose_runtime_cohort(manifest):
    sid_col = find_sample_id_column(manifest)

    x = manifest.copy()
    x["_runtime_sid"] = x[sid_col].astype(str)
    x["_runtime_hash"] = x["_runtime_sid"].map(stable_hash)

    if x["_runtime_sid"].duplicated().any():
        raise RuntimeError(
            f"Runtime cohort identifier is not unique: {sid_col}"
        )

    x = x.sort_values(
        ["_runtime_hash", "_runtime_sid"],
        kind="mergesort",
    ).reset_index(drop=True)

    n = min(BENCHMARK_FRAMES, len(x))
    selected = x.iloc[:n].copy()

    return sid_col, selected


def configure_exact_historical_caches():
    """
    Exact cache lineage used by frozen R10K2A / R10L3B deployment.
    """
    hf = Path(DEFAULT_HF_CACHE)
    tc = Path(DEFAULT_TORCH_CACHE)

    if not hf.is_dir():
        raise FileNotFoundError(hf)
    if not tc.is_dir():
        raise FileNotFoundError(tc)

    os.environ["HF_HOME"] = str(hf)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf / "transformers")
    os.environ["TORCH_HOME"] = str(tc)

    return hf, tc


def audit_transformers_cache():
    print("\n===== DINO EXACT HISTORICAL LOCAL-CACHE AVAILABILITY =====")

    result = {
        "transformers_import": False,
        "local_files_only_load": False,
        "model_id": MODEL_ID,
        "hf_cache": str(DEFAULT_HF_CACHE),
        "torch_cache": str(DEFAULT_TORCH_CACHE),
        "cache_dir_explicitly_passed": True,
        "error": None,
    }

    try:
        hf_cache, torch_cache = configure_exact_historical_caches()

        import transformers
        from transformers import AutoImageProcessor, AutoModel

        result["transformers_import"] = True
        result["transformers_version"] = transformers.__version__

        print("transformers:", transformers.__version__)
        print("HF cache:", hf_cache)
        print("Torch cache:", torch_cache)
        print("local_files_only:", True)

        # IMPORTANT:
        # Frozen R10K2A/R10L3B did NOT rely on the user's default HF cache.
        # They always passed the project-local F-drive cache explicitly.
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=str(hf_cache),
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            cache_dir=str(hf_cache),
            local_files_only=True,
        )

        result["local_files_only_load"] = True
        result["model_class"] = model.__class__.__name__
        result["processor_class"] = processor.__class__.__name__
        result["parameter_count"] = int(
            sum(p.numel() for p in model.parameters())
        )
        result["hidden_size"] = int(
            getattr(model.config, "hidden_size", -1)
        )

        if result["hidden_size"] != 768:
            raise RuntimeError(
                f"Unexpected DINO hidden size: {result['hidden_size']}"
            )

        print("local project-cache load: PASS")
        print("processor:", result["processor_class"])
        print("model:", result["model_class"])
        print("hidden size:", result["hidden_size"])
        print("parameter count:", result["parameter_count"])

        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = repr(e)
        print("local project-cache load: FAIL")
        print("error:", repr(e))

    return result


def hardware_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
    }

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev)

        info.update({
            "cuda_device_index": int(dev),
            "gpu_name": torch.cuda.get_device_name(dev),
            "gpu_total_memory_bytes": int(
                prop.total_memory
            ),
            "gpu_compute_capability": (
                f"{prop.major}.{prop.minor}"
            ),
        })

    return info


def artifact_size_info(path: Path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "MiB": float(
            path.stat().st_size / (1024 ** 2)
        ),
    }


def audit_joblibs():
    print("\n===== FROZEN PCA / HEAD LOAD AUDIT =====")

    pca = joblib.load(PCA_PATH)
    head = joblib.load(HEAD_PATH)

    pca_info = {
        "class": pca.__class__.__name__,
        "n_components": int(
            getattr(pca, "n_components_", getattr(pca, "n_components", -1))
        ),
        "input_dim": int(
            getattr(pca, "n_features_in_", -1)
        ),
    }

    head_info = {
        "class": head.__class__.__name__,
    }

    # The frozen head is historically a sklearn pipeline.
    if hasattr(head, "named_steps"):
        head_info["named_steps"] = list(
            head.named_steps.keys()
        )
        if "lr" in head.named_steps:
            lr = head.named_steps["lr"]
            if hasattr(lr, "coef_"):
                head_info["lr_input_dim"] = int(
                    lr.coef_.shape[1]
                )

    print("PCA:", pca_info)
    print("HEAD:", head_info)

    if pca_info["n_components"] != 64:
        raise RuntimeError(
            "Frozen PCA component count changed."
        )

    return pca_info, head_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    print(
        "===== Q1 R15D0 RUNTIME / COMPUTATIONAL OVERHEAD PREFLIGHT ====="
    )
    print("STATUS=POST_FREEZE_RUNTIME_PREFLIGHT")
    print("MODEL_BENCHMARK_INFERENCE=NO")
    print("MODEL_FITTING=NO")
    print("SCORE_RECOMPUTATION=NO")
    print("GT_ACCESS=NO")
    print("OUTCOME_ACCESS=NO")
    print("TARGET_CALIBRATION=NO")
    print("METHOD_CHANGE=NO")
    print("MODEL_ID=", MODEL_ID)
    print("BENCHMARK_FRAMES=", BENCHMARK_FRAMES)
    print("DINO_BATCH_SIZES=", DINO_BATCH_SIZES)
    print("WARMUP_STEPS=", WARMUP_STEPS)
    print("TIMED_STEPS=", TIMED_STEPS)
    print("DOWNSTREAM_REPEATS=", DOWNSTREAM_REPEATS)
    print("FIX2_CHANGE=USE_EXACT_R10K2A_R10L3B_PROJECT_LOCAL_HF_CACHE")
    print("SCIENTIFIC_PROTOCOL_CHANGE_FROM_FIX1=NO")

    print("\n===== EXACT FROZEN LINEAGE =====")
    assert_file_sha(
        PCA_PATH,
        EXPECTED_PCA_SHA,
        "PCA",
    )
    assert_file_sha(
        HEAD_PATH,
        EXPECTED_HEAD_SHA,
        "HEAD",
    )
    assert_file_sha(
        THRESH_PATH,
        EXPECTED_THRESHOLD_SHA,
        "THRESHOLD",
    )
    assert_file_sha(
        R14C2B_LOCK,
        EXPECTED_R14C2B_LOCK_SHA,
        "R14C2B_LOCK",
    )
    assert_file_sha(
        R14C1_LOCK,
        EXPECTED_R14C1_LOCK_SHA,
        "R14C1_LOCK",
    )
    assert_file_sha(
        SUN_MANIFEST,
        EXPECTED_SUN_MANIFEST_SHA,
        "SUN_RUNTIME_MANIFEST",
    )
    assert_file_sha(
        AUTHORITATIVE_R10K2A_SCRIPT,
        EXPECTED_R10K2A_SCRIPT_SHA,
        "R10K2A_SCRIPT",
    )

    threshold = load_json(THRESH_PATH)
    # Accept historical JSON nesting, but verify the frozen number exists.
    threshold_text = json.dumps(threshold)
    if str(FROZEN_THRESHOLD) not in threshold_text:
        raise RuntimeError(
            "Frozen threshold value not present in threshold JSON."
        )

    pca_info, head_info = audit_joblibs()

    manifest = pd.read_csv(
        SUN_MANIFEST,
        low_memory=False,
    )
    sid_col, selected = choose_runtime_cohort(
        manifest
    )

    print("\n===== RUNTIME COHORT LOCK =====")
    print("manifest rows:", len(manifest))
    print("identifier column:", sid_col)
    print("selected frames:", len(selected))
    print(
        "selection rule: first N by "
        "SHA256(str(identifier)) ascending"
    )
    print("GT/outcomes used for selection: NO")

    if len(selected) < 100:
        raise RuntimeError(
            "Too few runtime cohort frames."
        )

    hw = hardware_info()

    print("\n===== HARDWARE / SOFTWARE =====")
    for k, v in hw.items():
        print(k, "=", v)

    if not hw["cuda_available"]:
        raise RuntimeError(
            "CUDA unavailable; GPU benchmark cannot proceed."
        )

    dino = audit_transformers_cache()
    if not dino["transformers_import"]:
        raise RuntimeError(
            "transformers unavailable."
        )
    if not dino["local_files_only_load"]:
        raise RuntimeError(
            "facebook/dinov2-base is not available from the exact historical "
            "project-local cache after explicit cache_dir use. "
            "Do not benchmark with network download in R15D1."
        )

    if args.output_dir.exists():
        raise FileExistsError(
            args.output_dir
        )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    cohort_path = (
        args.output_dir
        / "R15D0_RUNTIME_COHORT_256.csv"
    )

    # Save all original frozen-manifest columns for path resolution later,
    # plus the deterministic runtime identifier/hash.
    selected.to_csv(
        cohort_path,
        index=False,
    )

    lock = {
        "status": "PASS",
        "decision": DECISION,
        "script_version": VERSION,
        "build": BUILD,
        "scientific_role": (
            "post-freeze runtime/computational-overhead protocol preflight"
        ),
        "method": {
            "dino_model_id": MODEL_ID,
            "dino_input": "RGB bicubic 224x224",
            "patch_grid": "16x16",
            "conditioned_feature": (
                "FG768 + BG768 -> PCA64 -> append M2 -> frozen safety head"
            ),
            "frozen_threshold": FROZEN_THRESHOLD,
        },
        "runtime_cohort": {
            "source": str(SUN_MANIFEST),
            "source_sha256": EXPECTED_SUN_MANIFEST_SHA,
            "identifier_column": sid_col,
            "selection_rule": (
                "first N rows after ascending "
                "SHA256(str(identifier)), tie identifier"
            ),
            "requested_frames": BENCHMARK_FRAMES,
            "actual_frames": len(selected),
            "gt_or_outcome_used": False,
            "cohort_csv": str(cohort_path),
            "cohort_csv_sha256": sha256_file(
                cohort_path
            ),
        },
        "benchmark_protocol": {
            "device": "CUDA",
            "dino_batch_sizes": DINO_BATCH_SIZES,
            "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
            "downstream_repeats": DOWNSTREAM_REPEATS,
            "cuda_synchronize_before_after_timer": CUDA_SYNCHRONIZE,
            "report": [
                "median_ms_per_image",
                "mean_ms_per_image",
                "p95_ms_per_image",
                "images_per_second",
                "peak_cuda_memory_MiB",
                "serialized_artifact_MiB",
                "parameter_count",
            ],
            "source_segmentation_or_tta_timed": False,
            "reason": (
                "report incremental safety-module cost separately; "
                "segmentation/TTA cost is model/action specific"
            ),
        },
        "hardware": hw,
        "historical_cache": {
            "hf_cache": str(DEFAULT_HF_CACHE),
            "torch_cache": str(DEFAULT_TORCH_CACHE),
            "cache_dir_explicitly_passed": True,
        },
        "dino_local_cache": dino,
        "frozen_artifacts": {
            "pca": artifact_size_info(
                PCA_PATH
            ),
            "head": artifact_size_info(
                HEAD_PATH
            ),
            "threshold": artifact_size_info(
                THRESH_PATH
            ),
            "pca_schema": pca_info,
            "head_schema": head_info,
        },
        "information_boundary": {
            "benchmark_inference_performed": False,
            "model_fitting": False,
            "score_recomputation": False,
            "gt_access": False,
            "outcome_access": False,
            "target_calibration": False,
            "method_change": False,
        },
        "next_stage": (
            "R15D1_RUNTIME_COMPUTATIONAL_OVERHEAD_BENCHMARK"
        ),
    }

    lock_path = (
        args.output_dir
        / "R15D0_RUNTIME_OVERHEAD_PROTOCOL_LOCK.json"
    )
    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== R15D0 FINAL =====")
    print("Decision=", DECISION)
    print("COHORT=", cohort_path)
    print(
        "COHORT SHA256=",
        sha256_file(cohort_path),
    )
    print("LOCK=", lock_path)
    print(
        "LOCK SHA256=",
        sha256_file(lock_path),
    )
    print("PASS")


if __name__ == "__main__":
    main()
