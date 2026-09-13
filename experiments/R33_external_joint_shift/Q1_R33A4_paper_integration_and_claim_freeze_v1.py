#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA R33A4
Paper Integration + Claim Freeze

This stage performs NO new experiment and NO new statistics.
It only SHA-verifies frozen R33A3 outputs and converts them into
paper-ready numeric anchors, claim boundaries, Results text, and LaTeX.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


VERSION = "2026-09-12-R33A4-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

R33A3_SCRIPT = (
    CODE / "Q1_R33A3_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1.py"
)
EXPECTED_R33A3_SCRIPT_SHA256 = (
    "3229a63cd1669cbe5b68e87c35f8a4431f8e29294fa9f01a9c079d6a762f20a3"
)

R33A3_DIR = (
    ROOT
    / "R33A3_first_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1"
)
R33A3_FINAL = R33A3_DIR / "R33A3_FINAL_LOCK.json"
EXPECTED_R33A3_FINAL_SHA256 = (
    "9ef2fa17bb9481ca65a90f8066f80996fd3b9b21a6f86d233b8f87728a9a44ee"
)

ANALYSIS_LOCK = R33A3_DIR / "R33A3_PRE_NEW_MEMO_OUTCOME_ANALYSIS_LOCK.json"
PARITY = R33A3_DIR / "R33A3_HISTORICAL_SOURCE_DICE_PARITY.json"
POOLED = R33A3_DIR / "R33A3_PRIMARY_POOLED_METHOD_METRICS.csv"
STATE = R33A3_DIR / "R33A3_STATE_SECONDARY_METRICS.csv"
MACRO = R33A3_DIR / "R33A3_MACRO_STATE_SECONDARY_METRICS.csv"
SEG = R33A3_DIR / "R33A3_SEGMENTATION_OUTCOME_SUMMARY.csv"
BOOT = R33A3_DIR / "R33A3_PHYSICAL_CASE_CLUSTERED_BOOTSTRAP.csv"
PAIRED = R33A3_DIR / "R33A3_PAIRED_PUBLISHED_BASELINE_DELTAS.csv"
GATE = R33A3_DIR / "R33A3_EXTERNAL_JOINT_SHIFT_CONFIRMATION_GATE.json"

SAFE = "SafeTTA-Q66+dSemantic64"
DEFAULT_OUT = ROOT / "R33A4_paper_integration_and_claim_freeze_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch\nexpected={expected}\nobserved={got}\npath={path}"
        )
    return got


def atomic_json(path: Path, payload: Any):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def get_one(df: pd.DataFrame, **conds) -> pd.Series:
    g = df
    for key, value in conds.items():
        g = g[g[key] == value]
    if len(g) != 1:
        raise RuntimeError(f"Expected one row for {conds}; got={len(g)}")
    return g.iloc[0]


def verify_chain() -> Dict[str, Any]:
    require_sha(R33A3_SCRIPT, EXPECTED_R33A3_SCRIPT_SHA256, "R33A3 script")
    require_sha(R33A3_FINAL, EXPECTED_R33A3_FINAL_SHA256, "R33A3 final")

    final = json.loads(R33A3_FINAL.read_text(encoding="utf-8"))

    if final.get("status") != (
        "PASS_R33A3_POLYPGEN_MEMO_EXTERNAL_JOINT_SHIFT_EVALUATION_COMPLETE"
    ):
        raise RuntimeError(f"R33A3 status drift={final.get('status')}")
    if final.get("decision") != "EXTERNAL_JOINT_DOMAIN_ACTION_SHIFT_CONFIRMED":
        raise RuntimeError(f"R33A3 decision drift={final.get('decision')}")
    if final.get("joint_confirmation") is not True:
        raise RuntimeError("R33A3 joint confirmation is no longer True.")

    for key in [
        "model_inference",
        "TTA_rerun",
        "predictor_refit",
        "target_calibration",
        "score_reversal",
        "target_subset_selection",
    ]:
        if final.get(key) is not False:
            raise RuntimeError(f"R33A3 guard drift: {key}={final.get(key)}")

    artifact_sha = {
        ANALYSIS_LOCK: final["pre_outcome_analysis_lock_sha256"],
        PARITY: final["historical_source_dice_parity_sha256"],
        POOLED: final["pooled_metrics_sha256"],
        STATE: final["state_metrics_sha256"],
        MACRO: final["macro_state_metrics_sha256"],
        SEG: final["segmentation_summary_sha256"],
        BOOT: final["bootstrap_sha256"],
        PAIRED: final["paired_baseline_delta_sha256"],
        GATE: final["gate_sha256"],
    }
    for path, expected in artifact_sha.items():
        require_sha(path, str(expected), path.name)

    return final


def main() -> int:
    print("=" * 120)
    print("SafeTTA R33A4 Paper Integration + Claim Freeze")
    print("Version                :", VERSION)
    print("New experiment         : NO")
    print("New statistics         : NO")
    print("Model/predictor change : NO")
    print("Target calibration     : NO")
    print("=" * 120)

    final = verify_chain()

    pooled = pd.read_csv(POOLED, low_memory=False)
    seg = pd.read_csv(SEG, low_memory=False)
    boot = pd.read_csv(BOOT, low_memory=False)
    paired = pd.read_csv(PAIRED, low_memory=False)

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_LOCK.read_text(encoding="utf-8"))

    safe = get_one(pooled, method=SAFE)
    seg_pool = get_one(seg, scope="POOLED")

    safe_auc_b = get_one(boot, method=SAFE, metric="AUROC")
    safe_ap_b = get_one(boot, method=SAFE, metric="AUPRC")
    safe_apm_b = get_one(
        boot,
        method=SAFE,
        metric="AUPRC_MINUS_PREVALENCE",
    )

    def paired_row(comp: str, metric: str):
        return get_one(paired, comparator=comp, metric=metric)

    ccd_auc = paired_row("SicTTA-CCD", "AUROC")
    ccd_ap = paired_row("SicTTA-CCD", "AUPRC")
    adic_auc = paired_row("TEGDA-ADIC", "AUROC")
    adic_ap = paired_row("TEGDA-ADIC", "AUPRC")
    mc_auc = paired_row("MC-dropout", "AUROC")
    mc_ap = paired_row("MC-dropout", "AUPRC")

    anchors = {
        "SafeTTA_AUROC": float(safe["AUROC"]),
        "SafeTTA_AUPRC": float(safe["AUPRC"]),
        "HARM_prevalence": float(safe["HARM_PREVALENCE"]),
        "AUPRC_minus_prevalence": float(safe["AUPRC_MINUS_PREVALENCE"]),
        "AUPRC_lift": float(safe["AUPRC_LIFT"]),
        "SafeTTA_AUROC_CI95": [
            float(safe_auc_b["ci95_low"]),
            float(safe_auc_b["ci95_high"]),
        ],
        "SafeTTA_AUPRC_CI95": [
            float(safe_ap_b["ci95_low"]),
            float(safe_ap_b["ci95_high"]),
        ],
        "SafeTTA_AUPRC_minus_prevalence_CI95": [
            float(safe_apm_b["ci95_low"]),
            float(safe_apm_b["ci95_high"]),
        ],
        "SOURCE_Dice_mean": float(seg_pool["source_dice_mean"]),
        "MEMO_Dice_mean": float(seg_pool["memo_dice_mean"]),
        "MEMO_DeltaDice_mean": float(seg_pool["memo_delta_dice_mean"]),
        "HARM_rows": int(seg_pool["harm_rows"]),
        "NEUTRAL_rows": int(seg_pool["neutral_rows"]),
        "BENEFIT_rows": int(seg_pool["benefit_rows"]),
        "BENEFIT_prevalence": float(seg_pool["benefit_prevalence"]),
        "SOURCE_Dice_parity_mismatch_rows": int(parity["mismatch_rows"]),
        "SOURCE_Dice_parity_max_abs": float(parity["max_abs_difference"]),
        "SafeTTA_minus_CCD_AUROC": {
            "delta": float(ccd_auc["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(ccd_auc["ci95_low"]), float(ccd_auc["ci95_high"])],
        },
        "SafeTTA_minus_CCD_AUPRC": {
            "delta": float(ccd_ap["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(ccd_ap["ci95_low"]), float(ccd_ap["ci95_high"])],
        },
        "SafeTTA_minus_ADIC_AUROC": {
            "delta": float(adic_auc["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(adic_auc["ci95_low"]), float(adic_auc["ci95_high"])],
        },
        "SafeTTA_minus_ADIC_AUPRC": {
            "delta": float(adic_ap["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(adic_ap["ci95_low"]), float(adic_ap["ci95_high"])],
        },
        "SafeTTA_minus_MC_AUROC": {
            "delta": float(mc_auc["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(mc_auc["ci95_low"]), float(mc_auc["ci95_high"])],
        },
        "SafeTTA_minus_MC_AUPRC": {
            "delta": float(mc_ap["delta_SafeTTA_minus_comparator"]),
            "ci95": [float(mc_ap["ci95_low"]), float(mc_ap["ci95_high"])],
        },
    }

    # Freeze exact scientific interpretation.
    if anchors["SafeTTA_AUROC_CI95"][0] <= 0.5:
        raise RuntimeError("R33A1 criterion 1 no longer passes.")
    if anchors["SafeTTA_AUPRC_minus_prevalence_CI95"][0] <= 0.0:
        raise RuntimeError("R33A1 criterion 2 no longer passes.")
    if gate.get("joint_confirmation") is not True:
        raise RuntimeError("R33A3 gate no longer confirms joint shift.")
    if anchors["SOURCE_Dice_parity_mismatch_rows"] != 0:
        raise RuntimeError("Historical SOURCE Dice parity no longer passes.")

    allowed_claims = [
        "SafeTTA supports action-transferable future-HARM ranking under strict leave-one-action-out evaluation.",
        "A NeoPolyp TENT1+PL predictor retained useful ranking on PolypGen MEMO under simultaneous domain and unseen-action shift.",
        "The pre-specified external joint-shift confirmation criterion was satisfied.",
        "On PolypGen MEMO, SafeTTA significantly exceeded SicTTA-CCD and MC-dropout in both AUROC and AUPRC.",
        "Versus TEGDA-ADIC, SafeTTA had statistically indistinguishable AUROC and significantly higher AUPRC.",
        "Use 'action-transferable' or 'partially action-invariant'; do not use 'fully invariant'.",
    ]

    forbidden_claims = [
        "Do not call R33 pristine prospective external validation.",
        "Do not claim PolypGen GT was first accessed in R33.",
        "Do not claim SafeTTA has significantly higher AUROC than TEGDA-ADIC.",
        "Do not claim SafeTTA is uniformly best for every metric, action, or domain.",
        "Do not claim fully action-invariant or fully domain-invariant prediction.",
        "Do not claim a successful risk-controlled multi-action controller.",
        "Do not claim MEMO consistently improves segmentation across states.",
        "Do not perform post-hoc PolypGen calibration or method redesign from R33 outcomes.",
    ]

    english = f"""External joint domain-and-action shift.
The safety predictor was fit only on NeoPolyp outcomes from TENT1 and PL-CONF90 and
was applied, without target refitting, calibration, score reversal, or ActionID, to
a newly generated MEMO-SEG4-1STEP evaluation on PolypGen (1,532 images, three
DeepLabV3-R50 states; 4,596 model-cases). Under the pre-specified DeltaDice <= -0.02
definition, MEMO HARM prevalence was {100*anchors['HARM_prevalence']:.2f}%.
SafeTTA achieved AUROC={anchors['SafeTTA_AUROC']:.3f}
(95% physical-case clustered-bootstrap CI
[{anchors['SafeTTA_AUROC_CI95'][0]:.3f}, {anchors['SafeTTA_AUROC_CI95'][1]:.3f}])
and AUPRC={anchors['SafeTTA_AUPRC']:.3f}
(95% CI [{anchors['SafeTTA_AUPRC_CI95'][0]:.3f},
{anchors['SafeTTA_AUPRC_CI95'][1]:.3f}]), corresponding to a
{anchors['AUPRC_lift']:.2f}x lift over HARM prevalence. The 95% CI for
AUPRC minus prevalence was
[{anchors['SafeTTA_AUPRC_minus_prevalence_CI95'][0]:.3f},
{anchors['SafeTTA_AUPRC_minus_prevalence_CI95'][1]:.3f}]. Both pre-specified
confirmation criteria were satisfied, supporting transfer of the harmful-update
ranking under simultaneous domain and unseen-action shift. Because PolypGen ground
truth had been accessed in earlier experiments, this is a protocol-locked new-action
evaluation on a historically accessed external cohort rather than pristine
prospective external validation.

On the same PolypGen MEMO endpoint, SafeTTA exceeded SicTTA-CCD and MC-dropout in
both AUROC and AUPRC. Relative to TEGDA-ADIC, the AUROC difference was
{anchors['SafeTTA_minus_ADIC_AUROC']['delta']:+.3f}
(95% CI [{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][0]:+.3f},
{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][1]:+.3f}]), indicating no supported
AUROC difference, whereas the AUPRC difference was
{anchors['SafeTTA_minus_ADIC_AUPRC']['delta']:+.3f}
(95% CI [{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][0]:+.3f},
{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][1]:+.3f}]), favoring SafeTTA.
"""

    chinese = f"""外部域与未见动作联合偏移验证：
安全预测器仅使用 NeoPolyp 上 TENT1 与 PL-CONF90 的 14,400 行结果进行训练，
随后在不进行目标域重训练、校准、分数翻转且不输入 ActionID 的条件下，直接应用于
PolypGen 的 MEMO-SEG4-1STEP。测试包含 1,532 个图像、3 个 DeepLabV3-R50 状态，
共 4,596 个 model-case。按预先固定的 ΔDice≤-0.02 定义，MEMO HARM 比例为
{100*anchors['HARM_prevalence']:.2f}%。SafeTTA 的 AUROC 为
{anchors['SafeTTA_AUROC']:.3f}，95% 病例聚类 bootstrap CI 为
[{anchors['SafeTTA_AUROC_CI95'][0]:.3f}, {anchors['SafeTTA_AUROC_CI95'][1]:.3f}]；
AUPRC 为 {anchors['SafeTTA_AUPRC']:.3f}，95% CI 为
[{anchors['SafeTTA_AUPRC_CI95'][0]:.3f}, {anchors['SafeTTA_AUPRC_CI95'][1]:.3f}]，
相对 HARM 基线比例提升 {anchors['AUPRC_lift']:.2f} 倍。
AUPRC-HARM prevalence 的 95% CI 为
[{anchors['SafeTTA_AUPRC_minus_prevalence_CI95'][0]:.3f},
{anchors['SafeTTA_AUPRC_minus_prevalence_CI95'][1]:.3f}]。
因此两个预先冻结的确认条件均通过，支持 SafeTTA 在“域变化 + 未见 TTA 动作变化”
同时发生时仍保持有用的未来 harmful-update 排序能力。由于 PolypGen GT 在更早的
实验中已经被访问，这一结果应称为“历史已访问 GT 的外部队列上的协议锁定新动作验证”，
而不是完全前瞻性的外部验证。

在相同 PolypGen MEMO endpoint 下，SafeTTA 相比 SicTTA-CCD 和 MC-dropout 的
AUROC、AUPRC 均有统计支持的优势。相对 TEGDA-ADIC，AUROC 差值为
{anchors['SafeTTA_minus_ADIC_AUROC']['delta']:+.3f}，
95% CI [{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][0]:+.3f},
{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][1]:+.3f}]，不支持二者 AUROC 存在差异；
但 SafeTTA 的 AUPRC 高 {anchors['SafeTTA_minus_ADIC_AUPRC']['delta']:+.3f}，
95% CI [{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][0]:+.3f},
{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][1]:+.3f}]。
"""

    contribution = """Frozen contribution wording
1. Prediction of action-defined future harm before adaptation, rather than only current prediction reliability.
2. Prediction-conditioned source-state plus semantic-transition representation for harmful-update ranking.
3. Strict leave-one-action-out action transfer across TENT, pseudo-labeling, and MEMO, with protocol-locked external support under simultaneous domain and unseen-action shift.
4. Negative controller evidence showing that reliable ranking does not automatically imply a useful high-coverage multi-action controller.
"""

    # Compact LaTeX table from frozen pooled metrics.
    p = pooled[["method", "AUROC", "AUPRC", "HARM_PREVALENCE", "AUPRC_LIFT"]].copy()
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & AUROC & AUPRC & HARM prev. & AUPRC lift \\",
        r"\midrule",
    ]
    for _, row in p.iterrows():
        lines.append(
            f"{row['method']} & {row['AUROC']:.3f} & {row['AUPRC']:.3f} & "
            f"{100*row['HARM_PREVALENCE']:.2f}\\% & {row['AUPRC_LIFT']:.2f}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines) + "\n"

    out = DEFAULT_OUT.resolve()
    if out.exists():
        raise RuntimeError(f"Output exists; R33A4 will not overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    numeric_path = out / "R33A4_NUMERIC_ANCHORS.json"
    en_path = out / "R33A4_MANUSCRIPT_RESULTS_EN.txt"
    zh_path = out / "R33A4_MANUSCRIPT_RESULTS_ZH.txt"
    tex_path = out / "R33A4_EXTERNAL_JOINT_SHIFT_TABLE.tex"
    guard_path = out / "R33A4_ALLOWED_AND_FORBIDDEN_CLAIMS.txt"

    atomic_json(numeric_path, anchors)
    atomic_text(en_path, english)
    atomic_text(zh_path, chinese)
    atomic_text(tex_path, latex)
    atomic_text(
        guard_path,
        "ALLOWED CLAIMS\n==============\n"
        + "\n".join(f"- {x}" for x in allowed_claims)
        + "\n\nFORBIDDEN CLAIMS\n================\n"
        + "\n".join(f"- {x}" for x in forbidden_claims)
        + "\n\n"
        + contribution,
    )

    claim = {
        "status": "PASS_R33A4_PAPER_INTEGRATION_AND_CLAIM_FREEZE_COMPLETE",
        "version": VERSION,
        "R33A3_final_sha256": EXPECTED_R33A3_FINAL_SHA256,
        "R33A3_decision": final["decision"],
        "joint_confirmation": True,
        "historical_PolypGen_GT_previously_available": True,
        "pristine_prospective_external_claim_allowed": False,
        "primary_claim": (
            "Action-transferable future-HARM ranking with protocol-locked "
            "external support under simultaneous domain + unseen-action shift."
        ),
        "numeric_anchors": anchors,
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
        "analysis_lock_primary_scope": analysis["primary_scope"],
        "no_new_experiment": True,
        "no_new_statistics": True,
        "artifacts": {
            "numeric_anchors_sha256": sha256_file(numeric_path),
            "manuscript_results_en_sha256": sha256_file(en_path),
            "manuscript_results_zh_sha256": sha256_file(zh_path),
            "latex_table_sha256": sha256_file(tex_path),
            "claim_guard_sha256": sha256_file(guard_path),
        },
        "next": "PAPER_TEXT_TABLE_FIGURE_INTEGRATION",
    }
    claim_path = out / "R33A4_CLAIM_FREEZE.json"
    atomic_json(claim_path, claim)

    final_lock = {
        "status": "PASS_R33A4_PAPER_INTEGRATION_AND_CLAIM_FREEZE_COMPLETE",
        "version": VERSION,
        "R33A3_script_sha256": EXPECTED_R33A3_SCRIPT_SHA256,
        "R33A3_final_sha256": EXPECTED_R33A3_FINAL_SHA256,
        "claim_freeze_sha256": sha256_file(claim_path),
        "numeric_anchors_sha256": sha256_file(numeric_path),
        "results_en_sha256": sha256_file(en_path),
        "results_zh_sha256": sha256_file(zh_path),
        "latex_table_sha256": sha256_file(tex_path),
        "claim_guard_sha256": sha256_file(guard_path),
        "new_experiment": False,
        "new_statistics": False,
        "model_change": False,
        "target_calibration": False,
        "next": "PAPER_TEXT_TABLE_FIGURE_INTEGRATION",
    }
    final_path = out / "R33A4_FINAL_LOCK.json"
    atomic_json(final_path, final_lock)

    print("\nR33A4 CORE CLAIM")
    print("  Action-transferable future-HARM ranking with external support")
    print("  under simultaneous domain + unseen-action shift.")
    print("  Pristine prospective external claim: NO")

    print("\nR33A4 NUMERIC ANCHORS")
    print(f"  SafeTTA AUROC      : {anchors['SafeTTA_AUROC']:.6f}")
    print(
        "  AUROC 95% CI       : "
        f"[{anchors['SafeTTA_AUROC_CI95'][0]:.6f}, "
        f"{anchors['SafeTTA_AUROC_CI95'][1]:.6f}]"
    )
    print(f"  SafeTTA AUPRC      : {anchors['SafeTTA_AUPRC']:.6f}")
    print(f"  HARM prevalence    : {anchors['HARM_prevalence']:.6f}")
    print(f"  AUPRC lift         : {anchors['AUPRC_lift']:.6f}x")
    print(
        "  SafeTTA-ADIC AUROC : "
        f"{anchors['SafeTTA_minus_ADIC_AUROC']['delta']:+.6f} "
        f"[{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][0]:+.6f}, "
        f"{anchors['SafeTTA_minus_ADIC_AUROC']['ci95'][1]:+.6f}]"
    )
    print(
        "  SafeTTA-ADIC AUPRC : "
        f"{anchors['SafeTTA_minus_ADIC_AUPRC']['delta']:+.6f} "
        f"[{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][0]:+.6f}, "
        f"{anchors['SafeTTA_minus_ADIC_AUPRC']['ci95'][1]:+.6f}]"
    )

    print("\nFINAL STATUS : PASS_R33A4_PAPER_INTEGRATION_AND_CLAIM_FREEZE_COMPLETE")
    print("New experiment    : NO")
    print("New statistics    : NO")
    print("Final lock SHA256 :", sha256_file(final_path))
    print("Output            :", out)
    print("NEXT              : PAPER_TEXT_TABLE_FIGURE_INTEGRATION")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
