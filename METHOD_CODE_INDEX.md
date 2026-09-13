# SafeTTA Method Code Index

This repository exposes **two complementary reproducibility layers**:

1. **Method implementation layer** — retained scripts implementing the paper-facing representation, safety estimator, TTA actions, external evaluation, MRI/PROMISE12 workflow, and R31-R33 action-transfer extension.
2. **Paper-statistic replay layer** — compact reviewer-facing replays that verify frozen manuscript-level numerical anchors from released outputs.

The original `v1.0-paper-v14` snapshot remains immutable. R31-R33 are additive and correspond to the current manuscript extension.

---

## Reviewer entry points

### 1. SOURCE-only prediction-conditioned representation

Primary representation implementation:

```text
method_core/representation/Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py
```

Morphology implementation:

```text
method_core/representation/Q1_R10J1_source_mask_morphology_feature_builder_fix1.py
```

This branch uses frozen DINOv2 patch features conditioned by the current SOURCE mask, followed by SOURCE-fitted PCA64 and two morphology descriptors to form the 66-D SOURCE safety state (`Q66`).

The SOURCE-only branch is strictly **pre-adaptation**.

---

### 2. Frozen SOURCE-only safety estimator

```text
method_core/safety_estimator/Q1_R10L0_final_source_safety_estimator_lock_fix3.py
```

Implements:

```text
median imputation
→ standardization
→ class-balanced logistic regression
```

Serialized PCA/head/threshold artifacts:

```text
artifacts/colonoscopy_safety_estimator/
```

This estimator underlies the frozen SOURCE-only analyses and the legacy public replay.

---

### 3. TTA action implementations

Reviewer-facing TTA implementations are grouped under:

```text
method_core/tta_actions/
experiments/tta_actions/
```

The final manuscript evaluates:

- `TENT1` — episodic one-step entropy minimization;
- `PL-CONF90` — confidence-thresholded pseudo-label adaptation;
- `MEMO-SEG4-1STEP` — four-view segmentation-compatible MEMO action used for the third-action / unseen-action extension;
- architecture-compatible MRI TENT1 behavior.

---

### 4. Transition-augmented pre-commit representation

The R31-R33 extension evaluates a transition representation constructed after a candidate action is executed on a reversible shadow copy:

```text
SOURCE state: Q66
candidate semantic state: 64-D
DeltaSemantic64 = candidate64 - SOURCE64
transition representation = [Q66 ; DeltaSemantic64] = 130-D
```

Important semantic boundary:

- SOURCE-only risk is **pre-adaptation**;
- transition-augmented risk is **pre-commit**, not strictly pre-update;
- no explicit ActionID is required by the primary shared transition predictor.

See:

```text
experiments/R31_action_transfer/
experiments/R32_validation/
experiments/R33_external_joint_shift/
```

---

## Validation workflows

### 5. External colonoscopy workflows

```text
experiments/polypgen/
experiments/sunseg/
```

These directories contain the retained manifest/prediction/safety-score/GT-reveal lineage used by the earlier SOURCE-only external evaluations.

---

### 6. R31 — action transfer

```text
experiments/R31_action_transfer/
```

Purpose:

- introduce MEMO as a third mechanistically distinct action;
- evaluate cross-action transfer rather than only cross-domain transfer;
- establish whether a shared harmful-update representation transfers to an unseen action.

Primary actions:

```text
TENT1
PL-CONF90
MEMO-SEG4-1STEP
```

The final interpretation is **action-transferable harmful-update structure**, not full action invariance.

Compact paper-facing outputs are under:

```text
outputs/R31_R33/
```

---

### 7. R31C/R31D — controller negative analysis

```text
experiments/R31_controller_negative/
```

These experiments test whether the ranking signal is sufficient to support a complete multi-action controller.

The pre-registered controller criteria were not met. Therefore the manuscript retains these experiments as a **negative feasibility diagnostic** and does not claim a validated automatic controller.

---

### 8. R32 — strict three-action LOAO validation

```text
experiments/R32_validation/
```

Protocol:

- DeepLabV3-R50 action-transfer panel;
- three actions: TENT1, PL-CONF90, MEMO;
- strict leave-one-action-out evaluation;
- frozen shared transition representation;
- no target calibration;
- no score reversal;
- no action-specific refitting for the primary shared model.

Compact result table:

```text
outputs/R31_R33/r32a_three_action_loao_summary.csv
```

Primary shared `Q66 + DeltaSemantic64` macro AUROC is approximately `0.7416`.

---

### 9. R33 — simultaneous domain + unseen-action shift

```text
experiments/R33_external_joint_shift/
```

Final locked direction:

```text
NeoPolyp TENT1 + PL-CONF90
            ↓
PolypGen MEMO-SEG4-1STEP
```

This combines:

- domain shift: NeoPolyp → PolypGen;
- unseen-action shift: TENT1/PL-CONF90 → MEMO.

Primary compact outputs:

```text
outputs/R31_R33/r33_polypgen_memo_primary_metrics.csv
outputs/R31_R33/r33_polypgen_memo_paired_deltas.csv
outputs/R31_R33/r33_claim_freeze_summary.json
```

Paper-facing SafeTTA result:

```text
AUROC = 0.743238
95% CI = [0.699569, 0.783784]
AUPRC = 0.255458
HARM prevalence = 0.065492
AUPRC lift = 3.90x
```

Claim boundary:

> external support for action-transferable future-HARM ranking under simultaneous domain and unseen-action shift.

A pristine prospective external-validation claim is not made.

---

### 10. MRI / PROMISE12 framework replication

```text
experiments/mri_prostate_promise/
```

Includes retained preprocessing/model protocol, source OOF segmentation, TENT1 outcome generation, MRI safety-estimator fitting, PROMISE12 pre-GT prediction/scoring, operating-point transport, locked-policy evaluation, external ranking analysis, and matched-coverage utility analysis.

This is a **framework replication** of the SOURCE-only design on a different task/modality. It is not a zero-shot transfer of the colonoscopy safety head to MRI.

---

### 11. Paper analyses

```text
experiments/paper_analysis/
```

Contains retained analyses for:

- SOURCE-side ablation;
- paired clustered bootstrap;
- selective-adaptation utility;
- runtime;
- HARM-margin sensitivity;
- reliability-baseline comparisons;
- legacy 44-anchor replay support.

---

## Public replay entry points

### Legacy frozen primary replay

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root .
```

Expected:

```text
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
```

### R31-R33 compact manuscript replay

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Expected:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

---

## Provenance and claim boundaries

Legacy publication provenance:

```text
provenance/R17A/
```

R31-R33 provenance:

```text
provenance/R31_R33/
```

Important files include:

```text
CLAIM_BOUNDARIES.md
REPRODUCIBILITY_SCOPE.md
PUBLIC_REPLAY_REPORT.txt
```

These records are part of the public scientific-scope guardrail and should be checked before manuscript wording is changed.

---

## Historical retained code

```text
legacy_all_retained_code/
```

contains SafeTTA-related historical scripts retained by the earlier code audit. These files are preserved for provenance/completeness but are **not** the preferred reviewer entry points.

Use `method_core/`, grouped `experiments/`, compact `outputs/`, and the public replay scripts for the current paper-facing workflow.

---

## Upstream segmentation-training boundary

Retained source-training implementations are available under:

```text
experiments/source_training/
```

Complete per-image upstream training manifests/checkpoints were not recoverable for every historical NeoPolyp segmentation state. Therefore the repository does **not** claim bit-identical retraining of every historical upstream segmenter.

The supported public reproducibility claim is deterministic verification of the released SafeTTA paper-level numerical anchors from frozen intermediate artifacts and compact audit outputs.
