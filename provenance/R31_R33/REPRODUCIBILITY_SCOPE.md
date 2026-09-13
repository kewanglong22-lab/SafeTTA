# R31-R33 Reproducibility Scope

This document defines what is released, what is replayable, and what is intentionally not claimed as a complete rerun package.

## Purpose

The R31-R33 package is an additive update after the frozen `v1.0-paper-v14` SafeTTA release.

It provides a reviewer-oriented path for verifying the new manuscript claims without requiring redistribution of private or large assets.

## Current compact public layer

The current PR provides:

1. reviewer-facing R31/R32/R33 protocol documentation;
2. compact manuscript-facing result tables;
3. claim-boundary and figure/table provenance records;
4. deterministic paper-statistic replay for released R32/R33 numerical anchors;
5. release-checklist documentation.

The compact public layer is sufficient for **paper-statistic reproducibility from frozen released outputs**.

## Full execution-lineage layer

A separate method-audit layer consists of the exact final retained R31-R33 execution scripts.

The complete expected final set contains **22 scripts** across:

```text
experiments/R31_action_transfer/
experiments/R31_controller_negative/
experiments/R32_validation/
experiments/R33_external_joint_shift/
```

At the current PR state, those exact scripts are **not yet all synchronized into the repository**. Therefore the repository should not yet be described as containing the complete R31-R33 execution lineage.

This distinction is deliberate:

- compact paper-statistic replay: currently available;
- full R31-R33 execution-script auditability: pending script synchronization.

## Public replay scope

The recommended verification path is:

```text
code/Q1_R31_R33_public_paper_stat_replay_v1.py
```

The replay verifies frozen numerical anchors for:

- R32 three-action LOAO summary;
- R33 PolypGen/MEMO primary metrics;
- R33 paired baseline comparisons;
- R33 claim-lock consistency.

The replay uses exact-value checks with a fixed absolute tolerance and fails if a released compact table or claim lock differs from the frozen anchors.

## Not claimed

This repository does not claim that every historical heavy experiment can be retrained from zero with identical floating-point outputs.

Reasons include:

- historical segmentation checkpoints are not all redistributed;
- datasets are not redistributed;
- DINOv2 weights are not redistributed;
- target ground-truth masks are not redistributed;
- some historical experiments depend on frozen intermediate artifacts;
- wall-clock behavior is hardware dependent.

## Inference and artifact boundary

Large or externally governed assets remain external:

```text
- datasets
- model checkpoints
- pretrained foundation-model weights
- target ground truth
- large cached inference tensors
```

Small deterministic public artifacts are retained:

```text
- compact CSV summaries
- claim locks
- README/protocol documentation
- SHA / provenance records where released
- public replay outputs
```

## SOURCE-only and transition scope

Two endpoints are intentionally separated.

### SOURCE-only

```text
Q66
```

This is a strictly pre-adaptation safety estimate.

### Transition-augmented

```text
Q66 + DeltaSemantic64
```

This is a pre-commit estimate after reversible candidate execution.

The transition-augmented mode must not be described as strictly pre-update.

The SOURCE-only PolypGen/TENT1 endpoint and the later R33 PolypGen/MEMO transition endpoint must not be numerically conflated.

## MRI replication scope

The MRI/PROMISE12 study is a framework replication of the SOURCE-only design.

It demonstrates portability of the SafeTTA design principle across task/modality shift, but it is not a zero-shot transfer of the colonoscopy safety head to MRI.

## Final release condition

Before advertising a new manuscript tag, choose one of two explicit public-release scopes:

### Scope A — compact reproducibility release

The manuscript states that the repository reproduces frozen paper-level numerical anchors from compact outputs.

### Scope B — compact replay + full R31-R33 script audit

In addition to Scope A, synchronize and verify all 22 final retained R31-R33 scripts (plus any deliberately retained SHA-bound mechanical ancestors if claimed).

Do not describe Scope A as Scope B.
