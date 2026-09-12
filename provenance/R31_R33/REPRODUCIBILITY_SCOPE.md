# R31-R33 Reproducibility Scope

This document defines what is released, what is replayable, and what is intentionally not claimed as a complete rerun package.

## Purpose

The R31-R33 package is an additive update after the frozen `v1.0-paper-v14` SafeTTA release.

It provides a reviewer-oriented path for verifying the new manuscript claims without requiring redistribution of private/large assets.

## Released components

The package provides:

1. retained R31-R33 protocol scripts;
2. compact manuscript-facing result tables;
3. frozen manifests and SHA records;
4. claim-boundary documentation;
5. deterministic paper-statistic replay for released numerical anchors.

## Public replay scope

The recommended verification path is:

```text
code/Q1_R31_R33_public_paper_stat_replay_v1.py
```

The replay verifies:

- R32 three-action LOAO summary anchors;
- R33 PolypGen/MEMO primary metrics;
- paired baseline comparisons;
- claim-freeze consistency.

## Not claimed

This repository does not claim that every historical heavy experiment can be retrained from zero with identical floating-point outputs.

Reasons:

- historical segmentation checkpoints are not all redistributed;
- datasets are not redistributed;
- DINOv2 weights are not redistributed;
- target ground truth masks are not redistributed;
- some historical experiments depend on frozen intermediate artifacts.

## Inference and artifact boundary

Large artifacts remain external:

```text
- datasets
- model checkpoints
- pretrained foundation-model weights
- private/local cached assets
```

Small deterministic metadata are retained:

```text
- manifests
- SHA256 records
- compact CSV summaries
- protocol locks
- paper replay outputs
```

## Historical experiment scripts

Historical scripts are retained mainly for provenance.

Reviewer-facing entry points are:

```text
method_core/
experiments/R31_action_transfer/
experiments/R32_validation/
experiments/R33_external_joint_shift/
code/*public*replay*
```

## SOURCE-only and transition scope

Two endpoints are intentionally separated.

### SOURCE-only

```text
Q66
```

A pre-adaptation safety estimate.

### Transition-augmented

```text
Q66 + DeltaSemantic64
```

A pre-commit estimate after reversible candidate execution.

These endpoints should not be treated as the same method or merged into one numerical claim.

## MRI replication scope

The MRI/PROMISE12 study is released as a framework replication.

It demonstrates portability of the SafeTTA design principle across task/modality shift, but it is not a zero-shot transfer experiment.
