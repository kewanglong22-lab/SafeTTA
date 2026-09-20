# SafeTTA

**SafeTTA: Prediction-Conditioned Safety Ranking for Medical Test-Time Adaptation under Domain, Action, and Task Shifts**

SafeTTA asks a deployment-oriented question: after a specified test-time adaptation candidate has been produced, should that candidate replace the frozen SOURCE prediction?

## Current manuscript method: Final68

```text
Final68 = Geometry4 + Transition64
```

- **Geometry4**: SOURCE-to-candidate prediction Dice, IoU, foreground-area change, and boundary-density change.
- **Transition64**: candidate-conditioned minus SOURCE-conditioned semantic PCA64 representation derived from frozen DINOv2 patch features.
- **Controller**: class-balanced tri-state logistic regression predicting HARM / NEUTRAL / BENEFIT.
- **Utility**: `u = P(BENEFIT) - P(HARM)`.
- **Decision**: accept candidate if `u > 0`; otherwise retain SOURCE.
- **Timing**: pre-commit, not strictly pre-update. The candidate prediction exists before the safety decision, but SOURCE remains recoverable.

Final68 deliberately contains candidate-induced transition information rather than an absolute SOURCE-state branch. No explicit ActionID is supplied to the frozen controller.

## Corrected v7 public reproducibility layer

The active compact paper-facing layer is under:

```text
reproducibility/final68_20260919/
```

Reviewer-facing replay:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected gate:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

The replay now hashes the **actual controller file bytes** before validating the frozen SHA256.

Exact controller:

```text
artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
```

Expected actual-file SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

## Current evidence line

### External PolypGen

| Metric | Final68 |
|---|---:|
| HARM-vs-rest AUROC | 0.5798 |
| HARM-vs-rest AUPRC | 0.2683 |
| HARM-vs-BENEFIT AUROC | 0.6886 |
| HARM-vs-BENEFIT AUPRC | 0.5338 |
| SOURCE Dice | 0.75490 |
| Deployed Dice | 0.75692 |
| Deployed − SOURCE | +0.00202 |

The frozen PolypGen operating-point deployment improvement is positive but **not statistically significant**.

### Primary grouped image-disjoint action shift

The primary action-shift analysis is the grouped physical-image-disjoint held-out-action protocol. Historical E4C is retained only as secondary provenance and is not called strict LOAO.

| Held-out action | HARM-vs-BENEFIT AUROC | 95% CI | Deployed − SOURCE |
|---|---:|---:|---:|
| TENT1 | 0.817440 | [0.784944, 0.846591] | +0.005709 [-0.002026, 0.013128] |
| PL-CONF90 | 0.729658 | [0.679815, 0.777933] | +0.011781 [0.008599, 0.015475] |
| MEMO | 0.735553 | [0.687708, 0.783476] | N/A |

The TENT1 deployment interval crosses zero. Absolute MEMO deployment is unavailable in the frozen primary lineage.

### Endpoint sensitivity

The frozen primary HARM/NEUTRAL/BENEFIT Dice margin is `±0.02`. Margins `±0.01`, `±0.03`, and `±0.05` are sensitivity analyses only.

### PROMISE12 prostate-MRI replication

| Metric | Final68 |
|---|---:|
| HARM-vs-rest AUROC | 0.6802 |
| HARM-vs-BENEFIT AUROC | 0.7708 |
| SOURCE Dice | 0.74525 |
| Deployed Dice | 0.75135 |
| Deployed − SOURCE | +0.00610 |

PROMISE12 is an **independently re-fitted framework replication**, not zero-shot transfer of the colonoscopy controller. The deployment endpoint is panel-level with patient-clustered inference and is not presented as patient-level pooled 3-D Dice improvement.

### Frozen score-level comparator sensitivity

The PolypGen comparator panel uses pre-frozen risk scores under a common 50% accept/rollback budget. It compares score ordering and deployment consequences; it is **not** a harmonized end-to-end reproduction of the original QCResUNet, TEGDA, SicTTA, or MC-dropout pipelines.

## Controller-only overhead

Once Final68 is already available:

```text
mean single-case CPU latency: 0.0869 ms
serialized controller size:   5,188 bytes
```

These numbers describe the **controller only**. They exclude candidate TTA execution, segmentation inference, DINOv2 extraction, Geometry4, and Transition64 construction.

## Public-asset exclusions

Two previously published figure-data files are intentionally excluded from the corrected active evidence layer because their provenance is not independently sufficient for the current manuscript claims:

```text
fig2_coverage_utility_frozen_points.csv
fig3_action_shift_loao_frozen_points.csv
```

The corrected action-shift figure data are provided instead as:

```text
reproducibility/final68_20260919/figure_data/fig2_e4d_action_shift_frozen_points.csv
```

## Historical releases

Historical tags remain immutable. In particular, `v2.0-paper-final68` remains the pre-correction Final68 public snapshot and resolves to commit:

```text
84c907bcd91de228c5f480659f1d9c29e39d0195
```

Earlier 66-D / 70-D / 130-D representations, old R31-R33 transition analyses, and historical E4C results are scientifically distinct from the current Final68 primary evidence line.

The old PolypGen 130-D full-transition result `0.7432 / 0.2555` is historical provenance and is not a current Final68 result.

## Claim boundaries

See:

```text
reproducibility/final68_20260919/CLAIM_BOUNDARIES.md
```

Key boundaries include:

- do not claim Full68 has the best global HARM AUROC;
- do not claim significant improvement at the frozen PolypGen operating point;
- do not claim uniform superiority over QCResUNet;
- do not treat the comparator table as an end-to-end method reproduction;
- do not call historical E4C the primary strict LOAO result;
- do not report primary absolute MEMO deployment where the frozen paired lineage is incomplete;
- do not call PROMISE12 zero-shot controller transfer;
- do not call `0.087 ms` end-to-end latency;
- do not interpret `u=0` as a clinically calibrated optimum.

## Reproducibility scope

The repository supports deterministic verification of released paper-level numerical anchors from frozen compact evidence and SHA-bound artifacts.

It does not claim bit-identical retraining of every historical segmentation checkpoint, redistribution of raw datasets/ground truth/third-party DINOv2 weights, or bit-identical wall-clock runtime across hardware.

## License

Original SafeTTA materials that the authors are entitled to license are released under Apache License 2.0. Third-party models, datasets, and software retain their original terms.
