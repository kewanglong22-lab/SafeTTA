# Q1-R17A Published Reliability Baseline Comparison — Frozen Protocol v1

**Status:** PRE-REGISTERED BEFORE NEW TARGET-SCORE GENERATION

## Scientific question
For a frozen SOURCE segmentation state and an already-frozen TTA action, can an adaptation-free score computed **before the update** rank samples that later satisfy:

HARM = I[DeltaDice <= -0.02]?

This experiment compares **pre-adaptation scoring signals**, not full TTA algorithms.

## Baselines

### B1. TEGDA-ADIC
Use only the published Agreement with Dropout Inferences calibrated by Confidence (ADIC) score.

- Deterministic SOURCE probability map P.
- M=10 dropout inferences.
- Dropout rate 0.5 only when a faithful architecture-specific implementation can be established.
- ADI = mean class-wise soft Dice agreement between P and dropout predictions.
- b = 1 - Eavg / log(C).
- ADIC q = b * ADI.
- Larger ADIC means better current predicted quality, so HARM-risk orientation is -ADIC (or any strictly monotonic equivalent).
- No AFFR, teacher update, feature bank, or TEGDA adaptation is run.

### B2. SicTTA-CCD
Use the published Class Compact Density (CCD) score from the frozen SOURCE softmax map.

- Reshape p to C x N.
- Compute class similarity p p^T.
- Apply the published column-wise softmax.
- Compute entropy of the normalized C x C matrix.
- Lower CCD is more source-friendly/compact, therefore use CCD directly as the HARM-risk direction.
- One SOURCE forward pass only. No adaptation and no target fitting.

### B3. MC-dropout uncertainty
If ADIC is feasible, reuse the exact same 10 dropout passes to report predictive entropy of the mean dropout prediction as a standard uncertainty baseline.

## Not included as a primary baseline
Do not report a hand-made "CertainTTA-like" score. The exact CertainTTA method requires a variational probabilistic source model and its adaptability score is tied to its prompt-optimization procedure. Include it only if an exact architecture-compatible implementation can be reproduced faithfully.

## Frozen evaluation matrix
Primary external evaluation:
- PolypGen / TENT1
- SUN-SEG / TENT1
- SUN-SEG / PL-CONF90
- PROMISE12 / TENT1

Source/development support:
- NeoPolyp
- Prostate158

Do not add another dataset.

## No-leakage rules
1. SafeTTA remains frozen.
2. Segmentation weights remain frozen during score generation.
3. Existing TTA outcomes remain unchanged.
4. No target GT may select score orientation, dropout parameters, layers, thresholds, calibration, or hyperparameters.
5. Target scores must be serialized and hashed before GT is joined.
6. Any operating threshold must be selected on source development only.
7. Binary segmentation uses [1-p, p] when a published metric needs C classes.

## Metrics
Primary:
- HARM AUROC
- HARM AUPRC
- paired clustered/bootstrap Delta AUROC and Delta AUPRC versus frozen SafeTTA
- 95% CI using the same physical grouping unit as the manuscript

Secondary:
- recall
- FPR
- PPV at source-selected threshold

## Manuscript-facing table
| Pre-adaptation score | NeoPolyp | PolypGen/TENT1 | SUN/TENT1 | SUN/PL-CONF90 | PROMISE12/TENT1 |
|---|---:|---:|---:|---:|---:|
| Entropy | existing | existing | existing | existing | existing |
| DINO CLS | existing | existing | existing | existing | existing |
| SicTTA-CCD | new | new | new | new | new |
| TEGDA-ADIC | new | new | new | new | new |
| MC-dropout uncertainty | new | new | new | new | new |
| SafeTTA | frozen | frozen | frozen | frozen | frozen |

Primary cell = HARM AUROC. AUPRC and paired CIs go to supplement.

## Interpretation
The comparison tests whether published pre-adaptation prediction-quality / uncertainty signals are sufficient for the different endpoint of future action-induced HARM.

Do not claim reproduction of the full TEGDA or SicTTA adaptation frameworks.

## Stop/go rule
- GO if CCD + ADIC can both be implemented faithfully.
- GO with two baselines only; a third published method is not mandatory.
- If faithful ADIC cannot be established, STOP rather than invent an approximation.
- Keep R16-A2 as the learned/current-quality control.

## Asset preservation during R17A
Until this experiment is closed, do not delete:
- S06_C PolypGen locked prediction/logit assets
- S05_D / S01_D1 source/TTA logit assets
- final SOURCE segmentation checkpoints
- PolypGen, SUN-SEG, Prostate158, PROMISE12 images/manifests
- frozen HARM/outcome tables
