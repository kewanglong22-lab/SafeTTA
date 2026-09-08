# SafeTTA R16 Reviewer-Defense Experiment Preregistration v1

**Protocol ID:** Q1-R16-2026-09-08-v1
**Status:** frozen before R16 external-result inspection
**Purpose:** address the two highest-value reviewer concerns identified in the MIA mock review without changing the frozen SafeTTA method or its primary endpoints.

## 1. Scientific questions

### R16-A — Is SafeTTA merely a current-segmentation-quality detector?

Post-reveal explanatory analysis asks whether the already frozen SafeTTA risk score remains associated with adaptation harm after accounting for the current SOURCE segmentation quality.

This is **not** a deployable model because SOURCE Dice requires GT. It is a mechanism/novelty audit only.

Primary explanatory models, fit separately within each eligible external cohort:

- `M_quality`: `HARM ~ z(SOURCE_Dice) + architecture_family`
- `M_quality+risk`: `HARM ~ z(SOURCE_Dice) + z(SafeTTA_risk) + architecture_family`

Primary R16-A quantities:

1. coefficient/significance of `z(SafeTTA_risk)` in `M_quality+risk`;
2. likelihood-ratio improvement of `M_quality+risk` over `M_quality`;
3. AUROC difference between the two explanatory models, reported descriptively;
4. clustered bootstrap CI for the risk coefficient and AUROC difference when a physical-unit identifier is available.

Preferred external cohorts: **PolypGen/TENT1** and **PROMISE12/TENT1**. SUN-SEG may be included only if the required row-level fields are already frozen and recoverable.

No SafeTTA score is retrained, recalibrated, or re-thresholded.

## 2. R16-B — Matched learned current-reliability baseline

A SOURCE-only learned baseline will use conventional pre-adaptation predictive-reliability summaries and the **same low-capacity training family** as SafeTTA, so that any advantage cannot be attributed merely to using a supervised logistic-regression head.

### Fixed feature family

When SOURCE probability/logit maps or already frozen summaries are available, use only current-prediction information available before adaptation:

- mean predictive entropy;
- standard deviation of predictive entropy;
- 50th, 90th, and 95th percentiles of predictive entropy;
- mean maximum class probability / binary confidence;
- 10th percentile of confidence;
- fractions of pixels with confidence below 0.60, 0.70, 0.80, and 0.90;
- predicted foreground fraction.

For binary segmentation, entropy and confidence are computed from the SOURCE probability map using standard Bernoulli predictive entropy and `max(p, 1-p)`.

No DINO feature, GT-derived feature, post-adaptation quantity, `DeltaDice`, or target-domain statistic is allowed as a baseline input.

### Head and fitting

Use the same general low-capacity head family as the frozen SafeTTA safety estimator:

`median imputation -> standardization -> class-balanced logistic regression`

with the frozen SafeTTA logistic settings wherever compatible (`C=1`, L2, `lbfgs`, `max_iter=3000`, fixed seed `20260820`).

All fitting is SOURCE-only and grouped by the same physical-unit rules used by the corresponding SafeTTA development protocol.

### Evaluation

Primary comparisons are threshold-free HARM AUROC and AUPRC.

Where the frozen external row tables permit exact alignment, evaluate the learned reliability baseline on the same external rows used by SafeTTA.

No external HARM label, external Dice, external threshold selection, or target score statistic may be used in fitting.

## 3. Leakage lock

The following are prohibited for R16-B feature construction or model fitting:

- target GT;
- target `DeltaDice`;
- target HARM/BENEFIT labels;
- post-adaptation predictions;
- any target-derived feature normalization;
- target threshold tuning;
- model/feature selection based on external AUROC/AUPRC.

R16-A is explicitly a **post-reveal explanatory audit** and may use SOURCE Dice only as an explanatory covariate. Its outputs cannot modify the deployed SafeTTA policy.

## 4. Statistical unit

External uncertainty must remain clustered by physical unit:

- PolypGen: physical sample/image identifier;
- SUN-SEG: physical case/video identifier;
- PROMISE12: patient identifier.

Segmentation families/checkpoint states remain fixed experimental factors; confidence intervals quantify uncertainty over physical units conditional on the frozen model panel.

## 5. Decision rule

The R16 package is considered supportive if:

- the SafeTTA risk term contributes positive information beyond current SOURCE Dice in at least one principal external cohort and does not show a contradictory strong negative effect in the other principal cohort; and/or
- the matched learned reliability baseline remains materially below prediction-conditioned semantics on the principal external ranking evaluation.

No numerical pass threshold is chosen after seeing R16 results.

If the required raw SOURCE probabilities/summaries were not retained, R16-B is not reconstructed from target GT or approximate surrogate data. The audit must report the limitation transparently.

## 6. Scope

R16 is a reviewer-defense analysis only. It does **not** replace or alter:

- the frozen HARM definition `DeltaDice <= -0.02`;
- the frozen SafeTTA representation;
- frozen PCA/head/threshold artifacts;
- lock-before-GT external evaluations;
- manuscript primary numerical anchors;
- the public 44-anchor replay.

The current manuscript remains frozen until the R16 audit/experiment decision is completed.
