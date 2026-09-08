# Q1-R17A Published Reliability Baseline Comparison — Frozen Protocol v1-fix1

**Status:** PRE-REGISTERED BEFORE NEW TARGET-SCORE GENERATION  
**Reason for fix1:** Correct the SicTTA CCD implementation to match the authors' official released code exactly rather than a simplified prose interpretation.

## Frozen study question

For a frozen SOURCE segmentation state and an already-frozen TTA action, can a score computed **before adaptation** rank samples that later satisfy:

\[
\mathrm{HARM}=\mathbf 1[\Delta Dice\le -0.02]?
\]

SafeTTA, segmentation weights, TTA actions, HARM outcomes, external datasets, and all existing manuscript anchors remain frozen.

## B1. SicTTA Class Compact Density (CCD): official-code-faithful instantiation

The official SicTTA implementation computes CCD from the SOURCE model prediction as follows.

For one image:

1. Obtain SOURCE softmax probabilities with shape `[1,C,H,W]`.
2. Permute/reshape to a pixel-by-class matrix `[N,C]`.
3. Randomly sample `min(200,N)` pixels.
4. L2-normalize each sampled pixel probability vector across the class dimension.
5. Form the class similarity Gram matrix:
   \[
   G = P^\top P \in \mathbb R^{C\times C}.
   \]
6. Apply softmax along `dim=1` to `G`.
7. For each row compute:
   \[
   H_r=-\sum_c q_{r,c}\log(q_{r,c}+10^{-5}).
   \]
8. CCD is the mean row entropy.

The official SicTTA selection logic regards **lower CCD as more Source-Friendly**. Therefore R17A fixes:

\[
s_{\mathrm{CCD-risk}}=\mathrm{CCD}.
\]

No target-GT orientation fitting is allowed.

### Reproducibility lock for the official random 200-pixel sampling

The official code uses `torch.randperm`. R17A freezes:
- global seed: `20260908`;
- deterministic sorted sample traversal;
- one official-style random pixel draw per sample;
- no target-label tuning;
- no averaging over multiple random draws unless separately declared as a sensitivity analysis.

This changes only RNG reproducibility, not the CCD mathematical procedure.

### Binary segmentation

For a binary foreground probability `p`, represent the SOURCE probability map as:

\[
[1-p,\;p]
\]

before CCD.

## B2. TEGDA-ADIC

Retain the previously frozen plan: use only the published adaptation-free ADIC prediction-quality score, if and only if a faithful architecture-specific MC-dropout inference path can be established without changing SOURCE weights or using target GT.

Do not run the TEGDA adaptation framework. Do not invent a "TEGDA-like" approximation.

## B3. MC-dropout predictive entropy

If B2 is feasible, reuse exactly the same MC-dropout passes for predictive entropy of the mean prediction.

## Evaluation matrix

Primary:
- PolypGen / TENT1
- SUN-SEG / TENT1
- SUN-SEG / PL-CONF90
- PROMISE12 / TENT1

Development/support:
- NeoPolyp
- Prostate158

No new dataset.

## Lock-before-GT rule

For every external dataset:
1. generate score table using image + frozen SOURCE state only;
2. serialize score table;
3. compute SHA256;
4. write a pre-GT lock report;
5. only then join the already-frozen HARM/outcome table for evaluation.

No target GT may select:
- score direction,
- random seed,
- sampled pixel count,
- dropout configuration,
- threshold,
- model layer,
- calibration,
- preprocessing variant.

## Metrics

Primary:
- HARM AUROC
- HARM AUPRC
- paired clustered/bootstrap `SafeTTA - baseline` differences
- 95% CI using the same physical grouping unit as the manuscript.

Operating-point metrics are secondary and may use SOURCE-development thresholds only.

## Manuscript interpretation

CCD/ADIC are evaluated as **published pre-adaptation reliability/quality signals under SafeTTA's frozen future-HARM endpoint**. This is not a claim that the full SicTTA or TEGDA adaptation frameworks were reproduced.

## Asset-preservation lock

Until R17A closes, do not delete:
- PolypGen locked logits/predictions;
- SUN locked SOURCE/TENT/PL assets;
- PROMISE12 SOURCE prediction assets;
- final SOURCE segmentation checkpoints;
- dataset images/manifests;
- frozen outcome/HARM tables.
