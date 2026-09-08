# SafeTTA R16 reviewer-defense reproducibility

This directory records the post-reveal reviewer-defense analysis added for the
v12 manuscript. It does **not** modify the frozen SafeTTA representation,
deployment threshold, external lock-before-GT protocol, or the pre-existing
44-anchor public replay.

## Scientific purpose

R16-A/R16-A2 test whether the already frozen SafeTTA risk score contains
information about future adaptation-induced Dice degradation beyond the
current SOURCE segmentation quality itself.

R16-A2 is the manuscript-facing robustness analysis:

- PolypGen rows: **13788**
- physical images: **1532**
- five-fold physical-image grouped cross-fitting
- flexible cubic-spline control for SOURCE Dice
- fixed architecture-family effects
- frozen SafeTTA risk added only as an explanatory covariate
- 2000 paired physical-image clustered bootstrap replicates

Primary R16-A2 results:

- flexible quality-only OOF AUROC: **0.812464**
- flexible quality + frozen risk OOF AUROC: **0.840719**
- delta AUROC: **0.028254**
  (95% CI 0.021740 to 0.034372)
- delta AUPRC: **0.071083**
  (95% CI 0.059380 to 0.082511)
- delta log loss (quality+risk minus quality):
  **-0.031656**
  (95% CI -0.038183 to -0.024796)

Gate:

`PASS_R16A2_FROZEN_RISK_ADDS_CROSSFITTED_INFORMATION_BEYOND_FLEXIBLE_SOURCE_QUALITY`

## Re-run R16-A2

From the repository root:

```bash
python experiments/reviewer_defense/Q1_R16A2_crossfitted_flexible_source_quality_control_v1.py \
  --input outputs/r16_reviewer_defense/R16A_ANALYSIS_TABLE.csv \
  --out-dir outputs/r16_reviewer_defense/replay_r16a2
```

The committed derived analysis table contains only the variables required for
this post-reveal explanatory audit; it does not contain raw medical images.

## R16-B boundary

The attempted matched learned reliability baseline was **not** forced through
an approximate reconstruction. The frozen-input alignment audit concluded:

`GATE=BLOCKED_R16B_EXACT_FROZEN_ALIGNMENT_NOT_ESTABLISHED`

Therefore no target-GT-derived or approximate probability reconstruction was
used to create that baseline. The audit code/report are retained for
provenance.

## Statistical interpretation

R16-A2 confidence intervals quantify uncertainty over PolypGen physical images
conditional on the frozen segmentation-model panel. SOURCE Dice is used only
as a post-reveal explanatory covariate and is never a deployable SafeTTA input.
