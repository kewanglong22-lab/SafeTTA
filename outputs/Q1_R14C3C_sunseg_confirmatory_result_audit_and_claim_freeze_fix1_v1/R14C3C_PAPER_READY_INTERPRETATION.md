# R14C3C SUN-SEG Confirmatory Claim Freeze

## Frozen interpretation

On an untouched SUN-SEG cohort, the frozen pre-adaptation SOURCE-conditioned safety score retained statistically supported harm-ranking signal for both TENT1 and PL_CONF90 despite severe adaptation-action outcome shift. However, the frozen high-recall operating point did not transfer reliably.

## TENT1

- AUROC: 0.676046
- AUROC 95% CI: [0.626603, 0.719233]
- AUPRC: 0.352846
- Recall at frozen threshold: 0.665033
- FPR at frozen threshold: 0.420945
- Oracle diagnostic FPR at Recall≈0.90: 0.685429

## PL_CONF90

- AUROC: 0.639533
- AUROC 95% CI: [0.590198, 0.685808]
- AUPRC: 0.157028
- Recall at frozen threshold: 0.665824
- FPR at frozen threshold: 0.457580
- Oracle diagnostic FPR at Recall≈0.90: 0.651684

## Action shift

- HARM Jaccard: 0.090682
- HARM Cohen kappa: 0.010813
- DeltaDice Spearman: -0.149065
- Paired PL-TENT1 AUROC difference 95% CI:
  [-0.097718, 0.024020]
- Paired PL-TENT1 frozen-threshold FPR difference 95% CI:
  [0.016588, 0.057287]

## Interpretation boundary

Supported:
- independent harm-ranking signal for TENT1;
- independent harm-ranking signal for PL_CONF90;
- dual-action ranking signal despite action-specific harm identities;
- weak/unreliable high-recall operating-point transfer.

Not supported:
- automatic deployment gate is solved;
- TENT1 AUROC is significantly better than PL;
- the same samples are harmed across actions;
- direct AUPRC superiority claim across actions with different prevalences.
