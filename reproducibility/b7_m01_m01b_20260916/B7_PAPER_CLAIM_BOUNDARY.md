# SafeTTA B7-M01/M01B — paper-facing evidence boundary

Bundle builder version: `2026-09-16-B7-PUBLIC-SYNC-BUNDLE-v1`

## Frozen scope

This public extension documents the post-B6 PolypGen unseen-MEMO
HARM-versus-BENEFIT explanatory analysis.

Primary four-stratum analysis:
**NOT EVALUABLE** because Q1/Q2 contain no decisive target events.
No target re-binning and no primary-endpoint replacement are permitted.

Overall secondary population:
- 860 model-cases;
- 301 HARM;
- 559 BENEFIT;
- conditional HARM prevalence = 0.35;
- 557 unique physical images;
- 64 physical images show both HARM and BENEFIT across different frozen model states.

## Paper-facing overall raw metrics

| Frozen score | AUROC | AUPRC |
|---|---:|---:|
| SOURCE state | 0.525713 | 0.367530 |
| SOURCE + simple change | 0.480967 | 0.344740 |
| Full transition | 0.640227 | 0.471926 |
| Change magnitude only | 0.492586 | 0.365923 |

## Paired physical-image bootstrap: Full transition minus comparator

- vs magnitude-only:
  - AUROC bootstrap mean delta = +0.146617
  - 95% CI = [0.077495, 0.219825]
- vs simple-change:
  - AUROC bootstrap mean delta = +0.159357
  - 95% CI = [0.112601, 0.204173]
- vs SOURCE-state:
  - AUROC bootstrap mean delta = +0.114505
  - 95% CI = [0.073318, 0.156702]

AUPRC paired intervals are also positive in the frozen report.

## Allowed interpretation

The frozen full model better discriminates explicit HARM from explicit
BENEFIT than the specified magnitude-only, simple-change, and SOURCE-state
scores in the overall PolypGen unseen-MEMO secondary analysis.

## Forbidden interpretation

Do not claim:
1. that the four-stratum primary succeeded;
2. that change magnitude was successfully controlled;
3. that the overall increment proves an independent/unique/causal semantic contribution;
4. that the geometry-versus-semantics division of labor is an established mechanism.

## Reproducibility lineage

Historical immutable B6 tag: `v1.2-paper-final-b6-20260915`

Planned next paper-facing tag after repository synchronization:
`v1.3-paper-final-b7-20260916`

This bundle does not create or modify Git tags.
