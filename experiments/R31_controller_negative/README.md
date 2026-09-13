# R31 — Controller Negative Analysis

## Purpose

This directory records the negative analysis of extending SafeTTA into a closed-loop risk controller.

The manuscript claim is intentionally restricted to:

> prediction-conditioned safety ranking for selective adaptation.

It does not claim a complete automatic adaptation controller.

## Interpretation

The controller analysis showed that ranking future harmful updates is reliable, but deciding the final intervention policy introduces additional assumptions, including:

- action availability,
- utility/cost trade-offs,
- deployment policy constraints.

Therefore controller results are reported as analysis rather than the primary contribution.

## Reproducibility

Final compact statistics are stored under:

```
outputs/R31_R33/
```
