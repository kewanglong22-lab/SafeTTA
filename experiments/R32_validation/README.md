# R32 — Three-Action Validation

## Purpose

R32 provides the strict validation of action transfer under frozen protocol conditions.

## Protocol

- Family: DeepLabV3-R50
- Actions:
  - MEMO-SEG4-1STEP
  - PL-CONF90
  - TENT1
- Evaluation: leave-one-action-out (LOAO)
- Predictor: frozen shared transition representation

No target calibration, score reversal, or action-specific refitting is used in the final shared evaluation.

## Outputs

Compact paper statistics:

```
outputs/R31_R33/r32a_three_action_loao_summary.csv
```

## Reproduction

Use the public replay entry point after extracting released compact outputs.
