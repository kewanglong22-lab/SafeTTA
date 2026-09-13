# R31–R33 compact paper-facing outputs

This directory contains lightweight frozen summaries used by the public R31–R33 paper-statistic replay. It is intentionally small: raw datasets, segmentation checkpoints, DINOv2 weights, and large intermediate tensors are not redistributed here.

## Files

- `r31_controller_negative_summary.csv` — pre-registered multi-action controller feasibility outcome. The controller did **not** meet the full success criterion and is retained as a negative diagnostic.
- `r32a_three_action_loao_summary.csv` — strict three-action leave-one-action-out action-transfer summary for the shared transition representation and ablations.
- `r33_polypgen_memo_primary_metrics.csv` — pooled PolypGen unseen-MEMO metrics for SafeTTA and published reliability baselines.
- `r33_polypgen_memo_paired_deltas.csv` — physical-image clustered paired-bootstrap SafeTTA-minus-baseline deltas.
- `r33_claim_freeze_summary.json` — frozen paper-facing R33 claim boundary and confirmation decision.

## Public replay

From repository root:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Expected terminal gate:

```text
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

## Scope

The compact replay verifies released manuscript-level numerical anchors. It does not claim bit-identical re-execution of all upstream segmentation/TTA/DINO inference from redistributed raw assets.
