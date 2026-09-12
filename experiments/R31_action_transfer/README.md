# R31 — Action-Transfer Safety

## Purpose

R31 evaluates whether harmful-update ranking transfers across mechanistically different TTA actions rather than only across domains.

The three actions are:

- `TENT1`
- `PL-CONF90`
- `MEMO-SEG4-1STEP`

The final interpretation is **action-transferable harmful-update structure**, not full action invariance.

## Key design

- NeoPolyp is split into a 200-image MEMO design subset and an independent 800-image action-transfer panel.
- MEMO learning rate is selected only on the design subset and then frozen.
- The 800-image panel is evaluated under all nine frozen segmentation states and all three actions.
- The transition representation combines the 66-D SOURCE state with a 64-D semantic transition: `Q66 + dSemantic64`.
- No explicit ActionID is required by the primary shared transition model.

## Main paper-facing result

Strict leave-one-action-out evaluation gives approximately:

- unseen MEMO AUROC: `0.7180`
- unseen PL-CONF90 AUROC: `0.7363`
- unseen TENT1 AUROC: `0.7705`
- macro AUROC: `0.7416`

Separate within-action models remain stronger and are treated as action-specific headroom rather than the transferable baseline.

## Public scope

This directory is for method/protocol provenance. Compact paper-facing statistics are replayed from `outputs/R31_R33/` through:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```
