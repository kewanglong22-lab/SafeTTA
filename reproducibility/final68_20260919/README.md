# Final68 public reproducibility layer

This directory is the paper-facing compact provenance layer for the current SafeTTA-Final68 manuscript.

It does not rewrite or invalidate historical SafeTTA tags/releases. Historical 66-D / 70-D / 130-D evidence remains immutable and is scientifically distinct from Final68.

## Current method

`Final68 = Geometry4 + Transition64`

The controller predicts HARM / NEUTRAL / BENEFIT and uses:

```text
utility = P(BENEFIT) - P(HARM)
accept if utility > 0
otherwise retain SOURCE
```

The transition is scored **after the candidate prediction has been produced but before it is committed**.

## Files

- `FINAL68_NUMERIC_ANCHORS.csv` — compact frozen paper-facing numerical anchors.
- `FINAL68_LOCKS.json` — exact known artifact and lock SHA256 identifiers.
- `CLAIM_BOUNDARIES.md` — supported and forbidden interpretations.
- `figure_data/` — frozen paper-facing figure point tables.

## Public replay

From repository root:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected final line:

```text
GATE=PASS_FINAL68_PUBLIC_PAPER_STAT_REPLAY
```

This replay verifies the compact public evidence layer. It does not retrain segmentation models, re-run TTA, reconstruct DINOv2 features, or access target ground truth.

## Binary-author artifacts

The exact author-side frozen controller and target68 matrix are SHA-bound in `FINAL68_LOCKS.json`. They must not be reconstructed or substituted with approximations. If redistributed in a later release asset, the bytes must match the recorded SHA256 exactly.
