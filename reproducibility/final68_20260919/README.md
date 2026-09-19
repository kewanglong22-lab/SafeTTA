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
- `../../artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib` — exact frozen Final68 controller.

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

## Binary artifacts

The exact frozen controller is redistributed in this repository and is SHA-bound in `FINAL68_LOCKS.json`:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

The target68 NPY remains SHA-bound provenance only and is **not** redistributed in the compact Final68 layer. Neither artifact should be reconstructed or substituted with an approximation.
