# Final68 corrected public reproducibility layer

This directory is the compact paper-facing provenance layer for the current SafeTTA-Final68 manuscript.

It does **not** rewrite historical SafeTTA releases. The historical tag `v2.0-paper-final68` remains immutable and corresponds to the pre-correction public state. Historical 66-D / 70-D / 130-D evidence and historical E4C action-held-out results remain provenance only.

## Current method

```text
Final68 = Geometry4 + Transition64
```

- Geometry4: SOURCE-to-candidate prediction Dice, IoU, foreground-area change, and boundary-density change.
- Transition64: candidate-conditioned minus SOURCE-conditioned semantic PCA64 descriptor from frozen DINOv2 patch features.
- Controller: class-balanced tri-state logistic regression predicting HARM / NEUTRAL / BENEFIT.
- Utility: `u = P(BENEFIT) - P(HARM)`.
- Decision: accept candidate if `u > 0`; otherwise retain SOURCE.

The candidate is executed reversibly before the safety decision; SOURCE remains recoverable until commit.

## Active evidence line

### External PolypGen
The frozen Final68 controller is evaluated on unseen MEMO under domain shift. The operating-point deployed-minus-SOURCE difference is positive but not statistically significant.

### Primary action shift: grouped image-disjoint held-out-action evaluation
The current primary action-shift evidence is E4D, not historical E4C.

| Held-out action | HARM-vs-BENEFIT AUROC | 95% CI | Deployed − SOURCE |
|---|---:|---:|---:|
| TENT1 | 0.817440 | [0.784944, 0.846591] | +0.005709 [-0.002026, 0.013128] |
| PL-CONF90 | 0.729658 | [0.679815, 0.777933] | +0.011781 [0.008599, 0.015475] |
| MEMO | 0.735553 | [0.687708, 0.783476] | N/A |

The TENT1 deployment interval crosses zero. Absolute MEMO deployment is unavailable in the frozen primary lineage.

### Endpoint margin sensitivity
The frozen primary HARM/NEUTRAL/BENEFIT margin is `±0.02` Dice. Margins `±0.01`, `±0.03`, and `±0.05` are sensitivity analyses only and are not used to select the primary endpoint.

### PROMISE12
PROMISE12 is an independently re-fitted prostate-MRI framework replication. Its reported deployment endpoint is panel-level with patient-clustered inference and is not claimed to be patient-level pooled 3-D Dice improvement.

### Score-level comparator sensitivity
The PolypGen comparator table uses frozen score vectors under a common 50% accept/rollback budget. It is not presented as a harmonized end-to-end reproduction of QCResUNet, TEGDA, SicTTA, or MC-dropout.

## Files

- `FINAL68_NUMERIC_ANCHORS.csv` — corrected active paper-facing numerical anchors.
- `FINAL68_LOCKS.json` — artifact SHA bindings, active evidence roles, exclusions, and claim boundaries.
- `CLAIM_BOUNDARIES.md` — supported and forbidden interpretations.
- `figure_data/fig2_e4d_action_shift_frozen_points.csv` — canonical E4D action-shift points used by the corrected manuscript figure.
- `figure_data/figS1_margin_sensitivity_frozen_points.csv` — frozen margin-sensitivity points.
- `../../artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib` — exact frozen Final68 controller.

The following previously public assets are deliberately excluded from the corrected active layer because they are not independently sufficient for current paper claims:

- `fig2_coverage_utility_frozen_points.csv`
- `fig3_action_shift_loao_frozen_points.csv`

## Public replay

From repository root:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected final line:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

The replay verifies corrected numerical anchors, verifies the active figure-data hashes, confirms exclusion of the two unsafe public assets, and computes SHA256 from the **actual controller file bytes** before comparing it with the frozen digest.

It does not retrain segmentation models, re-run TTA/DINO inference, or access target ground truth.

## Controller artifact

Path:

```text
artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
```

Expected actual-file SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

## Runtime boundary

The reported approximately `0.087 ms/case` is the controller-only decision stage after Final68 already exists. It excludes candidate TTA execution, segmentation inference, DINOv2 extraction, Geometry4, and Transition64 construction.
