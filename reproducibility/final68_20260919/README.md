# Final68 corrected public reproducibility layer

This directory is the compact paper-facing provenance layer for the current SafeTTA-Final68 manuscript. The corrected v7 public core was released first; the fix11 manuscript-aligned sync adds the final paired component statistics, endpoint-margin summary, and closure audit without rewriting historical tags.

Historical tags `v2.0-paper-final68` and `v2.1-paper-final68-v7-20260920` remain unchanged.

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

## Final fix11 component evidence

On the frozen PolypGen component-ablation panel:

```text
Final68 - Geometry4 H-v-B AUROC
  raw delta  +0.117610
  95% CI     [0.067823, 0.165476]

Final68 - Transition64 H-v-B AUROC
  raw delta  +0.059111
  95% CI     [0.030953, 0.087875]
```

Both paired physical-image-clustered bootstrap intervals exclude zero. This supports incremental directional HARM-vs-BENEFIT discrimination from fusion; it does not imply that Final68 maximizes global HARM-vs-rest AUROC or guarantees deployment Dice improvement.

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

### Endpoint-margin sensitivity
The frozen primary HARM/NEUTRAL/BENEFIT margin is `±0.02` Dice. Margins `±0.01`, `±0.03`, and `±0.05` are sensitivity analyses only.

| Margin | H-v-B AUROC | H-v-B AUPRC | HARM rollback | BENEFIT accepted | Role |
|---:|---:|---:|---:|---:|---|
| ±0.01 | 0.722411 | 0.716600 | 0.716243 | 0.695829 | sensitivity |
| ±0.02 | 0.760884 | 0.737825 | 0.710086 | 0.742179 | primary |
| ±0.03 | 0.770854 | 0.737988 | 0.693338 | 0.773107 | sensitivity |
| ±0.05 | 0.782102 | 0.740094 | 0.673712 | 0.798075 | sensitivity |

### PROMISE12
PROMISE12 is an independently re-fitted prostate-MRI framework replication. Its reported deployment endpoint is panel-level with patient-clustered inference and is not claimed to be patient-level pooled 3-D Dice improvement.

### Score-level comparator sensitivity
The PolypGen comparator table uses frozen score vectors under a common 50% accept/rollback budget. It is not presented as a harmonized end-to-end reproduction of QCResUNet, TEGDA, SicTTA, or MC-dropout.

The historical QC score is described in the final manuscript as `QC proxy (QCResUNet-lineage)`: the pre-outcome frozen risk vector is recoverable, but the original checkpoint, training data, and complete implementation provenance cannot be independently reconstructed. No faithful end-to-end QCResUNet reproduction is claimed.

### PCA / grouped-LOAO scope
The supervised `StandardScaler + logistic-regression` controller fitting is grouped image-disjoint: validation-image rows are excluded from the corresponding fit. The retained lineage does not establish fold-local PCA64 refitting for every LOAO split, so no stronger whole-pipeline zero-image-exposure claim is made for the upstream SOURCE-development PCA transform.

## Files

Core files:

- `FINAL68_NUMERIC_ANCHORS.csv` — corrected active paper-facing numerical anchors.
- `FINAL68_LOCKS.json` — artifact SHA bindings, active evidence roles, exclusions, and claim boundaries.
- `CLAIM_BOUNDARIES.md` — supported and forbidden interpretations.
- `figure_data/fig2_e4d_action_shift_frozen_points.csv` — canonical E4D action-shift points used by the corrected manuscript figure.
- `figure_data/figS1_margin_sensitivity_frozen_points.csv` — frozen decision-threshold sensitivity points.
- `../../artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib` — exact frozen Final68 controller.

Fix11 additions:

- `FINAL68_TABLE4_PAIRED_BOOTSTRAP_SUMMARY.csv` — paired component AUROC differences and 95% CIs.
- `FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv` — endpoint-definition sensitivity at Dice margins 0.01/0.02/0.03/0.05.
- `Q1_Final68_Table4_paired_bootstrap_ci_v1.py` — Table 4 physical-image-clustered paired bootstrap script.
- `V7_FIX11_FINAL_MIA_CLOSURE_AUDIT.md` — final manuscript closure audit.
- `../../code/Q1_Final68_fix11_release_asset_audit_v1.py` — public asset/hash audit for the fix11 additions.

The following previously public assets remain deliberately excluded from the corrected active layer because they are not independently sufficient for current paper claims:

- `fig2_coverage_utility_frozen_points.csv`
- `fig3_action_shift_loao_frozen_points.csv`

## Public replay

Core corrected replay from repository root:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected final line:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

Fix11 release-asset audit:

```bash
python code/Q1_Final68_fix11_release_asset_audit_v1.py --root .
```

Expected final line:

```text
GATE=PASS_FINAL68_V7_FIX11_RELEASE_ASSET_AUDIT
```

The core replay verifies corrected numerical anchors, active figure-data hashes, unsafe-asset exclusions, and the actual controller file bytes. The fix11 audit verifies the hashes and bound values for the final Table 4 paired-CI and endpoint-margin assets.

Neither audit retrains segmentation models, re-runs TTA/DINO inference, or accesses target ground truth.

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
