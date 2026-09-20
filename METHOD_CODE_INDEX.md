# SafeTTA method/code index — current Final68 evidence line

This repository preserves historical SafeTTA implementations while exposing a compact reviewer-facing layer for the current **Final68** manuscript.

The current method is:

```text
Final68 = Geometry4 + Transition64
```

The historical `v2.0-paper-final68` tag remains immutable. It is provenance for the pre-correction public state and must not be moved or rewritten.

---

## 1. Current reviewer entry point

Public replay:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected gate:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

Active compact evidence:

```text
reproducibility/final68_20260919/
```

Exact frozen controller:

```text
artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
```

Expected actual-file SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

The replay computes SHA256 from the **actual controller bytes**, not only from a recorded manifest string.

---

## 2. Final68 representation

### Geometry4

Four SOURCE-to-candidate prediction-change features:

1. prediction-to-prediction Dice;
2. prediction-to-prediction IoU;
3. foreground-area change;
4. boundary-density change.

### Transition64

Frozen DINOv2 patch features are pooled using SOURCE and candidate prediction supports. SOURCE-fitted PCA64 is held fixed and the semantic transition is:

```text
Transition64 = candidate-conditioned PCA64 - SOURCE-conditioned PCA64
```

### Final representation

```text
Final68 = [Geometry4 ; Transition64]
```

Absolute SOURCE-state descriptors are deliberately excluded from the frozen final representation. No explicit ActionID is supplied.

---

## 3. Tri-state controller

The frozen controller is a class-balanced logistic model over:

```text
HARM / NEUTRAL / BENEFIT
```

with utility:

```text
u = P(BENEFIT) - P(HARM)
```

Deployment rule:

```text
accept candidate if u > 0
otherwise retain SOURCE
```

The decision is **pre-commit**, not strictly pre-update: the candidate prediction has already been produced on a reversible copy while SOURCE remains recoverable.

---

## 4. TTA actions

Reviewer-facing action implementations are retained under:

```text
method_core/tta_actions/
experiments/tta_actions/
```

The manuscript evaluates:

- `TENT1` — episodic one-step entropy minimization;
- `PL-CONF90` — confidence-thresholded pseudo-label adaptation;
- `MEMO-SEG4-1STEP` — four deterministic segmentation-compatible views with one-step marginal-entropy minimization.

---

## 5. Primary action-shift evidence

The current primary action-shift evidence is the **grouped physical-image-disjoint held-out-action** analysis (E4D in the internal provenance naming).

Protocol boundary:

- 800 physical NeoPolyp images;
- three frozen DeepLabV3-R50 checkpoints;
- three actions: TENT1, PL-CONF90, MEMO;
- all rows from a validation image are excluded from the corresponding controller fit;
- the controller is fitted only on the two non-held-out actions from training-fold images;
- evaluation is performed on the held-out action from validation-fold images;
- no explicit ActionID is supplied;
- the audit found zero physical-image/action leakage.

Paper-facing HARM-vs-BENEFIT AUROC:

```text
TENT1      0.817440  [0.784944, 0.846591]
PL-CONF90  0.729658  [0.679815, 0.777933]
MEMO       0.735553  [0.687708, 0.783476]
```

Deployment interpretation:

```text
TENT1      deployed-SOURCE +0.005709  CI crosses zero
PL-CONF90  deployed-SOURCE +0.011781  CI excludes zero
MEMO       absolute deployment N/A in the frozen primary lineage
```

Historical E4C remains secondary provenance only and is **not** described as strict LOAO.

---

## 6. Endpoint-margin sensitivity

The frozen primary HARM/NEUTRAL/BENEFIT Dice margin is:

```text
±0.02
```

Margins `±0.01`, `±0.03`, and `±0.05` are sensitivity analyses only. They do not select or replace the frozen primary endpoint.

---

## 7. External PolypGen evaluation

The external joint-shift direction evaluates unseen MEMO on PolypGen using source-side development from NeoPolyp TENT1 + PL-CONF90.

PolypGen contributes no rows to controller fitting, calibration, score reversal, feature selection, or threshold tuning for the Final68 outcome analysis.

The frozen operating-point deployed-minus-SOURCE difference is positive but not statistically significant.

---

## 8. Frozen score-level comparator sensitivity

The paper-facing comparator panel uses frozen PolypGen risk-score vectors for:

- QCResUNet;
- TEGDA-ADIC;
- SicTTA-CCD;
- MC-dropout.

All scores are evaluated under the same 50% accept/rollback budget. This analysis compares **score ordering and deployment consequences**; it is not a harmonized end-to-end reproduction of the original methods' training pipelines.

QCResUNet is included through a pre-outcome locked QC risk vector and is treated as a score-level sensitivity comparator.

---

## 9. MRI / PROMISE12 framework replication

```text
experiments/mri_prostate_promise/
```

The Final68 representation/controller framework is independently re-fitted on Prostate158 and evaluated on PROMISE12.

This is **framework-level task/modality replication**, not zero-shot transfer of the colonoscopy controller.

The reported deployment endpoint is panel-level with patient-clustered inference; it is not claimed to be patient-level pooled 3-D Dice improvement.

---

## 10. Controller-only runtime boundary

The reported approximately:

```text
0.087 ms/case
```

covers only:

```text
StandardScaler
→ logistic-regression probability prediction
→ utility computation
→ accept/rollback decision
```

after Final68 already exists.

It excludes segmentation inference, candidate TTA execution, DINOv2 extraction, Geometry4, and Transition64 construction.

---

## 11. Active figure-data provenance

Current active figure-data files:

```text
reproducibility/final68_20260919/figure_data/fig2_e4d_action_shift_frozen_points.csv
reproducibility/final68_20260919/figure_data/figS1_margin_sensitivity_frozen_points.csv
```

The following two historical public files are intentionally excluded from the corrected active layer because their provenance is not independently sufficient for current paper claims:

```text
fig2_coverage_utility_frozen_points.csv
fig3_action_shift_loao_frozen_points.csv
```

---

## 12. Historical retained code

Historical implementation and provenance remain under:

```text
legacy_all_retained_code/
experiments/R31_action_transfer/
experiments/R32_validation/
experiments/R33_external_joint_shift/
provenance/R17A/
provenance/R31_R33/
```

These materials are preserved for traceability but do not define the current Final68 primary evidence line.

In particular, the historical 130-D `Q66 + DeltaSemantic64` representation and the old PolypGen result `0.743238 / 0.255458` are **not** current Final68 results.

---

## 13. Reproducibility scope

The supported public claim is deterministic verification of released paper-facing numerical anchors and SHA-bound frozen artifacts.

The repository does not claim:

- bit-identical retraining of every historical upstream segmenter;
- redistribution of raw datasets or target ground truth;
- redistribution of third-party DINOv2 weights;
- bit-identical wall-clock runtime across hardware.

Before changing manuscript claims, reviewers and maintainers should check:

```text
reproducibility/final68_20260919/CLAIM_BOUNDARIES.md
```
