# SafeTTA Method Code Index

This repository exposes **two complementary reproducibility layers**:

1. **Method implementation layer** — exact retained scripts implementing the
   paper-facing representation, safety estimator, TTA actions, data processing,
   external evaluation, MRI/PROMISE12 workflow, and analysis.
2. **Paper-statistic replay layer** — the validated minimal replay reproducing
   all 44 prespecified paper-level numerical anchors from released frozen
   intermediate artifacts.

## Reviewer entry points

### 1. Prediction-conditioned representation

`method_core/representation/Q1_R10K2A_dinov2_image_mask_representation_lock_fix1.py`

Implements the frozen DINOv2-base image/mask representation used by SafeTTA,
including the direct 352-to-16x16 non-overlapping mask occupancy mapping,
foreground/background prediction-conditioned semantic pooling, and the frozen
precision boundary used before PCA.

Morphology:
`method_core/representation/Q1_R10J1_source_mask_morphology_feature_builder_fix1.py`

### 2. Frozen colonoscopy safety estimator

`method_core/safety_estimator/Q1_R10L0_final_source_safety_estimator_lock_fix3.py`

Implements the final source-only PCA64 + median imputation + standardization +
class-balanced logistic-regression safety estimator and frozen operating point.

The serialized PCA/head/threshold artifacts are under:
`artifacts/colonoscopy_safety_estimator/`.

### 3. TTA actions

Exact retained action implementations are under:
`method_core/tta_actions/` and `experiments/tta_actions/`.

The final paper evaluates:
- episodic one-step TENT-style adaptation;
- confidence-0.90 pseudo-label adaptation;
- architecture-compatible MRI TENT1 behavior.

### 4. External colonoscopy workflows

`experiments/polypgen/`
`experiments/sunseg/`

These directories include the retained manifest/prediction/safety-score/
GT-reveal evaluation lineage used in the final study.

### 5. MRI / PROMISE12 workflow

`experiments/mri_prostate_promise/`

Includes retained preprocessing/model protocol, source OOF segmentation,
TENT1 outcome generation, MRI safety-estimator fitting, PROMISE12 pre-GT
prediction/scoring, operating-point transport, locked-policy evaluation,
external ranking baseline, and matched-coverage utility analysis.

### 6. Paper analysis

`experiments/paper_analysis/`

Contains the final ablation, paired bootstrap, selective utility, runtime,
harm-margin sensitivity, and public 44-anchor replay scripts.

## Historical retained code

`legacy_all_retained_code/` contains all SafeTTA-related Python scripts selected
by the 332-file retained-code audit. It is included for provenance and reviewer
completeness. Historical exploratory revisions are **not all recommended
entry points**; use the grouped `method_core/` and `experiments/` directories
above for the final paper-facing implementation.

## Upstream segmentation training boundary

Retained source-training implementations are included under
`experiments/source_training/`.

However, complete per-image upstream training manifests/checkpoints were not
recoverable for every one of the nine historical NeoPolyp segmentation states.
Accordingly, the repository does **not** claim bit-identical retraining of every
historical upstream segmenter. This limitation does not affect the validated
44-anchor paper-statistic replay from the released frozen artifacts.
