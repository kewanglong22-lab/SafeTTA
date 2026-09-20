# Final68 scientific claim boundaries

## Supported
- SafeTTA is a **prediction-conditioned, pre-commit** safety-ranking layer for a specified candidate TTA update.
- Final68 = Geometry4 + Transition64.
- Final68 contains candidate-induced transition information; absolute SOURCE-state descriptors are not part of the frozen final representation.
- The controller is tri-state: HARM / NEUTRAL / BENEFIT.
- Frozen utility: `u = P(BENEFIT) - P(HARM)`.
- Accept candidate when `u > 0`; otherwise retain SOURCE.
- On the frozen PolypGen component-ablation panel, Final68 improves HARM-vs-BENEFIT AUROC over Geometry4 by `+0.117610` (paired 95% CI `[0.067823, 0.165476]`) and over Transition64 by `+0.059111` (`[0.030953, 0.087875]`). These intervals support incremental **directional** discrimination from fusion.
- The paired component result does **not** establish that Final68 maximizes global HARM-vs-rest AUROC or that fusion necessarily improves deployed Dice.
- The **primary action-shift analysis** is grouped by physical image and held out by action. HARM-vs-BENEFIT discrimination remains above chance for held-out TENT1, PL-CONF90, and MEMO.
- TENT1 primary deployment-minus-SOURCE is not statistically significant because its clustered-bootstrap CI crosses zero.
- PL-CONF90 primary deployment-minus-SOURCE has a positive clustered-bootstrap CI.
- Absolute MEMO deployment is **N/A** in the primary grouped image-disjoint lineage because the required paired absolute fields are unavailable.
- The frozen primary HARM/NEUTRAL/BENEFIT Dice margin is **±0.02**. Margins ±0.01, ±0.03, and ±0.05 are endpoint-sensitivity analyses only. The corresponding frozen H-v-B AUROCs are 0.722411, 0.760884, 0.770854, and 0.782102 for margins 0.01, 0.02, 0.03, and 0.05.
- PROMISE12 supports an independently re-fitted prostate-MRI framework replication with patient-clustered inference; it is not zero-shot transfer of the colonoscopy controller.
- The comparator panel is a **frozen score-level sensitivity analysis** under a common 50% accept/rollback budget, not a harmonized end-to-end reproduction of the original methods.
- The historical QC score used in the comparator panel is reported in the final manuscript as **QC proxy (QCResUNet-lineage)**. The pre-outcome frozen score vector is recoverable, but the original checkpoint/training/full implementation provenance is not independently reconstructed; no faithful end-to-end QCResUNet reproduction is claimed.
- Grouped image-disjointness is claimed for the supervised `StandardScaler + logistic-regression` controller fitting stage. The retained lineage does not establish fold-local PCA64 refitting for every LOAO split, so no stronger full-pipeline zero-image-exposure claim is made for the upstream SOURCE-development PCA transform.
- Ranking quality is not equivalent to deployment utility.
- `0.087 ms` describes the **controller-only** decision stage after Final68 is already available.

## Not supported
- Final68 has the best global HARM AUROC among all comparators.
- Final68 significantly improves the frozen PolypGen operating point over SOURCE.
- Final68 uniformly outperforms the QC proxy or the original QCResUNet method.
- The score-level comparator table proves one complete end-to-end method is uniformly superior.
- The QC proxy is a verified faithful reproduction of the original QCResUNet checkpoint/training pipeline.
- TEGDA-ADIC is globally worse simply because its deployed Dice can be lower.
- Historical E4C is the primary or strict LOAO result.
- MEMO has a primary absolute deployed-minus-SOURCE result when the frozen paired lineage does not support it.
- The whole representation pipeline is guaranteed fold-local image-disjoint merely because the supervised controller fit is image-disjoint.
- PROMISE12 is zero-shot transfer of the exact colonoscopy controller.
- The PROMISE12 panel-level Dice difference is patient-level pooled 3-D Dice improvement.
- `0.087 ms` is end-to-end SafeTTA latency.
- `u=0` is a clinically calibrated or mathematically optimal threshold.
- The logistic outputs are calibrated clinical probabilities.
- Final68 is action-invariant merely because no ActionID is provided.

## Fix11 release assets
- `FINAL68_TABLE4_PAIRED_BOOTSTRAP_SUMMARY.csv` binds the paired component-effect intervals.
- `FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv` binds the four endpoint-margin sensitivity metrics.
- `Q1_Final68_Table4_paired_bootstrap_ci_v1.py` reproduces the Table 4 paired clustered-bootstrap calculation from the frozen per-case component-score table when that private/frozen input is available.
- `V7_FIX11_FINAL_MIA_CLOSURE_AUDIT.md` records the final manuscript closure boundary.
- `../../code/Q1_Final68_fix11_release_asset_audit_v1.py` verifies the public fix11 release assets and their hashes.

## Historical boundary
Historical 66-D / 70-D / 130-D representations, historical E4C shared-image-panel action-held-out evidence, and the old PolypGen `0.7432 / 0.2555` full-transition result are retained only as provenance. They are **not** the current Final68 primary evidence line.

Two previously public figure-data files are deliberately excluded from the corrected active layer because their provenance is not independently sufficient for current paper claims:

- `fig2_coverage_utility_frozen_points.csv`
- `fig3_action_shift_loao_frozen_points.csv`
