# Final68 scientific claim boundaries

## Supported
- SafeTTA is a **prediction-conditioned, pre-commit** safety-ranking layer for a specified candidate TTA update.
- Final68 = Geometry4 + Transition64.
- Final68 contains candidate-induced transition information; absolute SOURCE-state descriptors are not part of the frozen final representation.
- The controller is tri-state: HARM / NEUTRAL / BENEFIT.
- Frozen utility: `u = P(BENEFIT) - P(HARM)`.
- Accept candidate when `u > 0`; otherwise retain SOURCE.
- The **primary action-shift analysis** is grouped by physical image and held out by action. HARM-vs-BENEFIT discrimination remains above chance for held-out TENT1, PL-CONF90, and MEMO.
- TENT1 primary deployment-minus-SOURCE is not statistically significant because its clustered-bootstrap CI crosses zero.
- PL-CONF90 primary deployment-minus-SOURCE has a positive clustered-bootstrap CI.
- Absolute MEMO deployment is **N/A** in the primary grouped image-disjoint lineage because the required paired absolute fields are unavailable.
- The frozen primary HARM/NEUTRAL/BENEFIT Dice margin is **±0.02**. Margins ±0.01, ±0.03, and ±0.05 are endpoint-sensitivity analyses only.
- PROMISE12 supports an independently re-fitted task/modality framework replication with patient-clustered inference; it is not zero-shot transfer of the colonoscopy controller.
- The comparator panel is a **frozen score-level sensitivity analysis** under a common 50% accept/rollback budget, not a harmonized end-to-end reproduction of the original methods.
- Ranking quality is not equivalent to deployment utility.
- `0.087 ms` describes the **controller-only** decision stage after Final68 is already available.

## Not supported
- Full68 has the best global HARM AUROC among all comparators.
- Full68 significantly improves the frozen PolypGen operating point over SOURCE.
- Full68 uniformly outperforms QCResUNet.
- The score-level comparator table proves one complete end-to-end method is uniformly superior.
- TEGDA-ADIC is globally worse simply because its deployed Dice can be lower.
- Historical E4C is the primary or strict LOAO result.
- MEMO has a primary absolute deployed-minus-SOURCE result when the frozen paired lineage does not support it.
- PROMISE12 is zero-shot transfer of the exact colonoscopy controller.
- The PROMISE12 panel-level Dice difference is patient-level pooled 3-D Dice improvement.
- `0.087 ms` is end-to-end SafeTTA latency.
- `u=0` is a clinically calibrated or mathematically optimal threshold.
- The logistic outputs are calibrated clinical probabilities.
- Final68 is action-invariant merely because no ActionID is provided.

## Historical boundary
Historical 66-D / 70-D / 130-D representations, historical E4C shared-image-panel action-held-out evidence, and the old PolypGen `0.7432 / 0.2555` full-transition result are retained only as provenance. They are **not** the current Final68 primary evidence line.

Two previously public figure-data files are deliberately excluded from the corrected active layer because their provenance is not independently sufficient for current paper claims:

- `fig2_coverage_utility_frozen_points.csv`
- `fig3_action_shift_loao_frozen_points.csv`
