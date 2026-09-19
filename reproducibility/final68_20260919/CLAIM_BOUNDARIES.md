# Final68 scientific claim boundaries

## Supported
- SafeTTA is a **prediction-conditioned, pre-commit** safety-ranking layer for a specified candidate TTA update.
- Final68 = Geometry4 + Transition64.
- The controller is tri-state: HARM / NEUTRAL / BENEFIT.
- Frozen utility: `u = P(BENEFIT) - P(HARM)`.
- Accept candidate when `u > 0`; otherwise retain SOURCE.
- Strict LOAO HARM-vs-BENEFIT discrimination remains useful across TENT1, PL-CONF90, and MEMO.
- PROMISE12 supports an independently re-fitted task-domain framework replication.
- Matched coverage demonstrates that **ranking quality is not equivalent to deployment utility**.

## Not supported
- Full68 has the best global HARM AUROC among all comparators.
- Full68 significantly improves the frozen PolypGen operating point over SOURCE.
- Full68 uniformly outperforms QCResUNet across coverage.
- TEGDA-ADIC is globally worse simply because its deployed Dice can be lower.
- PROMISE12 is zero-shot transfer of the exact colonoscopy controller.
- `0.087 ms` is end-to-end SafeTTA latency.
- `u=0` is a clinically calibrated or mathematically optimal threshold.
- The logistic outputs are calibrated clinical probabilities.
- Final68 is action-invariant merely because no ActionID is provided.

## Legacy boundary
Historical 66-D / 70-D / 130-D representations and the old PolypGen `0.7432 / 0.2555` full-transition result are retained only as historical provenance. They are **not** current Final68 evidence.
