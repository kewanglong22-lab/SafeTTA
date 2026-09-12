# R31-R33 reproducibility scope

This additive package is intended to update the existing SafeTTA public repository after tag `v1.0-paper-v14`.

It provides:
1. exact retained R31-R33 experiment/protocol scripts available in the frozen workspace;
2. paper-facing compact R32/R33 result tables;
3. a portable deterministic paper-statistic replay for the new action-transfer and joint-shift anchors;
4. SHA manifests and claim-boundary documentation.

It does **not** claim standalone bit-identical rerunning of all heavy inference stages. Some retained experiment scripts intentionally preserve historical workspace defaults and depend on exact upstream frozen modules/assets already tracked by the SafeTTA repository or retained author workspace. The reviewer-facing public execution path is the compact replay script under `code/`.

Large model checkpoints, datasets, DINOv2 weights, and target ground truth are not redistributed.
