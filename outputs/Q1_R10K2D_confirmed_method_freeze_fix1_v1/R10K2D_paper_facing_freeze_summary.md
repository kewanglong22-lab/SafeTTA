# R10K2 Confirmed Method Freeze

**Frozen primary:** `M2_plus_CondDINO_PCA64`

- Macro AUROC: 0.805745 -> 0.826563
- Macro AUPRC: 0.656118 -> 0.709391
- Transferred macro FPR: 0.560358 -> 0.476519
- Macro PPV@1%: 0.016156 -> 0.020549
- Oracle R90 macro FPR: 0.566910 -> 0.471209

## Paired case-clustered bootstrap
- Delta AUROC: 0.020589, 95% CI [0.009964, 0.031082]
- Delta AUPRC: 0.052249, 95% CI [0.031803, 0.073188]
- Delta FPR: -0.083771, 95% CI [-0.115470, -0.052011]
- Delta PPV1pct: 0.004422, 95% CI [0.003005, 0.005966]
- Delta OracleR90FPR: -0.095425, 95% CI [-0.137479, -0.050464]

## Frozen claim
The current evidence supports cross-model-family TTA safety-risk ranking within NeoPolyp with macro case-clustered inferential support.

It does not yet support independent external-cohort generalization or a uniformly transferable automatic high-recall safety gate.

Next required evidence: independent external-cohort validation with the method frozen exactly as above.