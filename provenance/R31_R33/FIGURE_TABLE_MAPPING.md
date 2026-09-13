# R31-R33 Figure / Table Mapping

This file maps the current manuscript's R31-R33 figures/tables to the public repository assets that support them. It is intended as a reviewer-facing navigation layer rather than a replacement for the manuscript captions.

## Method overview

### SafeTTA framework overview

Paper concept:

- SOURCE-only pre-adaptation branch: `Q66`
- transition-augmented pre-commit branch: `Q66 + DeltaSemantic64 = 130-D`

Repository sources:

```text
method_core/representation/
method_core/safety_estimator/
method_core/tta_actions/
experiments/R31_action_transfer/
```

Scientific boundary:

- SOURCE-only score is available before any TTA update.
- Transition-augmented score is available after reversible candidate execution but before commit/rollback.

## Action-transfer validation

Paper content:

- three-action transfer under TENT1, PL-CONF90, and MEMO-SEG4-1STEP;
- strict leave-one-action-out evaluation;
- comparison of Q66, DeltaSemantic64, shared Q66+DeltaS, action-conditioned, and separate-action references.

Repository source:

```text
experiments/R32_validation/
outputs/R31_R33/r32a_three_action_loao_summary.csv
```

Primary manuscript anchor:

```text
Shared Q66+DeltaS macro AUROC = 0.741600
```

Interpretation:

> action-transferable harmful-update structure, not complete action invariance.

## External joint domain + unseen-action shift

Paper content:

```text
NeoPolyp TENT1 + PL-CONF90
            ->
PolypGen MEMO-SEG4-1STEP
```

Repository sources:

```text
experiments/R33_external_joint_shift/
outputs/R31_R33/r33_polypgen_memo_primary_metrics.csv
outputs/R31_R33/r33_polypgen_memo_paired_deltas.csv
outputs/R31_R33/r33_claim_freeze_summary.json
```

Primary SafeTTA anchors:

```text
AUROC = 0.7432377136
AUPRC = 0.2554579725
HARM prevalence = 0.06549173194
AUPRC lift = 3.900614092x
AUROC 95% CI = [0.699569, 0.783784]
```

Paired comparison anchors:

```text
vs SicTTA-CCD:
  AUROC +0.135941 [0.072382, 0.193820]
  AUPRC +0.172134 [0.125599, 0.228141]

vs TEGDA-ADIC:
  AUROC -0.006501 [-0.043762, 0.028323]
  AUPRC +0.101055 [0.055974, 0.149594]

vs MC-dropout:
  AUROC +0.126395 [0.083388, 0.168958]
  AUPRC +0.157264 [0.114321, 0.210393]
```

## Controller-feasibility negative result

Paper role:

- negative diagnostic only;
- not a claimed successful controller contribution.

Repository sources:

```text
experiments/R31_controller_negative/
outputs/R31_R33/r31_controller_negative_summary.csv
```

This result is retained specifically to prevent the action-transfer ranking claim from being inflated into a controller claim.

## Legacy SOURCE-only reliability comparison

The earlier SOURCE-only PolypGen/TENT1 reliability endpoint remains under the v14/R17A layer and is scientifically distinct from the R33 unseen-MEMO endpoint.

Repository sources:

```text
experiments/R17A/
outputs/R17A/
provenance/R17A/
```

Do not mix the earlier SOURCE-only R17A endpoint numerically with the R33 transition-augmented endpoint.

## MRI framework replication

Paper content:

- SOURCE-only SafeTTA design re-developed on Prostate158;
- independent PROMISE12 evaluation;
- no direct transfer of the colonoscopy safety head.

Repository source:

```text
experiments/mri_prostate_promise/
```

The correct wording is **framework replication across task/modality shift**, not zero-shot colonoscopy-to-MRI safety-head transfer.
