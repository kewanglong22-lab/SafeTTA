# R31-R33 Claim Boundaries

This file freezes the wording allowed for the current SafeTTA manuscript extension. It exists to keep the manuscript, README, code index, figures, and public replay scientifically consistent.

## 1. Primary supported claim

Preferred manuscript wording:

> **Action-transferable future-HARM ranking with external support under simultaneous domain + unseen-action shift.**

This wording is supported by the strict three-action LOAO evaluation (R32) and the PolypGen unseen-MEMO evaluation (R33).

## 2. Supported secondary wording

The following phrases are allowed when their scope is explicit:

- **action-transferable future-HARM ranking**;
- **partially action-invariant harmful-update structure**;
- **transition-augmented pre-commit safety ranking**;
- **external support under simultaneous domain + unseen-action shift**;
- **framework replication across task/modality shift** for the MRI study;
- **SOURCE-only pre-adaptation safety ranking** for the frozen 66-D branch.

## 3. Two safety modes must remain distinct

### SOURCE-only mode

```text
Q66 = 64-D prediction-conditioned semantic state + 2-D mask morphology
```

Properties:

- computed from the current image and frozen SOURCE prediction;
- no candidate TTA update is executed;
- therefore genuinely **pre-adaptation**.

### Transition-augmented mode

```text
Q66 + DeltaSemantic64 = 130-D
```

Properties:

- a specified candidate TTA action is executed only on a reversible shadow copy;
- the candidate mask is used to derive the 64-D semantic transition;
- risk is computed before the candidate state is committed or rolled back;
- therefore the correct term is **pre-commit**, not strictly pre-update;
- no explicit ActionID is required by the primary shared transition predictor.

## 4. Claims that are not allowed

Do **not** claim:

- full action invariance;
- full domain invariance;
- universal or action-agnostic safety prediction without qualification;
- pristine prospective R33 external validation;
- that R33 risk is computed strictly before candidate-action execution;
- that SafeTTA significantly outperforms TEGDA-ADIC in R33 AUROC;
- that SafeTTA is uniformly superior to all reliability baselines on all metrics;
- a successful high-coverage multi-action risk controller;
- that MEMO consistently improves segmentation;
- zero-shot transfer of the colonoscopy safety head to MRI.

## 5. R33 external-evaluation wording

Allowed:

> R33 provides external support under simultaneous domain and unseen-action shift.

Not allowed:

> R33 is a pristine prospective external validation.

Reason: the PolypGen target cohort had historical GT access in the broader project lineage. The R33 protocol itself was locked before the new MEMO execution/outcome construction, but the target dataset cannot be represented as never previously accessed.

## 6. Reliability-baseline wording

### Earlier SOURCE-only PolypGen/TENT1 endpoint

The legacy R17A comparison is a SOURCE-only reliability audit. SicTTA-CCD has a numerically higher pooled AUROC than SafeTTA and the paired AUROC CI crosses zero.

Therefore do not write that SOURCE-only SafeTTA significantly beats CCD on pooled AUROC.

### R33 unseen-MEMO endpoint

SafeTTA versus TEGDA-ADIC:

```text
AUROC delta = -0.006501
95% CI = [-0.043762, +0.028323]

AUPRC delta = +0.101055
95% CI = [+0.055974, +0.149594]
```

Correct wording:

> SafeTTA has comparable AUROC to TEGDA-ADIC on R33 while achieving a statistically supported AUPRC gain for rare future-HARM retrieval.

## 7. Controller boundary

R31C/R31D are retained as a negative feasibility diagnostic.

Correct interpretation:

- reliable ranking does not automatically imply a good intervention policy;
- operating-point choice, action availability, and utility/cost trade-offs remain separate problems.

The manuscript contribution is **safety ranking**, not a solved multi-action controller.

## 8. MRI boundary

The Prostate158/PROMISE12 study re-develops the SOURCE-only SafeTTA framework in a different task/modality.

Correct wording:

> framework replication across task/modality shift.

Do not describe it as direct colonoscopy-to-MRI transfer of the frozen colonoscopy safety head.

## 9. Endpoint separation rule

Never numerically merge or conflate:

- the earlier SOURCE-only PolypGen/TENT1 reliability endpoint;
- the R32 three-action LOAO endpoint;
- the R33 PolypGen/MEMO transition-augmented endpoint;
- the MRI SOURCE-only endpoint.

They answer related but distinct scientific questions.
