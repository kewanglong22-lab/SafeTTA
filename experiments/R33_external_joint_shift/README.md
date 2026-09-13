# R33 — External Joint Domain + Unseen-Action Shift

## Purpose

R33 evaluates whether SafeTTA ranking transfers beyond the original action/domain setting under simultaneous:

- domain shift: NeoPolyp → PolypGen
- unseen action shift: TENT1 / PL-CONF90 → MEMO-SEG4-1STEP

## Protocol

Target:

- PolypGen
- DeepLabV3-R50
- 1532 physical cases
- 3 model states

The MEMO outcome is evaluated only after the prediction lock. No target calibration or predictor refitting is performed.

## Main result

SafeTTA-Q66+dSemantic64:

- AUROC: 0.743238
- AUROC 95% CI: [0.699569, 0.783784]
- AUPRC: 0.255458
- HARM prevalence: 0.065492
- AUPRC lift: 3.90x

The final claim is:

> external support for action-transferable future-HARM ranking under simultaneous domain and unseen-action shift.

A pristine prospective external validation claim is not made.
