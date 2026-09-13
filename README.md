# SafeTTA

**Prediction-Conditioned Safety Ranking for Test-Time Adaptation across Domain, Action, and Task Shifts**

SafeTTA studies whether a test-time adaptation (TTA) update is likely to harm a segmentation result. The current manuscript distinguishes two safety modes:

1. **SOURCE-only pre-adaptation ranking** — a 66-D prediction-conditioned representation is computed from the current image and frozen SOURCE prediction before any TTA update.
2. **Transition-augmented pre-commit ranking** — a specified candidate TTA action is executed only on a reversible shadow copy; a 64-D candidate semantic transition is combined with the 66-D SOURCE state, yielding a 130-D risk representation before the candidate state is committed.

The transition-augmented mode uses **no explicit ActionID** and no target-label calibration.

---

## Reproducibility status

### Frozen v14 primary release

The original submission-locked release remains immutable:

- repository: `https://github.com/kewanglong22-lab/SafeTTA`
- tag: `v1.0-paper-v14`
- commit: `ca55622903ffe43065de2ed0ca558f0daf15aa7c`
- chain asset resolution: **43/43 PASS**
- canonical numeric provenance: **44/44 PASS**
- legacy primary paper anchors: **44/44 PASS**
- public paper-statistic replay: **PASS**

The additive R17A published-reliability audit also remains unchanged and does not modify the legacy 44-anchor replay.

### R31-R33 manuscript extension

The current manuscript additionally includes:

- **R31** — third-action MEMO development and action-transfer analysis;
- **R31C/R31D** — negative multi-action controller-feasibility diagnostic;
- **R32** — strict three-action leave-one-action-out (LOAO) validation;
- **R33** — protocol-locked `NeoPolyp TENT1 + PL-CONF90 -> PolypGen MEMO` simultaneous domain + unseen-action evaluation.

The R31-R33 public synchronization layer is additive. It does **not** rewrite `v1.0-paper-v14`.

---

## Key R31-R33 result

For the external PolypGen unseen-MEMO evaluation:

| Method | AUROC | AUPRC | HARM prevalence | AUPRC lift |
|---|---:|---:|---:|---:|
| SafeTTA-Q66+DeltaS | 0.743238 | **0.255458** | 0.065492 | **3.90x** |
| TEGDA-ADIC | **0.749739** | 0.154403 | 0.065492 | 2.36x |
| MC-dropout | 0.616843 | 0.098194 | 0.065492 | 1.50x |
| SicTTA-CCD | 0.607297 | 0.083324 | 0.065492 | 1.27x |

SafeTTA AUROC 95% physical-image clustered-bootstrap CI:

```text
[0.699569, 0.783784]
```

SafeTTA AUPRC lift over HARM prevalence:

```text
3.900614x
```

Paired SafeTTA-minus-TEGDA-ADIC results:

```text
AUROC delta = -0.006501, 95% CI [-0.043762, +0.028323]
AUPRC delta = +0.101055, 95% CI [+0.055974, +0.149594]
```

The supported interpretation is **action-transferable future-HARM ranking with external support under simultaneous domain and unseen-action shift**. The repository does not claim full action invariance, full domain invariance, or pristine prospective R33 validation.

---

## Quick reviewer replay

### 1. Legacy frozen primary paper statistics

Download and extract the release asset:

```text
SafeTTA_replay_assets_v1.zip
```

Then run from repository root:

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root .
```

Expected terminal gate:

```text
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
```

### 2. R31-R33 compact manuscript replay

Run:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Expected:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

The R31-R33 replay verifies released compact manuscript-level numerical anchors. It does not retrain upstream segmentation models, rerun all TTA/DINO inference, or access private/raw datasets.

---

## Frozen SOURCE-only estimator

The frozen colonoscopy SOURCE-only estimator is under:

```text
artifacts/colonoscopy_safety_estimator/
```

| Artifact | SHA256 |
|---|---|
| PCA64 | `2c6c289ce395105a0c79f7b7751df2a195d76a248c22b43e817d71511c4a24f1` |
| Safety head | `602574418a309400bf7e7fc854267a702496c1bd47e4f07f6f60415ec9bd14ce` |
| Frozen threshold | `4a37fb4f9a9b729a9aa6551b1417afea7348e2dfecdfcb6e0e2b5986e748e4d7` |

Frozen operating threshold:

```text
tau = 0.300584763193734
```

DINOv2-base weights are not redistributed. See `THIRD_PARTY_NOTICES.md`.

---

## Repository layout

```text
SafeTTA/
├── method_core/
│   ├── representation/
│   ├── safety_estimator/
│   └── tta_actions/
├── experiments/
│   ├── source_training/
│   ├── polypgen/
│   ├── sunseg/
│   ├── mri_prostate_promise/
│   ├── paper_analysis/
│   ├── R31_action_transfer/
│   ├── R31_controller_negative/
│   ├── R32_validation/
│   └── R33_external_joint_shift/
├── outputs/
│   ├── R17A/
│   └── R31_R33/
├── provenance/
│   ├── R17A/
│   └── R31_R33/
├── artifacts/
├── code/
└── legacy_all_retained_code/
```

Reviewer-facing method navigation starts at [`METHOD_CODE_INDEX.md`](METHOD_CODE_INDEX.md).

R31-R33 paper-output provenance is documented under [`provenance/R31_R33/`](provenance/R31_R33/).

---

## Scientific claim boundaries

Supported wording includes:

- action-transferable future-HARM ranking;
- partially action-invariant harmful-update structure;
- transition-augmented pre-commit safety ranking;
- external support under simultaneous domain + unseen-action shift.

Do **not** interpret the release as evidence for:

- full action invariance;
- full domain invariance;
- pristine prospective R33 validation;
- transition-mode risk computed strictly before the candidate action is simulated;
- SafeTTA significantly outperforming TEGDA-ADIC in R33 AUROC;
- a validated high-coverage multi-action risk controller.

See [`provenance/R31_R33/CLAIM_BOUNDARIES.md`](provenance/R31_R33/CLAIM_BOUNDARIES.md).

---

## Environment

For the full frozen dependency set:

```bash
conda env create -f environment.yml
conda activate safetta
```

For a smaller convenience environment:

```bash
conda env create -f environment_minimal.yml
conda activate safetta-minimal
```

See `ENVIRONMENT.md` for the authoritative/full versus convenience environment distinction.

---

## Reproducibility boundary

The public repository supports deterministic reproduction/verification of released paper-level numerical anchors from frozen intermediate artifacts and compact audit outputs.

It does not claim:

- bit-identical retraining of every historical upstream segmentation checkpoint;
- redistribution of all source datasets or external ground truth;
- redistribution of third-party DINOv2 weights;
- bit-identical wall-clock runtime across hardware.

Historical segmentation checkpoints are treated as frozen upstream experimental states. Exact retained scripts may preserve historical author-workspace defaults; reviewer-facing workflows expose portable entry points where practical. See `PORTABILITY.md`.

---

## R17A reliability-baseline audit

The v14 additive reliability audit compares SOURCE-only SafeTTA with SicTTA-CCD, TEGDA-ADIC, and MC-dropout on the earlier PolypGen/TENT1 endpoint.

| Method | Pooled AUROC | Pooled AUPRC |
|---|---:|---:|
| SafeTTA | 0.616419085 | 0.491783038 |
| SicTTA-CCD | 0.634544123 | 0.459995939 |
| TEGDA-ADIC | 0.602545392 | 0.391803539 |
| MC-dropout | 0.446858833 | 0.326636094 |

SafeTTA-minus-SicTTA-CCD pooled-AUROC difference:

```text
-0.018125038, 95% CI [-0.050926781, +0.016866089]
```

Therefore SicTTA-CCD is **not** described as significantly worse than SOURCE-only SafeTTA in pooled AUROC. This earlier endpoint is scientifically distinct from the later transition-augmented R33 unseen-MEMO endpoint.

---

## License and third-party materials

Original SafeTTA code/materials that the authors are entitled to license are released under Apache License 2.0. Third-party pretrained models, datasets, software, and externally governed assets retain their original terms.

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation

A formal citation entry will be added after final bibliographic information is available.
