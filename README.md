# SafeTTA

**Prediction-Conditioned Safety Ranking for Test-Time Adaptation across Domain,
Action, and Task Shifts**

SafeTTA asks a pre-adaptation question: **should this sample be adapted before
the adaptation update is executed?**

The submission-locked v14 publication snapshot additionally audits whether
published current-prediction quality and uncertainty signals can substitute for
ranking the harm caused by a specified future TTA action.

## Reproducibility status

The public release was validated in a relocated workspace independent of the
original author project path.

### Frozen primary SafeTTA replay

- chain asset resolution: **43/43 PASS**
- canonical numeric provenance: **44/44 PASS**
- public paper-statistic replay: **PASS**
- legacy primary paper anchors: **44/44 PASS**
- all legacy anchor states: **PASS**
- minimal public release validation: **PASS**

### Additive R17A published-reliability audit

The v14 publication snapshot additionally includes the frozen head-to-head
sensitivity audit against SicTTA-CCD, TEGDA-ADIC, and MC-dropout.

- R17A published-reliability assets: **PASS**
- R17A frozen four-method point-metric anchors: **16/16 PASS**
- paired clustered-bootstrap audit: **PASS**
- SicTTA-CCD pooled-AUROC significance guard: **PASS**
- PraNet ADIC exclusion audit: **PASS**
- SegFormer CCD exclusion audit: **PASS**
- v14 publication-sync validation: **PASS**
- remote commit/tag verification: **PASS**

The R17A audit is **additive**. It does not modify the previously frozen
44-anchor SafeTTA primary replay or any frozen primary SafeTTA result.

### Submission-locked snapshot

- GitHub repository: `https://github.com/kewanglong22-lab/SafeTTA`
- branch: `main`
- tag: `v1.0-paper-v14`
- commit: `ca55622903ffe43065de2ed0ca558f0daf15aa7c`

The supported public claim is **paper-statistic reproducibility from released
frozen intermediate artifacts and publication-audit outputs**. Complete
bit-identical retraining of every historical upstream segmentation model is not
claimed.

## Frozen SafeTTA estimator

The frozen colonoscopy safety estimator is included under
`artifacts/colonoscopy_safety_estimator/`.

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

## Required replay asset

Download the GitHub Release asset:

```text
SafeTTA_replay_assets_v1.zip
```

Extract it into the repository root. It preserves the required `outputs/...`
relative paths.

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

See `ENVIRONMENT.md` for the distinction between the authoritative full lock and
the optional compact environment.

## Reproduce frozen primary paper statistics

From the repository root:

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root .
```

Expected result:

```text
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
```

These 44 anchors correspond to the **legacy frozen primary SafeTTA replay**.
The additive R17A publication audit is tracked separately and does not alter
this validated replay.

## Reproducibility boundary

The public package supports deterministic reproduction/verification of the
reported paper-level numeric anchors from frozen released intermediate
artifacts and the released R17A publication-audit outputs.

It does not claim:

- bit-identical retraining of every historical upstream segmentation model;
- redistribution of all source datasets or external ground truth;
- bit-identical wall-clock runtime across hardware.

The historical segmentation checkpoints and their retained provenance are
treated as frozen upstream experimental states. The public reproducibility
claim therefore focuses on the released SafeTTA estimator, frozen intermediate
outputs, paper statistics, and associated audit trail.

## License and third-party materials

Original SafeTTA code and materials that the authors are entitled to license
are released under Apache License 2.0. Third-party pretrained models, datasets,
software, and externally governed assets retain their original terms.

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation

A formal citation entry should be added after final bibliographic information is
available.

## Full method code

The repository includes the exact retained SafeTTA method implementation and
final experiment workflow, not only paper-table/statistical replay scripts.

Start with [`METHOD_CODE_INDEX.md`](METHOD_CODE_INDEX.md).

- `method_core/` — prediction-conditioned representation, frozen safety
  estimator, TTA actions, and MRI safety implementation.
- `experiments/` — source training, PolypGen, SUN-SEG, Prostate158/PROMISE12,
  R17A published-reliability analyses, and final paper analyses.
- `legacy_all_retained_code/` — retained SafeTTA-related historical code for
  provenance/completeness.
- `code/` — compact validated public paper-statistic replay scripts.
- `provenance/R17A/` — R17A frozen anchor and publication-sync provenance.
- `outputs/R17A/` — frozen R17A tabular/audit outputs included in the
  publication snapshot.

The primary public numerical replay remains deterministic reproduction of the
**44/44 legacy frozen SafeTTA anchors**. The R17A publication audit adds
**16/16 frozen four-method point-metric anchors** and associated paired
bootstrap/scope audits without changing the original primary replay.

The repository does not claim bit-identical retraining of every historical
upstream segmentation checkpoint.

## Portability

Exact frozen scripts may retain historical author-machine paths as defaults.
Reviewer-facing scripts with historical defaults expose CLI path overrides so
that the released workflows can be run from relocated workspaces. See
[`PORTABILITY.md`](PORTABILITY.md).

<!-- R17A_V14_PUBLICATION_SYNC_BEGIN -->
## Published reliability-baseline audit (R17A)

The v14 publication audit compares SafeTTA with three published reliability
baselines: SicTTA-CCD, TEGDA-ADIC, and MC-dropout.

**Locked interpretation.** Current prediction quality and uncertainty are
associated with adaptation harm, but they are not sufficient proxies for the
harm caused by a specified future TTA action.

The public R17A package is intended for paper-statistic replay and audit. It
contains lightweight scripts, frozen tabular results, paired clustered-bootstrap
outputs, and provenance records only. Large segmentation checkpoints and
source datasets remain external to the GitHub repository.

### Frozen four-method point metrics

| Method | Pooled AUROC | Pooled AUPRC | Macro-3 AUROC | Macro-3 AUPRC |
|---|---:|---:|---:|---:|
| SafeTTA | 0.616419085 | 0.491783038 | 0.616370607 | 0.492769742 |
| SicTTA-CCD | 0.634544123 | 0.459995939 | 0.634958120 | 0.459201008 |
| TEGDA-ADIC | 0.602545392 | 0.391803539 | 0.602608630 | 0.393504367 |
| MC-dropout | 0.446858833 | 0.326636094 | 0.446499781 | 0.328112952 |

### Scientific scope guardrails

- SicTTA-CCD is **not** described as being significantly worse than SafeTTA
  in pooled AUROC because the paired clustered-bootstrap confidence interval
  for the SafeTTA-minus-CCD difference crosses zero.
- SafeTTA-minus-SicTTA-CCD pooled-AUROC difference:
  `-0.018125038`, 95% CI `[-0.050926781, +0.016866089]`.
- TEGDA-ADIC is reproduced only where faithful native-dropout support exists.
- PraNet is excluded from ADIC when native `nn.Dropout` support is absent;
  dropout is not injected merely to create a baseline.
- SegFormer regenerated CCD states are not mixed with historical HARM labels
  when the regenerated SOURCE state is inconsistent with the historical frozen
  state.
- No target recalibration, target threshold tuning, feature selection, or
  post-hoc score reversal is used in the frozen R17A head-to-head audit.

### R17A public audit status

- frozen four-method point-metric anchors: **16/16 PASS**
- paired clustered-bootstrap audit: **PASS**
- SicTTA-CCD pooled-AUROC significance guard: **PASS**
- PraNet ADIC exclusion audit: **PASS**
- SegFormer CCD exclusion audit: **PASS**
- v14 publication-sync validation: **PASS**
- remote branch SHA verification: **PASS**
- remote publication-tag verification: **PASS**

The R17A audit is additive and does not modify the legacy **44/44** frozen
SafeTTA primary replay.

See `provenance/R17A/` for the frozen anchor manifest and R17B/R17C audit
records, and `outputs/R17A/` for the released frozen tabular/audit outputs.
<!-- R17A_V14_PUBLICATION_SYNC_END -->
