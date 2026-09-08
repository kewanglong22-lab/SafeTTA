# SafeTTA

**Prediction-Conditioned Safety Ranking for Test-Time Adaptation across Domain,
Action, and Task Shifts**

SafeTTA asks a pre-adaptation question: **should this sample be adapted before
the adaptation update is executed?**

## Reproducibility status

The public release was validated in a relocated workspace independent of the
original author project path.

- chain asset resolution: **43/43 PASS**
- canonical numeric provenance: **44/44 PASS**
- public paper-statistic replay: **PASS**
- paper anchors accounted: **44/44**
- all anchor states: **PASS**
- minimal public release validation: **PASS**

The supported public claim is **paper-statistic reproducibility from released
frozen intermediate artifacts**. Complete bit-identical retraining of every
historical upstream segmentation model is not claimed.

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

## Reproduce paper statistics

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

## Reproducibility boundary

The public package supports deterministic reproduction/verification of the
reported paper-level numeric anchors from frozen released intermediate
artifacts.

It does not claim:

- bit-identical retraining of every historical upstream segmentation model;
- redistribution of all source datasets or external ground truth;
- bit-identical wall-clock runtime across hardware.

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
  and final paper analyses.
- `legacy_all_retained_code/` — retained SafeTTA-related historical code for
  provenance/completeness.
- `code/` — compact validated public paper-statistic replay scripts.

The public numerical claim remains deterministic reproduction of 44/44
prespecified paper-level anchors from released frozen intermediate artifacts.
The repository does not claim bit-identical retraining of every historical
upstream segmentation checkpoint.

## Portability

Exact frozen scripts may retain historical author-machine paths as defaults.
All 65 reviewer-facing scripts that contain such defaults expose CLI path
overrides; no portable wrapper is required. See
[`PORTABILITY.md`](PORTABILITY.md).

<!-- R17A_V14_PUBLICATION_SYNC_BEGIN -->
## Published reliability-baseline audit (R17A)

The v14 publication audit compares SafeTTA with three published reliability
baselines: SicTTA-CCD, TEGDA-ADIC, and MC-dropout.

**Locked interpretation.** Current prediction quality and uncertainty are
associated with adaptation harm, but they are not sufficient proxies for the
harm caused by a specified future TTA action.

The public R17A package is intended for paper-statistic replay and audit. It
contains lightweight scripts, frozen tabular results, paired-bootstrap outputs,
and provenance records only. Large segmentation checkpoints and source datasets
remain external to the GitHub repository.

Scientific scope guardrails:

- SicTTA-CCD is not described as being significantly worse than SafeTTA on
  pooled AUROC when the paired clustered-bootstrap confidence interval crosses
  zero.
- TEGDA-ADIC is reproduced only where faithful native-dropout support exists.
- PraNet is excluded from ADIC when native `nn.Dropout` support is absent;
  dropout is not injected only to create a baseline.
- SegFormer regenerated CCD states are not mixed with historical harm labels
  when the regenerated SOURCE state is inconsistent with the historical state.

See `provenance/R17A/` for the frozen anchor manifest and R17B/R17C audit
records.
<!-- R17A_V14_PUBLICATION_SYNC_END -->
