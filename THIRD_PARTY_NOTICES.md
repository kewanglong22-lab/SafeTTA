# Third-Party Notices

SafeTTA is released under Apache License 2.0 only for code and original
materials that the authors are entitled to license. This repository does not
relicense third-party software, pretrained models, datasets, or externally
governed assets.

## DINOv2

The method uses `facebook/dinov2-base`.

- model identifier: `facebook/dinov2-base`
- frozen revision used in the paper: commit prefix `f9e44c814b77203e`
- frozen weight SHA256 prefix recorded by the project: `d73036b56966966d`

DINOv2 weights are not redistributed by this repository. Obtain them from the
upstream distribution and comply with its applicable terms.

## Medical datasets

The study uses public/external datasets including NeoPolyp, PolypGen, SUN-SEG,
Prostate158, and PROMISE12. Dataset images and annotations are not bundled in
the main repository. Users are responsible for obtaining access from the
original maintainers and complying with the applicable dataset terms.

## Historical segmentation checkpoints

Historical segmentation checkpoints are not required for the validated
Level-A paper-statistic replay and are not included in the main public release.

## Python dependencies

Packages recorded in `environment.yml`, `requirements-lock.txt`, and
`environment/environment_freeze.txt` retain their respective licenses.

## Replay assets

`SafeTTA_replay_assets_v1.zip` contains frozen intermediate artifacts required
for the validated public 44-anchor paper-statistic replay. Inclusion for
scientific reproducibility does not supersede any applicable third-party rights.
