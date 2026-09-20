# Final68 v7 corrected public-layer synchronization report

Status: **CORRECTED PUBLIC LAYER STAGED ON PR BRANCH; HISTORICAL TAGS UNCHANGED; MERGE + NEW TAG PENDING**

Target repository:

```text
kewanglong22-lab/SafeTTA
```

Current correction branch:

```text
final68-v7-public-repair-20260920
```

Historical tag `v2.0-paper-final68` remains immutable and resolves to commit:

```text
84c907bcd91de228c5f480659f1d9c29e39d0195
```

It is retained as pre-correction provenance and must not be moved, deleted, or reassigned.

## Why the v7 correction exists

The manuscript-level P0-P7 audit identified four active public-layer issues that required repair without rewriting historical artifacts:

1. historical E4C was still presented as the primary/strict LOAO line;
2. two figure-data CSVs had insufficient independent provenance for current paper claims;
3. `0.087 ms` could be misread as end-to-end SafeTTA latency;
4. the public replay validated the recorded controller SHA string without hashing the actual controller file bytes.

The corrected branch repairs these public-facing issues while preserving historical files/tags as provenance.

## Corrected active evidence line

### Primary action shift

The primary action-shift analysis is the grouped physical-image-disjoint held-out-action analysis:

```text
TENT1      H-v-B AUROC 0.817440  [0.784944, 0.846591]
PL-CONF90  H-v-B AUROC 0.729658  [0.679815, 0.777933]
MEMO       H-v-B AUROC 0.735553  [0.687708, 0.783476]
```

Deployment boundaries:

```text
TENT1      deployed-SOURCE +0.005709  CI [-0.002026, 0.013128]
PL-CONF90  deployed-SOURCE +0.011781  CI [ 0.008599, 0.015475]
MEMO       absolute deployment N/A
```

Historical E4C remains secondary provenance and is not described as strict LOAO.

### Endpoint margin

The frozen primary HARM/NEUTRAL/BENEFIT Dice margin is `±0.02`. Margins `±0.01`, `±0.03`, and `±0.05` are sensitivity analyses only.

### Runtime

The approximately `0.087 ms/case` measurement is the controller-only stage **after Final68 already exists**. It excludes candidate TTA execution, segmentation inference, DINOv2 extraction, Geometry4, and Transition64 construction.

## Public asset repair

Removed from the corrected active layer:

```text
reproducibility/final68_20260919/figure_data/fig2_coverage_utility_frozen_points.csv
reproducibility/final68_20260919/figure_data/fig3_action_shift_loao_frozen_points.csv
```

Their historical SHA256 values remain documented in `FINAL68_LOCKS.json` for traceability.

Added canonical current action-shift figure data:

```text
reproducibility/final68_20260919/figure_data/fig2_e4d_action_shift_frozen_points.csv
```

Expected SHA256:

```text
36cb4a79697bd3aeffc63eaafaf97b6e72ba998ad6fc23425c9e942007af7558
```

The frozen margin-sensitivity data remain active:

```text
figS1_margin_sensitivity_frozen_points.csv
SHA256 96b2a58cffa3c799619e097a86f6165009b6ef01b954b5bcc60637dd14638fef
```

## Controller byte binding

Exact controller path:

```text
artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
```

Expected SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

The corrected replay opens this file in binary mode, computes SHA256 from its actual bytes, and compares that value to the frozen expected digest.

## Corrected replay

Command:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected final gate:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

The replay additionally checks that the two excluded unsafe figure-data files are absent and that historical E4C/E7 groups are not part of the active numeric-anchor registry.

## Remaining release steps

1. review the correction branch diff;
2. run the corrected public replay from a clean checkout;
3. open and review the PR against `main`;
4. merge the PR;
5. record the merged commit SHA;
6. create a **new** immutable corrected tag; do not reuse or move `v2.0-paper-final68`;
7. update manuscript Code Availability with the new tag and merged commit SHA.
