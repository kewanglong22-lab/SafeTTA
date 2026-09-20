# SafeTTA Final68 fix11 manuscript-aligned release sync

This sync extends the corrected Final68 v7 public core with the final paper-facing evidence additions used by the MIA fix11 manuscript.

## Added reproducibility assets

- `reproducibility/final68_20260919/FINAL68_TABLE4_PAIRED_BOOTSTRAP_SUMMARY.csv`
- `reproducibility/final68_20260919/FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv`
- `reproducibility/final68_20260919/Q1_Final68_Table4_paired_bootstrap_ci_v1.py`
- `reproducibility/final68_20260919/V7_FIX11_FINAL_MIA_CLOSURE_AUDIT.md`
- `code/Q1_Final68_fix11_release_asset_audit_v1.py`

## Final68 component evidence

On the frozen PolypGen component-ablation panel, paired physical-image-clustered bootstrap gives:

- Final68 - Geometry4 HARM-vs-BENEFIT AUROC: `+0.117610`, 95% CI `[0.067823, 0.165476]`.
- Final68 - Transition64 HARM-vs-BENEFIT AUROC: `+0.059111`, 95% CI `[0.030953, 0.087875]`.

Both intervals exclude zero. These results support incremental directional discrimination from combining Geometry4 and Transition64. They do not imply that Final68 maximizes global HARM-vs-rest AUROC or guarantees deployment Dice improvement.

## Endpoint-margin sensitivity

The primary HARM/NEUTRAL/BENEFIT Dice margin remains `±0.02`. Fixed-score sensitivity results are:

| Margin | H-v-B AUROC | H-v-B AUPRC | HARM rollback | BENEFIT accepted | Role |
|---:|---:|---:|---:|---:|---|
| ±0.01 | 0.722411 | 0.716600 | 0.716243 | 0.695829 | sensitivity |
| ±0.02 | 0.760884 | 0.737825 | 0.710086 | 0.742179 | primary |
| ±0.03 | 0.770854 | 0.737988 | 0.693338 | 0.773107 | sensitivity |
| ±0.05 | 0.782102 | 0.740094 | 0.673712 | 0.798075 | sensitivity |

The alternate margins are sensitivity analyses only and are not used to select the primary endpoint.

## Comparator provenance boundary

The historical score previously labeled QCResUNet in the comparator panel is described in the final manuscript as `QC proxy (QCResUNet-lineage)`. The pre-outcome frozen score vector is recoverable, but the original checkpoint, training data, and full implementation provenance cannot be independently reconstructed from the retained assets. It is therefore a score-level sensitivity comparator, not a claimed faithful end-to-end QCResUNet reproduction.

## PCA / LOAO boundary

The retained lineage does not establish fold-local PCA64 refitting for every grouped LOAO split. Image-disjointness is therefore claimed for supervised `StandardScaler + logistic-regression` controller fitting, where validation-image rows are excluded from the corresponding fit. No stronger claim is made for the upstream SOURCE-development PCA transform.

## Release gates

Core corrected replay:

```text
GATE=PASS_FINAL68_V7_CORRECTED_PUBLIC_PAPER_STAT_REPLAY
```

Fix11 release-asset audit:

```text
GATE=PASS_FINAL68_V7_FIX11_RELEASE_ASSET_AUDIT
```

Historical tags `v2.0-paper-final68` and `v2.1-paper-final68-v7-20260920` must remain unchanged.
