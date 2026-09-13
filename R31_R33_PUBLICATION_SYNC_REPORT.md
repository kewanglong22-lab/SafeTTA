# R31-R33 publication-sync report

Status: **FULL R31-R33 SCRIPT LAYER SYNCHRONIZED; BOTH REPLAY GATES PASS; PR #1 MERGED; READY FOR IMMUTABLE MANUSCRIPT TAG**

The R31-R33 publication synchronization layer has been merged into `main` through PR #1.

## Included

- updated root `README.md` and `METHOD_CODE_INDEX.md`;
- four reviewer-facing R31-R33 experiment directories;
- **22 exact final retained R31-R33 experiment/protocol scripts**;
- **6 mechanically failed/replaced SHA-bound ancestor scripts** retained under `provenance/R31_R33/sha_bound_ancestors/`;
- compact paper-facing outputs under `outputs/R31_R33/`;
- claim-boundary, figure/table mapping, release-checklist and reproducibility documents;
- `R31_R33_CODE_INDEX.csv` and `R31_R33_SHA256.csv`;
- numeric-anchor validating R31-R33 public replay;
- synchronization helper under `tools/`.

It does **not** mutate or replace the validated `v1.0-paper-v14` tag.

## Exact-script synchronization audit

Bundle SHA256:

```text
185e4dbc8af3fd19b91a423462e4f8f5f39f05ee2ad49493bd49147c61c318de
```

Exact-script synchronization commit:

```text
dabf3519499f1bd1e656e485b4585b8cd85da48c
```

Audit results:

```text
Final retained scripts: 22/22 PASS
SHA-bound ancestors:   6/6 present
SHA verification:      28/28 PASS
py_compile:             28/28 PASS
```

## R31-R33 replay status

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

## Legacy v14 44-anchor replay status

Observed final gate:

```text
Chain gate: PASS_CHAIN_ASSET_RESOLUTION (43/43)
Canonical provenance: PASS_CANONICAL_NUMERIC_PROVENANCE (44/44)
R10L0: PASS numeric_summary_pairs=3
R15A1: PASS numeric_summary_pairs=6
R15B1: PASS numeric_summary_pairs=8
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
Decision=READY_FOR_MINIMAL_PUBLIC_RELEASE_VALIDATION
```

Therefore both manuscript-level replay gates are PASS.

## Legacy replay reproducibility boundary

The full author-side 44-anchor validation uses retained frozen intermediate assets that are intentionally not all redistributed in the compact public tree. In particular, R15A1 requires the frozen `R15A0_all_8method_target_predictions.csv` panel recorded by the R15A0 lock (50,580,135 bytes; SHA256 `cff9b08f06dbafa0816bd4f4f732af66304761bf9105d3106b725686f3bf961e`). The public release should not claim that a clean public clone alone regenerates every historical upstream/intermediate artifact from zero.

## Repository safety

- raw datasets added: **0**
- target GT added: **0**
- large segmentation checkpoints added: **0**
- DINOv2 weights redistributed: **0**
- frozen `v1.0-paper-v14` rewritten: **NO**

## Merge verification

PR #1 merged successfully with merge commit:

```text
ba8facea49d13c01ac8fcf18100beef671d93f20
```

The previous immutable paper tag was re-verified and remains unchanged:

```text
v1.0-paper-v14 -> ca55622903ffe43065de2ed0ca558f0daf15aa7c
```

## Remaining release step

1. create the new immutable manuscript tag `v1.1-paper-r31-r33` at the final release snapshot commit;
2. verify the tag remotely;
3. update manuscript Data/Code Availability with the tag and final release snapshot commit SHA.
