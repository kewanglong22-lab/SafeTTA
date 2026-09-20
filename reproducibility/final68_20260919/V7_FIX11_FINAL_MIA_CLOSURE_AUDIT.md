# SafeTTA v7 fix11 - final MIA closure audit

## Scope
Text/reporting-only closure from v7_fix10. No model fitting, no threshold tuning, no new TTA action, no new dataset, and no change to frozen scientific point estimates.

## Changes
1. Abstract shortened to problem -> Final68 -> strongest paired component evidence -> held-out-action range -> setting-dependent deployment boundary.
2. QCResUNet comparator renamed to **QC proxy (QCResUNet-lineage)** wherever it appears as an experimental score comparator. The original QCResUNet paper remains cited in Related Work as literature. The manuscript now states that the frozen score vector is recoverable but the checkpoint/training/implementation provenance is not independently reconstructed.
3. Endpoint-margin sensitivity now reports the verified frozen E4E metrics at +/-0.01, +/-0.02, +/-0.03, and +/-0.05. The primary +/-0.02 support is 1,226 HARM / 5,063 NEUTRAL / 911 BENEFIT across the three held-out-action panels. Alternate-margin counts are not fabricated because the current paper-facing frozen summary does not bind them.
4. PCA scope is explicitly bounded: the retained lineage does not establish fold-local PCA refitting, so image-disjointness is claimed only for supervised StandardScaler/logistic-regression controller fitting.
5. Supplement section/table/figure/equation numbers use S-prefixes.
6. Table 2 external H-v-B AUROC CIs remain unfilled because no independently verified frozen intervals were recovered in the current manuscript-facing assets; no post-hoc values were invented.

## Frozen values preserved
- Table 4 paired deltas: +0.117610 [0.067823,0.165476], +0.059111 [0.030953,0.087875].
- E4D held-out AUROCs and deployment intervals unchanged.
- PolypGen and PROMISE12 deployment endpoints unchanged.
- Controller-only runtime boundary unchanged.

## Reproducibility additions
- `reproducibility/FINAL68_E4E_ENDPOINT_MARGIN_SUMMARY.csv` records the four frozen endpoint-margin metrics reported in Supplement Table S5.
- SHA256: `6891aed13b71289afd5f3fcf4e07f6d372677970c7bc0843e9950bac7ddf0b07`.
- Existing Table 4 paired-bootstrap script and summary are retained unchanged.

## Compile and layout gate
- Main: 27 pages; Supplement: 7 pages.
- Undefined citations/references: 0.
- Overfull boxes: 0 in main and Supplement.
- Render inspection covered the shortened abstract, PCA-scope paragraph, QC-proxy table/text, endpoint-margin table, S-numbering, and PROMISE12 aggregation section.
- `GATE=PASS_V7_FIX11_FINAL_MIA_CLOSURE_READY_FOR_RELEASE_SYNC`

## Remaining pre-submission release step
The existing public tag `v2.1-paper-final68-v7-20260920` predates the final Table-4 paired-CI and endpoint-margin reporting artifacts. It is therefore described as the corrected public **core** release rather than as the final manuscript-aligned release. The current manuscript source includes the exact new script/summaries; a new versioned public release must mirror them before submission, without moving either historical tag.
