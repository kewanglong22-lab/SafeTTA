# R31-R33 Release Checklist

This checklist is the final pre-merge / pre-tag gate for the SafeTTA R31-R33 manuscript synchronization layer.

## A. Scientific scope

- [x] R31 action-transfer scope documented.
- [x] R31C/R31D controller-feasibility result retained as a negative diagnostic.
- [x] R32 strict three-action LOAO result documented.
- [x] R33 simultaneous domain + unseen-action result documented.
- [x] SOURCE-only `Q66` and transition-augmented `Q66 + DeltaSemantic64` are explicitly separated.
- [x] Transition mode is described as **pre-commit**, not strictly pre-update.
- [x] R33 is not described as pristine prospective validation.
- [x] No claim of full action invariance or full domain invariance.
- [x] No claim that SafeTTA significantly beats TEGDA-ADIC in R33 AUROC.
- [x] No claim of a validated high-coverage multi-action controller.

## B. Reviewer-facing documentation

- [x] Root `README.md` synchronized with the current manuscript framing.
- [x] `METHOD_CODE_INDEX.md` updated with R31-R33 reviewer entry points.
- [x] `experiments/R31_action_transfer/README.md` present.
- [x] `experiments/R31_controller_negative/README.md` present.
- [x] `experiments/R32_validation/README.md` present.
- [x] `experiments/R33_external_joint_shift/README.md` present.
- [x] `FIGURE_TABLE_MAPPING.md` present.
- [x] `CLAIM_BOUNDARIES.md` present.
- [x] `REPRODUCIBILITY_SCOPE.md` present.
- [x] `RETAINED_SCRIPT_MANIFEST.md` defines the exact 22 final scripts and six optional SHA-bound ancestors.

## C. Compact public outputs

Expected under `outputs/R31_R33/`:

- [x] `r31_controller_negative_summary.csv`
- [x] `r32a_three_action_loao_summary.csv`
- [x] `r33_polypgen_memo_primary_metrics.csv`
- [x] `r33_polypgen_memo_paired_deltas.csv`
- [x] `r33_claim_freeze_summary.json`
- [x] `README.md`

## D. R31-R33 exact execution-script audit layer

For a **full method/protocol audit release** rather than only a compact paper-statistic release:

- [x] Exact 22-script target manifest frozen in `RETAINED_SCRIPT_MANIFEST.md`.
- [ ] All 22 final retained R31-R33 experiment scripts synchronized into the grouped `experiments/` directories.
- [ ] Script contents checked against the frozen author-workspace versions.
- [ ] Script SHA256 manifest generated/verified for the public tree.
- [ ] Mechanically failed but SHA-bound ancestors either synchronized under a clearly marked provenance directory or explicitly omitted from the public claim.

Until these items are complete, describe the current PR as the **compact R31-R33 publication/replay layer**, not the complete 22-script execution lineage.

## E. Public replay

Legacy v14 replay command:

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root .
```

Required gate before final tag:

```text
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
```

R31-R33 replay command:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Required gate before final tag:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

Interim audit status:

- [x] R31-R33 replay implementation upgraded from file-presence checks to frozen numeric-anchor validation.
- [x] Replay script syntax-checked and run successfully against the exact released compact table values before synchronization.
- [x] PR compact-layer changed-file list inspected; no dataset/checkpoint additions were introduced.
- [ ] Legacy 44-anchor replay rerun on the final merged candidate.
- [ ] R31-R33 compact replay rerun on the final merged candidate.
- [ ] `PUBLIC_REPLAY_REPORT.txt` refreshed from the final candidate state.

The final-candidate items remain unchecked until the tree that will actually be tagged is audited.

## F. Repository / release safety

- [x] Existing immutable tag `v1.0-paper-v14` is not rewritten.
- [x] R31-R33 changes are isolated on a separate synchronization branch / PR before merge.
- [x] `main` remains at the frozen pre-merge head during synchronization.
- [x] No raw datasets are added.
- [x] No target ground-truth files are added.
- [x] No DINOv2 weights are redistributed.
- [x] No large segmentation checkpoints are added as part of the compact R31-R33 synchronization layer.
- [x] Reproducibility boundary is stated explicitly.

## G. Final merge / tag gate

Do **not** create the final manuscript tag until all items required by the chosen release scope are complete.

For either release scope:

- [x] Inspect PR changed-file list and confirm the compact-layer files are intentional.
- [ ] Confirm GitHub reports the PR mergeable against current `main` immediately before merge.
- [ ] Run / verify the legacy v14 replay on the final candidate.
- [ ] Run / verify the R31-R33 compact replay on the final candidate.
- [ ] Refresh `PUBLIC_REPLAY_REPORT.txt` with final PASS state and candidate/merged commit SHA.
- [ ] Merge PR into `main`.
- [ ] Verify merged `main` commit SHA remotely.
- [ ] Create a **new immutable manuscript tag**; do not move or overwrite `v1.0-paper-v14`.
- [ ] Update manuscript Data/Code Availability with the new tag and commit SHA.

If the repository is advertised as including the full R31-R33 method/protocol execution lineage, all remaining items in Section D must also be completed before merge/tag.

## Recommended tag naming

Preferred:

```text
v1.1-paper-r31-r33
```

Alternative if this becomes the exact submission snapshot:

```text
v1.1-mia-submission
```

Only one immutable final manuscript tag should be advertised in the submitted manuscript.
