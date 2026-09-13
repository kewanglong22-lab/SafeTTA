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

## C. Compact public outputs

Expected under `outputs/R31_R33/`:

- [x] `r31_controller_negative_summary.csv`
- [x] `r32a_three_action_loao_summary.csv`
- [x] `r33_polypgen_memo_primary_metrics.csv`
- [x] `r33_polypgen_memo_paired_deltas.csv`
- [x] `r33_claim_freeze_summary.json`
- [x] `README.md`

## D. Public replay

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

Current status:

- [ ] Legacy 44-anchor replay rerun on the final merged candidate.
- [ ] R31-R33 compact replay rerun on the final merged candidate.
- [ ] `PUBLIC_REPLAY_REPORT.txt` refreshed from the final candidate state.

These items remain intentionally unchecked until the final candidate tree is audited.

## E. Repository / release safety

- [x] Existing immutable tag `v1.0-paper-v14` is not rewritten.
- [x] R31-R33 changes are isolated on a separate synchronization branch / PR before merge.
- [x] No raw datasets are added.
- [x] No target ground-truth files are added.
- [x] No DINOv2 weights are redistributed.
- [x] No large segmentation checkpoints are added as part of the compact R31-R33 synchronization layer.
- [x] Reproducibility boundary is stated explicitly.

## F. Final merge / tag gate

Do **not** create the final manuscript tag until all items below are complete:

- [ ] Inspect PR changed-file list and confirm no unintended files.
- [ ] Confirm PR is mergeable against current `main`.
- [ ] Run / verify the legacy v14 replay on the final candidate.
- [ ] Run / verify the R31-R33 compact replay on the final candidate.
- [ ] Refresh `PUBLIC_REPLAY_REPORT.txt` with final PASS state and commit SHA.
- [ ] Merge PR into `main`.
- [ ] Verify merged `main` commit SHA remotely.
- [ ] Create a **new immutable manuscript tag**; do not move or overwrite `v1.0-paper-v14`.
- [ ] Update manuscript Data/Code Availability with the new tag and commit SHA.

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
