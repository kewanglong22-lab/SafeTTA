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
- [x] `RETAINED_SCRIPT_MANIFEST.md` defines the exact 22 final scripts and six SHA-bound ancestors.

## C. Compact public outputs

Expected under `outputs/R31_R33/`:

- [x] `r31_controller_negative_summary.csv`
- [x] `r32a_three_action_loao_summary.csv`
- [x] `r33_polypgen_memo_primary_metrics.csv`
- [x] `r33_polypgen_memo_paired_deltas.csv`
- [x] `r33_claim_freeze_summary.json`
- [x] `README.md`

## D. R31-R33 exact execution-script audit layer

- [x] Exact 22-script target manifest frozen in `RETAINED_SCRIPT_MANIFEST.md`.
- [x] All 22 final retained R31-R33 experiment scripts synchronized into the grouped `experiments/` directories.
- [x] Script contents checked against the frozen author-workspace bundle.
- [x] Public-tree SHA256 manifest generated and used for 28/28 copy verification.
- [x] Six mechanically failed/replaced SHA-bound ancestors synchronized under `provenance/R31_R33/sha_bound_ancestors/`.
- [x] `py_compile` passed for all 28 synchronized Python files.

Exact-script synchronization evidence:

```text
Bundle SHA256:
185e4dbc8af3fd19b91a423462e4f8f5f39f05ee2ad49493bd49147c61c318de

Synchronization commit:
dabf3519499f1bd1e656e485b4585b8cd85da48c

SHA verification: 28/28 PASS
py_compile: 28/28 PASS
```

## E. Public replay

R31-R33 replay command:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Observed on the synchronized candidate:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

Legacy v14 replay command:

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root . --replay-root <fresh-directory-under-outputs>
```

Observed on the final synchronized candidate tree (source-tree commit `409f86a9e68ff178609ecec1bb4ca7b1ca3bfd00`):

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

Current status:

- [x] R31-R33 replay validates frozen numeric anchors.
- [x] R31-R33 compact replay rerun successfully on the synchronized candidate.
- [x] Legacy 44-anchor replay rerun successfully on the synchronized candidate.
- [x] `PUBLIC_REPLAY_REPORT.txt` records both PASS gates.
- [x] Public reproducibility boundary records that the full author-side 44-anchor replay uses retained frozen intermediate assets not all redistributed in the compact public tree.

## F. Repository / release safety

- [x] Existing immutable tag `v1.0-paper-v14` is not rewritten.
- [x] R31-R33 changes remain isolated on a synchronization branch / PR before merge.
- [x] `main` remains at the frozen pre-merge head during synchronization.
- [x] No raw datasets are added.
- [x] No target ground-truth files are added.
- [x] No DINOv2 weights are redistributed.
- [x] No large segmentation checkpoints are added.
- [x] Reproducibility boundary is stated explicitly.

## G. Final merge / tag gate

- [x] PR changed-file list inspected; intended publication/provenance/script/synchronization files only.
- [x] Exact-script synchronization and SHA audit PASS.
- [x] R31-R33 numeric replay PASS.
- [x] Legacy v14 44-anchor replay PASS.
- [x] Legacy replay PASS recorded in release provenance.
- [ ] Re-confirm GitHub mergeability after the final provenance commits.
- [ ] Mark PR ready for review / final merge.
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
