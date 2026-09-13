# SafeTTA R31-R33 public-repository update

This is an additive update for the existing public SafeTTA repository (`v1.0-paper-v14`). It keeps the validated v14 44-anchor replay unchanged and adds the later action-transfer / unseen-action evidence used by the current manuscript.

## What is already synchronized in this PR

- R31/R32/R33 reviewer-facing protocol documentation;
- compact R31-R33 paper-facing outputs;
- deterministic numeric-anchor replay for the new manuscript results;
- claim boundaries and reproducibility scope;
- figure/table provenance mapping;
- root README and method-code index updates.

The exact final retained R31-R33 execution scripts are a separate method-audit layer. They are **not yet all present in this PR** and should be synchronized before the repository is described as containing the complete R31-R33 execution lineage.

## Scientific additions represented by the compact layer

- R31: third-action MEMO development and strict three-action transfer workflow.
- R31C/R31D: negative controller-feasibility diagnostic.
- R32A: representation ablations and strict three-action LOAO comparison.
- R32B: matched published-reliability comparison.
- R33: protocol-locked `NeoPolyp TENT1+PL-CONF90 -> PolypGen MEMO` joint domain+unseen-action evaluation.

## Reviewer-facing replay

From repository root:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Expected:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

The replay validates the frozen numerical anchors themselves, rather than only checking that the compact files exist.

## Important boundary

The portable public claim supported by the compact layer is deterministic reproduction of the released paper-facing numerical anchors from frozen outputs.

Do not state that the current PR already contains the complete 22-script R31-R33 execution lineage until those exact retained scripts are synchronized.

Do not rewrite tag `v1.0-paper-v14`. Complete the final candidate, validate both replay layers, then create a new immutable manuscript tag.
