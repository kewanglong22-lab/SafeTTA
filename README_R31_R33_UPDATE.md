# SafeTTA R31-R33 public-repository update

This is a merge-ready additive update for the existing public SafeTTA repository (`v1.0-paper-v14`). It keeps the validated v14 44-anchor replay unchanged and adds the later action-transfer / unseen-action evidence used by the current manuscript.

## What is added
- R31: third-action MEMO development and strict three-action transfer workflow.
- R31C/R31D: negative controller-feasibility diagnostic.
- R32A: representation ablations and strict three-action LOAO comparison.
- R32B: faithful published-reliability matched audit.
- R33: protocol-locked `NeoPolyp TENT1+PL-CONF90 -> PolypGen MEMO` joint domain+unseen-action evaluation.
- portable paper-statistic replay for the public R31-R33 summary anchors.

## Reviewer-facing replay
After merging this directory into the public repository root:

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

## Important boundary
The exact retained experiment scripts preserve historical author-workspace assumptions and are provided for provenance/method audit. The portable public claim is deterministic reproduction of the released paper-facing numerical anchors from compact frozen outputs, consistent with the repository's existing reproducibility scope.

Do not rewrite tag `v1.0-paper-v14`. Merge this update, validate it, then create a new immutable manuscript tag.
