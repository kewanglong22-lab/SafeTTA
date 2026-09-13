# R31-R33 publication-sync report

Status: **FULL R31-R33 SCRIPT LAYER SYNCHRONIZED; LEGACY FINAL REPLAY PENDING**

The current PR now contains both the compact reviewer-facing publication layer and the complete retained R31-R33 execution-script audit layer.

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

It does **not** mutate or replace the validated `v1.0-paper-v14` tag or legacy 44-anchor replay.

## Exact-script synchronization audit

Bundle SHA256:

```text
185e4dbc8af3fd19b91a423462e4f8f5f39f05ee2ad49493bd49147c61c318de
```

Synchronized candidate commit:

```text
dabf3519499f1bd1e656e485b4585b8cd85da48c
```

Audit results:

```text
Final retained scripts: 22/22 PASS
SHA-bound ancestors:   6/6 present
SHA verification:      28/28 PASS
py_compile:             28/28 PASS
R31-R33 numeric replay: PASS
```

The synchronization commit added exactly the intended 22 final scripts plus six ancestor scripts. The PR changed-file list has been inspected and contains only the intended publication, provenance, script and synchronization files.

## R31-R33 replay status

The public replay validates frozen manuscript-level numerical anchors rather than only checking file presence.

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

## Repository safety

- raw datasets added: **0**
- target GT added: **0**
- large segmentation checkpoints added: **0**
- DINOv2 weights redistributed: **0**
- frozen `v1.0-paper-v14` rewritten: **NO**

## Remaining pre-merge gate

The only substantive replay gate still pending is the **legacy 44-anchor replay on the final synchronized candidate**:

```bash
python code/Q1_SAFETTA_public_paper_stat_replay_v3_fix2.py --root .
```

Required terminal gate:

```text
Anchors accounted: 44/44
All anchor states PASS: True
GATE=PASS_PUBLIC_PAPER_STATISTIC_REPLAY
```

After that PASS is recorded, re-confirm PR mergeability, update final provenance, merge into `main`, verify the merged commit, and create a new immutable manuscript tag without moving `v1.0-paper-v14`.
