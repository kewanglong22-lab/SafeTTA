# Final68 publication synchronization report

Status: **COMPACT PUBLIC LAYER + EXACT FROZEN CONTROLLER SYNCHRONIZED; WINDOWS LF ISSUE RESOLVED; FINAL68 PUBLIC REPLAY PASS; READY FOR PR FINAL REVIEW**

Target repository:

```text
kewanglong22-lab/SafeTTA
```

Synchronization branch:

```text
final68-public-sync-20260919
```

Historical tags/releases remain unchanged.

## Compact layer contents

- updated root README for Final68;
- `reproducibility/final68_20260919/` numerical/provenance layer;
- exact paper-facing figure-data CSVs;
- Final68 lock/artifact SHA registry;
- claim-boundary document;
- `code/Q1_Final68_public_paper_stat_replay_v1.py`;
- exact frozen Final68 controller;
- draft Final68 release notes.

## Exact frozen controller synchronization

Author-side sync observed:

```text
controller: source_sha=8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
controller: COPY=PASS -> artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
GATE=PASS_FINAL68_AUTHOR_BINARY_ASSET_SYNC
```

The controller was committed in:

```text
b4fcfcf  Add exact frozen Final68 controller artifact
```

The target68 NPY remains SHA-bound provenance only and is not redistributed in this compact layer.

## Windows line-ending audit

The first Windows replay after controller synchronization failed only the three figure-data SHA checks because the worktree files had CRLF line endings. All 155 numerical anchor rows still matched.

`.gitattributes` now enforces LF for Final68 frozen CSV/JSON/Markdown assets. After LF normalization, the exact frozen figure-data hashes are:

```text
fig2  b47c30e3cf56173e7b075b5876c5364720759cd9e9dda49e55bd419b39ae198e
fig3  68790ec7cb6a17cdf73932b87d34f3946f7689bdb245a63c39cb7ba90d422225
figS1 96b2a58cffa3c799619e097a86f6165009b6ef01b954b5bcc60637dd14638fef
```

The re-run on Windows completed with:

```text
Anchor rows: 155
Required anchors: 24
Missing: 0
Mismatched: 0
FIGURE_DATA fig2_coverage_utility_frozen_points.csv: PASS
FIGURE_DATA fig3_action_shift_loao_frozen_points.csv: PASS
FIGURE_DATA figS1_margin_sensitivity_frozen_points.csv: PASS
FINAL68_CONTROLLER_SHA=PASS
FINAL68_TARGET68_SHA=PASS
LEGACY_SIGNATURE_EXCLUSION=PASS
GATE=PASS_FINAL68_PUBLIC_PAPER_STAT_REPLAY
```

The CRLF event is therefore resolved as a checkout-byte issue, not a numerical or scientific-data mismatch.

## PR audit status

PR #2 has been checked against the current Final68 manuscript boundaries:

- Final68 = Geometry4 + Transition64;
- tri-state HARM / NEUTRAL / BENEFIT controller;
- no explicit ActionID;
- pre-commit, not strictly pre-update;
- PolypGen frozen operating-point gain remains non-significant;
- QCResUNet is not claimed to be uniformly worse;
- PROMISE12 is framework replication, not zero-shot controller transfer;
- 0.087 ms is controller-only overhead;
- historical 66-D / 70-D / 130-D results remain historical evidence only.

The duplicate root-level author-sync helper was removed; the canonical helper remains under `code/`.

## Remaining release steps

1. mark PR #2 ready for review;
2. re-confirm mergeability at the final branch head;
3. merge PR #2 into `main`;
4. record the final merged commit SHA;
5. create and verify immutable tag `v2.0-paper-final68`;
6. update manuscript Code Availability with the exact repository, tag, and commit SHA.
