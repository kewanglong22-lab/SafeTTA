# Final68 publication synchronization report

Status: **COMPACT PUBLIC LAYER + EXACT FROZEN CONTROLLER SYNCHRONIZED; WINDOWS LF NORMALIZATION FIX ADDED; PUBLIC REPLAY RE-RUN REQUIRED BEFORE MERGE**

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
- draft Final68 release notes.

## Exact frozen controller synchronization

Author-side sync observed:

```text
controller: source_sha=8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
controller: COPY=PASS -> artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
GATE=PASS_FINAL68_AUTHOR_BINARY_ASSET_SYNC
```

The controller was committed to the synchronization branch in commit:

```text
b4fcfcf  Add exact frozen Final68 controller artifact
```

The target68 NPY remains SHA-bound provenance only and is not redistributed in this compact layer.

## Windows replay diagnostic

The first Windows replay after controller synchronization produced SHA failures for all three figure-data CSVs while all 155 numeric-anchor rows matched. The observed hashes exactly equal the CRLF-converted forms of the frozen LF files:

```text
fig2 CRLF SHA 914a805c3a819a8960fc010b183579cad3139a1f41455c6813339d8676498594
fig3 CRLF SHA 55cf34c37d70cc86f0d060413bbcb0837c435744b0ed797b85ca24b646527fe7
figS1 CRLF SHA f1d4cb0193e297fcab92384b7365d99f58c1a3cb9d13e62ee53363a454fff8b5
```

This is a checkout line-ending issue, not a numerical/data mismatch. `.gitattributes` now enforces LF for the Final68 reproducibility CSV/JSON/Markdown layer.

## Final merge/tag gate

Do not merge or tag until:

1. the current Windows clone pulls the updated `.gitattributes`;
2. the three figure-data CSV worktree files are rewritten to LF;
3. `Q1_Final68_public_paper_stat_replay_v1.py --root .` ends with `GATE=PASS_FINAL68_PUBLIC_PAPER_STAT_REPLAY`;
4. PR #2 is re-audited on the final branch head;
5. the PR is merged into `main`;
6. the final merged commit SHA is recorded;
7. immutable tag `v2.0-paper-final68` is created and verified;
8. manuscript Code Availability is updated with the exact tag + commit.
