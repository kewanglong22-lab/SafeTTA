# Final68 publication synchronization report

Status: **TEXTUAL/COMPACT PUBLIC LAYER PREPARED; AUTHOR BINARY ASSET SYNC STILL REQUIRED BEFORE IMMUTABLE TAG**

Target repository:

```text
kewanglong22-lab/SafeTTA
```

Synchronization branch:

```text
final68-public-sync-20260919
```

Historical tags/releases must remain unchanged.

## Compact layer contents

- updated root README candidate;
- `reproducibility/final68_20260919/`;
- `code/Q1_Final68_public_paper_stat_replay_v1.py`;
- draft release notes.

## Required next author-side binary check

Exact controller:

```text
FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
SHA256 8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

Exact target68 matrix:

```text
FINAL68_E1B0_POLYPGEN_DEEPLAB_MEMO_68D_PREOUTCOME.npy
SHA256 0640a87e087398ff68a79229509da614451310c924c270ca718e0a235c9a88ab
```

Do not reconstruct either artifact.

## Final merge/tag policy

Do not create the proposed immutable `v2.0-paper-final68` tag until:
- exact binary sync is either completed or explicitly excluded from release scope;
- public replay passes from a clean candidate tree;
- final merged commit SHA is known.
