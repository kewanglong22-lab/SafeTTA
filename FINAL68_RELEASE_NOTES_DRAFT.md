# SafeTTA Final68 — release notes draft

Proposed immutable tag after merge and verification:

```text
v2.0-paper-final68
```

This tag is intentionally new and does not move or rewrite historical SafeTTA tags.

## Scientific change

The current paper-facing method is Final68:

```text
Geometry4 + Transition64 -> tri-state HARM / NEUTRAL / BENEFIT controller
```

This supersedes the historical SOURCE66 / 70-D / 130-D manuscript formulation for the current paper, while preserving all earlier tags as immutable provenance.

## Included compact public layer

- Final68 numerical anchor registry.
- Final68 lock/artifact SHA registry.
- Final68 claim-boundary document.
- Frozen paper-facing figure-data CSVs.
- Deterministic public paper-stat replay.

## Not included by this compact commit

The exact binary controller joblib and target68 NPY are author-side frozen artifacts with recorded SHA256 values. They must only be added if their exact bytes are copied from the frozen author workspace and pass SHA verification.

## Release gate before tag

1. Public replay PASS.
2. Author binary sync PASS if binaries are redistributed.
3. PR merged to `main`.
4. Final merged commit SHA recorded.
5. Immutable tag created and verified.
6. Paper Code Availability updated with exact tag + commit.
