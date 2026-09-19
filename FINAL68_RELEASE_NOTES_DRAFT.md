# SafeTTA Final68 — release notes draft

Proposed immutable tag after merge and final verification:

```text
v2.0-paper-final68
```

This tag is intentionally new and does not move or rewrite historical SafeTTA tags.

## Scientific change

The current paper-facing method is Final68:

```text
Geometry4 + Transition64 -> tri-state HARM / NEUTRAL / BENEFIT controller
```

This supersedes the historical SOURCE66 / 70-D / 130-D manuscript formulation for the current paper while preserving earlier tags as immutable provenance.

## Included compact public layer

- Final68 numerical anchor registry;
- Final68 lock/artifact SHA registry;
- Final68 claim-boundary document;
- frozen paper-facing figure-data CSVs;
- deterministic public paper-stat replay;
- exact frozen Final68 controller artifact.

Exact controller SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

The target68 NPY remains SHA-bound provenance only and is not redistributed in this compact release candidate.

## Cross-platform reproducibility

`.gitattributes` enforces LF checkout for Final68 frozen text/provenance assets so strict SHA256 replay remains stable on Windows and Linux.

## Release gate before tag

1. Windows/Linux public replay PASS on the final branch head.
2. PR #2 final audit PASS.
3. PR #2 merged to `main`.
4. Final merged commit SHA recorded.
5. Immutable tag created and verified.
6. Paper Code Availability updated with exact tag + commit.
