# Portability

The reviewer-facing SafeTTA repository preserves the **exact frozen method and
experiment scripts** used by the final study.

## Frozen path defaults

Some frozen scripts retain their original author-side Windows paths as default
values for provenance. The reviewer-portability audit verified that all
reviewer-facing scripts containing such defaults expose command-line path
overrides.

Validated reviewer-facing audit:

- Python files scanned: 65
- `PATH_DEFAULT_WITH_CLI`: 65
- `NEEDS_PORTABLE_WRAPPER`: 0
- syntax failures: 0
- unresolved project-local imports: 0
- blocking issues: 0

Therefore, users should pass their own dataset/output/artifact paths through the
script CLI rather than relying on historical default paths.

## Recommended entry points

See `METHOD_CODE_INDEX.md` for the final paper-facing implementation and
experiment map.

The directory `legacy_all_retained_code/` is included for provenance and
reviewer completeness. Historical exploratory scripts in that directory are
not all intended as portable entry points.

## Reproducibility scope

The validated public numerical replay reproduces 44/44 prespecified paper-level
numerical anchors from released frozen intermediate artifacts.

Complete bit-identical retraining of every historical upstream segmentation
checkpoint is outside the claimed reproducibility scope because complete
per-image training manifests/checkpoints were not retained for every historical
source state.

### Cross-platform Git line endings

The public replay verifies the SHA256 of frozen Python implementations.
`.gitattributes` therefore enforces `LF` checkout for `*.py`, including on
Windows systems with `core.autocrlf=true`. This prevents line-ending conversion
from changing frozen script bytes while leaving the Python source semantics
unchanged.
