# R31-R33 claim boundaries

## Supported wording
- action-transferable future-HARM ranking
- partially action-invariant harmful-update structure
- transition-augmented pre-commit safety ranking
- external support under simultaneous domain + unseen-action shift

## Do not claim
- fully action-invariant or fully domain-invariant safety prediction
- pristine prospective R33 validation
- transition mode is strictly pre-update
- SafeTTA significantly outperforms TEGDA-ADIC in R33 AUROC
- validated high-coverage multi-action risk controller

## Endpoint separation
- SOURCE-only: 66-D, strictly pre-adaptation.
- Transition-augmented: 130-D Q66 + DeltaSemantic64, candidate action evaluated on a reversible shadow state before commit.
- R33 PolypGen/MEMO uses the transition-augmented endpoint and must not be numerically conflated with the earlier SOURCE-only PolypGen/TENT1 endpoint.
