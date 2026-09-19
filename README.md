# SafeTTA

**SafeTTA: Prediction-Conditioned Safety Ranking for Medical Test-Time Adaptation under Domain, Action, and Task Shifts**

SafeTTA asks a deployment-oriented question: after a specified test-time adaptation candidate has been produced, should that candidate replace the frozen SOURCE prediction?

## Current manuscript method: Final68

The current paper-facing method is:

```text
Final68 = Geometry4 + Transition64
```

- **Geometry4**: prediction-to-prediction Dice, IoU, foreground-area change, and boundary-density change.
- **Transition64**: candidate-conditioned minus SOURCE-conditioned semantic PCA64 representation, built from frozen DINOv2 patch features.
- **Controller**: class-balanced tri-state logistic regression predicting HARM / NEUTRAL / BENEFIT.
- **Utility**: `u = P(BENEFIT) - P(HARM)`.
- **Decision**: accept candidate if `u > 0`; otherwise retain SOURCE.
- **Timing**: pre-commit, not strictly pre-update. The candidate prediction exists before the safety decision, but SOURCE remains recoverable.

No explicit ActionID is supplied to the Final68 controller.

## Final68 reproducibility layer

Current compact paper-facing provenance is under:

```text
reproducibility/final68_20260919/
```

Reviewer-facing replay:

```bash
python code/Q1_Final68_public_paper_stat_replay_v1.py --root .
```

Expected gate:

```text
GATE=PASS_FINAL68_PUBLIC_PAPER_STAT_REPLAY
```

The exact frozen Final68 controller is released at:

```text
artifacts/final68/FINAL68_E1B1_SOURCE_ONLY_TRISTATE_CONTROLLER.joblib
```

Expected SHA256:

```text
8dfa218efe259e9ee0205fde1abc525583d3156f0491cd0143e541b835d4f3f4
```

The compact replay verifies frozen paper-facing numerical anchors, figure-data tables, and SHA-bound artifacts. It does **not** retrain segmentation models, rerun TTA/DINO inference, or access target ground truth.

## Current Final68 evidence

### External PolypGen

| Metric | Final68 |
|---|---:|
| HARM-vs-rest AUROC | 0.5798 |
| HARM-vs-rest AUPRC | 0.2683 |
| HARM-vs-BENEFIT AUROC | 0.6886 |
| HARM-vs-BENEFIT AUPRC | 0.5338 |
| SOURCE Dice | 0.75490 |
| Deployed Dice | 0.75692 |
| Deployed − SOURCE | +0.00202 |

The frozen PolypGen operating-point deployment improvement is **not statistically significant**.

### Strict three-action LOAO

| Held-out action | HARM-vs-BENEFIT AUROC | HARM rollback | BENEFIT accept |
|---|---:|---:|---:|
| TENT1 | 0.8408 | 85.40% | 73.11% |
| PL-CONF90 | 0.7774 | 50.00% | 88.73% |
| MEMO | 0.8339 | 82.12% | 69.96% |

### PROMISE12 prostate-MRI replication

| Metric | Final68 |
|---|---:|
| HARM-vs-rest AUROC | 0.6802 |
| HARM-vs-BENEFIT AUROC | 0.7708 |
| SOURCE Dice | 0.74525 |
| Deployed Dice | 0.75135 |
| Deployed − SOURCE | +0.00610 |

PROMISE12 is an **independently re-fitted framework replication**, not zero-shot transfer of the exact colonoscopy controller.

## Deployment interpretation

The current manuscript explicitly separates:

1. **ranking quality**;
2. **HARM prevention**;
3. **BENEFIT retention**; and
4. **deployed segmentation quality**.

A high global HARM AUROC does not necessarily produce better deployment if the ranking is overly conservative.

## Controller-only overhead

Once the 68-D representation is already available:

```text
mean single-case CPU latency: 0.0869 ms
serialized controller size:   5,188 bytes
```

These numbers describe the **controller only** and are not end-to-end SafeTTA latency.

## Historical releases

Earlier immutable SafeTTA releases remain available for reproducibility. They include SOURCE66 / SOURCE+simple70 / SOURCE+semantic-transition130 experiments and the earlier R31-R33 extension.

Those historical representations and results are scientifically distinct from Final68 and are **not** used as current Final68 evidence. In particular, the old PolypGen full-transition result `0.7432 / 0.2555` belongs to the historical 130-D method.

The original frozen v14 tag remains immutable:

```text
v1.0-paper-v14
```

Historical cross-platform releases `v1.0.0` and `v1.0.1` also remain unchanged.

## Claim boundaries

See:

```text
reproducibility/final68_20260919/CLAIM_BOUNDARIES.md
```

Key boundaries:

- do not claim Full68 has the best global HARM AUROC;
- do not claim significant improvement at the frozen PolypGen operating point;
- do not claim uniform superiority over QCResUNet;
- do not call PROMISE12 zero-shot controller transfer;
- do not call 0.087 ms end-to-end latency;
- do not interpret `u=0` as a clinically calibrated optimum.

## Reproducibility scope

The public repository supports deterministic verification of released paper-level numerical anchors from frozen compact evidence and SHA-bound artifacts.

It does not claim:

- bit-identical retraining of every historical segmentation checkpoint;
- redistribution of raw datasets or external ground truth;
- redistribution of third-party DINOv2 weights;
- bit-identical wall-clock runtime across hardware.

## License

Original SafeTTA materials that the authors are entitled to license are released under Apache License 2.0. Third-party models, datasets, and software retain their original terms.

## Citation

A formal citation entry will be added after final bibliographic information is available.
