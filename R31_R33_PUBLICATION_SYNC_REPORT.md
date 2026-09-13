# R31-R33 publication-sync report

Status: **COMPACT_PUBLICATION_LAYER_READY; FULL R31-R33 SCRIPT SYNC PENDING**

The current PR contains the reviewer-facing compact publication layer:

- R31/R32/R33 experiment-directory README files;
- compact paper-facing outputs under `outputs/R31_R33/`;
- claim-boundary, figure/table mapping, release-checklist, and reproducibility documents;
- a numeric-anchor validating public replay script;
- updated root `README.md` and `METHOD_CODE_INDEX.md`.

It intentionally does **not** mutate or replace the validated `v1.0-paper-v14` tag or legacy 44-anchor replay.

## Current public-layer status

- compact R31-R33 outputs: **included**
- numeric-anchor replay: **included**
- claim boundaries: **included**
- reviewer documentation: **included**
- large checkpoints: **0**
- datasets / target GT: **0**
- DINOv2 weights: **0**
- exact final retained R31-R33 experiment scripts: **NOT YET SYNCHRONIZED IN THIS PR**
- mechanically failed but SHA-bound ancestor scripts: **NOT YET SYNCHRONIZED IN THIS PR**

## Exact final retained scripts expected for full method-audit completeness

### R31 action transfer

```text
Q1_R31A_action_conditional_harm_predictor_v1.py
Q1_R31B0_memo_seg4_third_action_protocol_lock_v1.py
Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py
Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py
Q1_R31B3_first_800case_gt_reveal_and_three_action_loao_v1.py
```

### R31 controller negative

```text
Q1_R31C0_multi_action_risk_controller_protocol_lock_v1.py
Q1_R31C1_nested_multi_action_risk_controller_oof_v1.py
Q1_R31D0_policy_level_crc_protocol_lock_v1.py
Q1_R31D1_nested_utility_first_policy_level_crc_oof_v1.py
```

### R32 validation

```text
Q1_R32A0_three_action_loao_comparison_protocol_lock_v1.py
Q1_R32A1_three_action_loao_comparison_execution_v1_fix1.py
Q1_R32B0_published_reliability_asset_comparability_audit_v1.py
Q1_R32B1_faithful_common_baseline_panel_lock_v1.py
Q1_R32B2A_exact_baseline_execution_context_bundle_v1.py
Q1_R32B2B_exact_ccd_adic_mc_score_generation_v1_fix1.py
Q1_R32B3_matched_three_action_harm_evaluation_v1_fix1.py
```

### R33 external joint shift

```text
Q1_R33A0_polypgen_joint_domain_action_shift_asset_audit_v1.py
Q1_R33A1_external_joint_shift_protocol_lock_v1.py
Q1_R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix2.py
Q1_R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1.py
Q1_R33A3_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1.py
Q1_R33A4_paper_integration_and_claim_freeze_v1.py
```

Total expected final retained scripts: **22**.

## Replay status

The compact replay now checks frozen numerical values rather than merely checking that files exist.

Expected command:

```bash
python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
```

Expected gate:

```text
R32A LOAO rows: 7/7 PASS
R33 primary rows: 4/4 PASS
R33 paired deltas: 6/6 PASS
R33 claim lock: PASS
GATE=PASS_R31_R33_PUBLIC_PAPER_STAT_REPLAY
```

## Merge boundary

If the intended public claim is only **paper-statistic reproducibility from compact frozen outputs**, the compact layer is sufficient after final replay audit.

If the intended repository is also to provide **exact R31-R33 method/protocol script auditability**, do **not** mark this PR final or create the new manuscript tag until the 22 retained scripts above are synchronized and checked.

After the final candidate is complete, rerun both the legacy 44-anchor replay and the R31-R33 replay before creating a new immutable tag.
