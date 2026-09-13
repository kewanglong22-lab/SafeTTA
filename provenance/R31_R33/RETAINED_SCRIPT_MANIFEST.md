# R31-R33 Retained Script Manifest

This manifest defines the **22 final retained R31-R33 execution scripts** expected for a full method/protocol audit release. It is separate from the compact paper-statistic replay layer already synchronized in PR #1.

Current public-branch audit status: **0/22 exact execution scripts present in the grouped R31-R33 experiment directories**. The four experiment directories currently contain reviewer-facing `README.md` files only.

Do not mark the repository as a complete R31-R33 execution-lineage release until all 22 files below are synchronized from the frozen author workspace and verified by content hash / syntax checks.

## R31 action transfer — 5 scripts

Target directory: `experiments/R31_action_transfer/`

1. `Q1_R31A_action_conditional_harm_predictor_v1.py`
2. `Q1_R31B0_memo_seg4_third_action_protocol_lock_v1.py`
3. `Q1_R31B1_memo_seg4_action_design_and_lr_freeze_v1.py`
4. `Q1_R31B2_800case_memo_prediction_and_semantic_lock_v1.py`
5. `Q1_R31B3_first_800case_gt_reveal_and_three_action_loao_v1.py`

## R31 controller negative — 4 scripts

Target directory: `experiments/R31_controller_negative/`

6. `Q1_R31C0_multi_action_risk_controller_protocol_lock_v1.py`
7. `Q1_R31C1_nested_multi_action_risk_controller_oof_v1.py`
8. `Q1_R31D0_policy_level_crc_protocol_lock_v1.py`
9. `Q1_R31D1_nested_utility_first_policy_level_crc_oof_v1.py`

## R32 validation — 7 scripts

Target directory: `experiments/R32_validation/`

10. `Q1_R32A0_three_action_loao_comparison_protocol_lock_v1.py`
11. `Q1_R32A1_three_action_loao_comparison_execution_v1_fix1.py`
12. `Q1_R32B0_published_reliability_asset_comparability_audit_v1.py`
13. `Q1_R32B1_faithful_common_baseline_panel_lock_v1.py`
14. `Q1_R32B2A_exact_baseline_execution_context_bundle_v1.py`
15. `Q1_R32B2B_exact_ccd_adic_mc_score_generation_v1_fix1.py`
16. `Q1_R32B3_matched_three_action_harm_evaluation_v1_fix1.py`

## R33 external joint shift — 6 scripts

Target directory: `experiments/R33_external_joint_shift/`

17. `Q1_R33A0_polypgen_joint_domain_action_shift_asset_audit_v1.py`
18. `Q1_R33A1_external_joint_shift_protocol_lock_v1.py`
19. `Q1_R33A2A_exact_polypgen_memo_transition_context_bundle_v1_fix2.py`
20. `Q1_R33A2B_external_polypgen_memo_transition_score_lock_v1_fix1.py`
21. `Q1_R33A3_polypgen_memo_harm_reveal_external_joint_shift_evaluation_v1.py`
22. `Q1_R33A4_paper_integration_and_claim_freeze_v1.py`

## SHA-bound mechanically failed ancestors

These are not part of the 22 final retained execution scripts, but the frozen provenance package also tracked the following failed/replaced ancestors for chronology and SHA binding:

- R32A1 original (pre-fix1)
- R32B2B original (pre-fix1)
- R32B3 original (pre-fix1)
- R33A2A original
- R33A2A fix1
- R33A2B original (pre-fix1)

Preferred provenance location if publicly synchronized:

```text
provenance/R31_R33/sha_bound_ancestors/
```

These ancestor files must be clearly labeled as mechanically failed/replaced and must not be presented as recommended execution entry points.

## Required audit after synchronization

For each of the 22 final scripts:

1. file exists at the expected grouped path;
2. Python AST / `py_compile` passes;
3. file content matches the frozen retained source used in the manuscript workflow;
4. SHA256 is recorded in the public manifest;
5. no dataset, checkpoint, DINOv2 weight, GT, or large intermediate asset is accidentally embedded;
6. historical absolute author-workspace paths, if retained for provenance, are disclosed in the reproducibility boundary.

After all 22 pass, update `RELEASE_CHECKLIST.md` Section D and only then advertise a complete R31-R33 method/protocol execution lineage.
