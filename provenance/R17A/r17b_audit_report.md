# SafeTTA R17B v14 GitHub Sync Audit

- Script: `2026-09-09-R17B-v1-fix3`
- Repo: `F:\MEDSEG_SAFETTA\release\SafeTTA_clean_test_v102`
- R17A root: `F:\MEDSEG_SAFETTA`
- Final status: **PASS_READY_TO_SYNC**

## Frozen R17A numeric anchors

- Point anchors: **16/16 PASS**
- SicTTA-CCD pooled-AUROC paired-bootstrap numbers found: **PASS**
- SicTTA-CCD pooled-AUROC CI crosses zero: **PASS**

## Scientific guardrails

- repo_is_git_worktree: **PASS**
- repo_origin_matches_kewanglong22_lab_SafeTTA: **PASS**
- public_repo_has_no_blocking_large_or_model_checkpoint_files: **PASS**
- r17a_ccd_assets_found: **PASS**
- r17a_adic_assets_found: **PASS**
- r17a_mc_dropout_assets_found: **PASS**
- r17a_bootstrap_assets_found: **PASS**
- r17a_frozen_point_anchors_16_of_16: **PASS**
- sict_pooled_auroc_bootstrap_values_found: **PASS**
- sict_pooled_auroc_ci_crosses_zero: **PASS**
- no_sict_significance_overclaim_detected: **PASS**
- pranet_adic_exclusion_evidence_found: **PASS**
- segformer_ccd_exclusion_evidence_found: **PASS**

## Publication-sync rule

Only rows marked `COPY_CANDIDATE` in `r17a_copy_candidates.csv` should be considered for the public v14 incremental sync.
Files marked `DO_NOT_COPY` must remain outside the public repo.

The script does **not** modify the existing 44/44 legacy frozen anchors.
R17A anchors are additive publication anchors for v14.

