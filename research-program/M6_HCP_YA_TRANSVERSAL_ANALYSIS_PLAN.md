# M6 HCP-YA Transversal Analysis Plan

**Status:** locked pending passed support preflight

**Model fitting enabled:** false

No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.

## Question

Can the same individual K/C_candidate/V coordinates transport across attention, working memory and reasoning while weights and layer-specific residuals vary by representational level?

## Gate Status

- Status: `blocked_preflight_not_passed`
- Preflight status: `data_support_incomplete`
- Preflight support passed: false
- Blocked reason: requires preflight status data_support_passed_ready_to_freeze_analysis with support_passed true

## Domain Plan

| Domain | Primary | Required | Analysis eligible | Status | Outcome columns | K columns after exclusion |
|---|---:|---:|---:|---|---|---|
| attention_control | false | true | true | analysis_eligible | CardSort_Unadj | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| wm_list_sorting | true | true | true | analysis_eligible | ListSort_Unadj | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| wm_nback | false | false | false | optional_known_issue_pending | WM_Task_2bk_Acc | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| reasoning_pmat | true | true | true | analysis_eligible | PMAT24_A_CR|PMAT24_A_RTCR | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| reasoning_relational | false | false | false | optional | Relational_Task_Acc | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |

## Model Sequence

| Domain | Order | Model | Predictors |
|---|---:|---|---|
| attention_control | 0 | K | K |
| attention_control | 1 | K_plus_C_candidate | K|C_candidate |
| attention_control | 2 | K_plus_V | K|V |
| attention_control | 3 | K_plus_C_candidate_plus_V | K|C_candidate|V |
| attention_control | 4 | K_plus_C_candidate_plus_V_plus_C_by_V | K|C_candidate|V|C_candidate_by_V |
| wm_list_sorting | 0 | K | K |
| wm_list_sorting | 1 | K_plus_C_candidate | K|C_candidate |
| wm_list_sorting | 2 | K_plus_V | K|V |
| wm_list_sorting | 3 | K_plus_C_candidate_plus_V | K|C_candidate|V |
| wm_list_sorting | 4 | K_plus_C_candidate_plus_V_plus_C_by_V | K|C_candidate|V|C_candidate_by_V |
| reasoning_pmat | 0 | K | K |
| reasoning_pmat | 1 | K_plus_C_candidate | K|C_candidate |
| reasoning_pmat | 2 | K_plus_V | K|V |
| reasoning_pmat | 3 | K_plus_C_candidate_plus_V | K|C_candidate|V |
| reasoning_pmat | 4 | K_plus_C_candidate_plus_V_plus_C_by_V | K|C_candidate|V|C_candidate_by_V |

## Layer-Specific Residual Gate

| Layer | Status | Candidate indicators | Outcome reuse allowed |
|---|---|---|---:|
| working_memory | blocked_until_independent_indicators_registered | reconstructed_2back_accuracy_future | false |
| reasoning | blocked_until_independent_indicators_registered | Relational_Task_Acc | false |

## Boundary

- Participant-level HCP data read by this plan: false
- Ordinary participant folds allowed: false
- Outcome columns reused to construct same-domain K/C_candidate/V: false
- Layer-specific residual factors registered: false
- NKI replication required for stronger transport claim: true
