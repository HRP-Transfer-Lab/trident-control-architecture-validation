# M6.1 HCP-YA Transversal Analysis Plan

**Status:** pre-outcome frozen analysis protocol

**Model fitting enabled after freeze:** true

No Trident-G/APC/PACE validation claim, transfer claim, g confirmation, C_signal confirmation, V mechanism confirmation, W-specific capacity claim, bottleneck claim, neural-criticality claim or cusp claim is authorised.

## Question

Do the same lower-level K/C_candidate/V coordinates provide out-of-sample predictive information across working memory and fluid reasoning, with attention/control as a secondary near-domain comparison?

## Gate Status

- Status: `ready_to_run_frozen_analysis`
- Preflight status: `data_support_passed_ready_to_freeze_analysis`
- Preflight support passed: true

## Domain Plan

| Domain | Role | Primary | Analysis eligible | Status | Outcome columns | K columns after exclusion |
|---|---|---:|---:|---|---|---|
| wm_list_sorting | primary | true | true | analysis_eligible | ListSort_Unadj | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| reasoning_pmat | primary | true | true | analysis_eligible | PMAT24_A_CR | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| attention_control | secondary_near_domain | false | true | analysis_eligible | CardSort_Unadj | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| wm_nback | blocked_optional | false | false | optional_known_issue_pending | WM_Task_2bk_Acc | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |
| reasoning_relational | blocked_optional | false | false | blocked | Relational_Task_Acc | PicSeq_Unadj|PicVocab_Unadj|ProcSpeed_Unadj|ReadEng_Unadj |

## Model Sequence

| Domain | Order | Model | Predictors |
|---|---:|---|---|
| wm_list_sorting | 0 | M0_intercept | intercept |
| wm_list_sorting | 1 | M1_K | K_candidate |
| wm_list_sorting | 2 | M2_K_plus_C | K_candidate|C_candidate |
| wm_list_sorting | 3 | M3_K_plus_V | K_candidate|V_candidate |
| wm_list_sorting | 4 | M4_K_plus_C_plus_V | K_candidate|C_candidate|V_candidate |
| wm_list_sorting | 5 | M5_K_plus_C_plus_V_plus_C_by_V | K_candidate|C_candidate|V_candidate|C_by_V |
| reasoning_pmat | 0 | M0_intercept | intercept |
| reasoning_pmat | 1 | M1_K | K_candidate |
| reasoning_pmat | 2 | M2_K_plus_C | K_candidate|C_candidate |
| reasoning_pmat | 3 | M3_K_plus_V | K_candidate|V_candidate |
| reasoning_pmat | 4 | M4_K_plus_C_plus_V | K_candidate|C_candidate|V_candidate |
| reasoning_pmat | 5 | M5_K_plus_C_plus_V_plus_C_by_V | K_candidate|C_candidate|V_candidate|C_by_V |
| attention_control | 0 | M0_intercept | intercept |
| attention_control | 1 | M1_K | K_candidate |
| attention_control | 2 | M2_K_plus_C | K_candidate|C_candidate |
| attention_control | 3 | M3_K_plus_V | K_candidate|V_candidate |
| attention_control | 4 | M4_K_plus_C_plus_V | K_candidate|C_candidate|V_candidate |
| attention_control | 5 | M5_K_plus_C_plus_V_plus_C_by_V | K_candidate|C_candidate|V_candidate|C_by_V |

## M6.2/M6.3 Gates

| Gate | Status | Latent capacity claim allowed |
|---|---|---:|
| M6.2 | blocked_pending_independent_second_WM_indicator | false |
| M6.3 | blocked_until_m6_2_establishes_defensible_independent_layer_specific_WM_candidate | false |

## Boundary

- Participant-level HCP data read by this plan: false
- Ordinary participant folds allowed: false
- Outcome columns reused to construct K/C_candidate/V: false
- M5 interaction is secondary only and low precision
- N=100 exploratory unrelated-subject test only
