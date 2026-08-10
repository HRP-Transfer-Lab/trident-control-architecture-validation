# M6 HCP-YA Transversal K/C Candidate/V Preflight

**Status:** data-support preflight only

**Model fitting allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.

## Question

Do lower-level K/C_candidate/V coordinates transport across attention/control, working memory and reasoning?

## Summary

- Status: `data_support_incomplete`
- Support passed: false
- Input path: `data\processed\hcp_ya_transversal_extract.csv`
- Input checksum: `sha256:6ddcf3a9463a0c1fa96639948ebb7a27107bfb1e99248096b0c28ac54f813a2e`

## Split Safeguard

- Family-isolated CV feasible: false
- Unrelated-only feasible: false
- Ordinary participant folds allowed: false

## Column Support

| Role | Variable/domain | Column | Required | Available | Nonmissing |
|---|---|---|---:|---:|---:|
| identity | participant_id | Subject | true | true | 0 |
| identity | family_id | Family_ID | true | true | 0 |
| predictor | K | ProcSpeed_Unadj | true | true | 0 |
| predictor | K | PicSeq_Unadj | true | true | 0 |
| predictor | K | ReadEng_Unadj | true | true | 0 |
| predictor | K | PicVocab_Unadj | true | true | 0 |
| predictor | C_candidate | Flanker_Unadj | true | true | 0 |
| predictor | V | SCPT_SEN | true | true | 0 |
| predictor | V | SCPT_SPEC | true | true | 0 |
| outcome | attention_control | CardSort_Unadj | true | true | 0 |
| outcome | wm_list_sorting | ListSort_Unadj | true | true | 0 |
| outcome | wm_nback | WM_Task_2bk_Acc | true | true | 0 |
| outcome | reasoning_pmat | PMAT24_A_CR | true | true | 0 |
| secondary_outcome | reasoning_pmat | PMAT24_A_RTCR | true | true | 0 |
| outcome | reasoning_relational | Relational_Task_Acc | false | true | 0 |

## Domain Support

| Domain | Primary | Required | Complete participants | Missing columns | Support passed |
|---|---:|---:|---:|---|---:|
| attention_control | false | true | 0 | none | false |
| wm_list_sorting | true | true | 0 | none | false |
| wm_nback | true | true | 0 | none | false |
| reasoning_pmat | true | true | 0 | none | false |
| reasoning_relational | false | false | 0 | none | false |

## Boundary

- Participant-level HCP data in Git: false
- Outcome columns reused to construct same-domain K/C_candidate/V: false by config validation
- Model outcomes interpreted: false
- Stage 1-3 reports changed: false
