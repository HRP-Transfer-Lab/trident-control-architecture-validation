# M6 HCP-YA Transversal K/C Candidate/V Preflight

**Status:** data-support preflight only

**Model fitting allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.

## Question

Do lower-level K/C_candidate/V coordinates transport across attention/control, working memory and reasoning?

## Summary

- Status: `data_support_passed_ready_to_freeze_analysis`
- Support passed: true
- Input path: `data\processed\hcp_ya_transversal_extract.csv`
- Input checksum: `sha256:b43456153602ece24b2e8ea283b68e4faca3146f2a9b373588913704ae86069b`
- Input unique subjects: 100
- Cohort mode: `hcp_100_unrelated`

## Split Safeguard

- Split strategy: `participant_isolated_official_hcp_100_unrelated`
- Family-isolated CV feasible: false
- Unrelated-only feasible: true
- Participant-isolated CV allowed: true
- Ordinary participant folds allowed: false

## Column Support

| Role | Variable/domain | Column | Required | Available | Nonmissing |
|---|---|---|---:|---:|---:|
| identity | participant_id | Subject | true | true | 100 |
| predictor | K | ProcSpeed_Unadj | true | true | 100 |
| predictor | K | PicSeq_Unadj | true | true | 100 |
| predictor | K | ReadEng_Unadj | true | true | 100 |
| predictor | K | PicVocab_Unadj | true | true | 100 |
| predictor | C_candidate | Flanker_Unadj | true | true | 100 |
| predictor | V | SCPT_SEN | true | true | 100 |
| predictor | V | SCPT_SPEC | true | true | 100 |
| outcome | attention_control | CardSort_Unadj | true | true | 100 |
| outcome | wm_list_sorting | ListSort_Unadj | true | true | 100 |
| outcome | wm_nback | WM_Task_2bk_Acc | false | true | 0 |
| outcome | reasoning_pmat | PMAT24_A_CR | true | true | 100 |
| secondary_outcome | reasoning_pmat | PMAT24_A_RTCR | true | true | 100 |
| outcome | reasoning_relational | Relational_Task_Acc | false | true | 0 |

## Domain Support

| Domain | Primary | Required | Analysis eligible | Status | Complete participants | Missing columns | Support passed |
|---|---:|---:|---:|---|---:|---|---:|
| attention_control | false | true | true | support_required | 100 | none | true |
| wm_list_sorting | true | true | true | support_required | 100 | none | true |
| wm_nback | false | false | false | optional_known_issue_pending | 0 | none | false |
| reasoning_pmat | true | true | true | support_required | 100 | none | true |
| reasoning_relational | false | false | false | optional | 0 | none | false |

## Boundary

- Participant-level HCP data in Git: false
- Outcome columns reused to construct same-domain K/C_candidate/V: false by config validation
- Model outcomes interpreted: false
- Stage 1-3 reports changed: false
