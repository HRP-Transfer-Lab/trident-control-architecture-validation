# M6 HCP-YA Transversal K/C/V Preflight

**Status:** data-support preflight only

**Model fitting allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.

## Question

Do lower-level K/C/V coordinates transport across attention/control, working memory and reasoning?

## Summary

- Status: `data_support_incomplete`
- Support passed: false
- Input path: `data\processed\hcp_ya_transversal_extract.csv`
- Input checksum: `sha256:fa4b69736a1d5fa62f1a51ff6518f29a160921a16319b0359fe222478fd359c9`

## Split Safeguard

- Family-isolated CV feasible: false
- Unrelated-only feasible: false
- Ordinary participant folds allowed: false

## Column Support

| Role | Variable/domain | Column | Available | Nonmissing |
|---|---|---|---:|---:|
| identity | participant_id | Subject | true | 0 |
| identity | family_id | Family_ID | true | 0 |
| predictor | K | NIH_Flanker_Unadj | true | 0 |
| predictor | K | NIH_ProcSpeed_Unadj | true | 0 |
| predictor | K | NIH_CardSort_Unadj | true | 0 |
| predictor | K | PicSeq_Unadj | true | 0 |
| predictor | K | ReadEng_Unadj | true | 0 |
| predictor | K | PicVocab_Unadj | true | 0 |
| predictor | C_signal | NIH_Flanker_Unadj | true | 0 |
| predictor | V | SCPT_SEN | true | 0 |
| predictor | V | SCPT_SPEC | true | 0 |
| outcome | attention_control | NIH_CardSort_Unadj | true | 0 |
| outcome | wm_list_sorting | ListSort_Unadj | true | 0 |
| outcome | wm_nback | tfMRI_WM_2bk_Acc | true | 0 |
| outcome | reasoning_pmat | PMAT24_A_CR | true | 0 |
| secondary_outcome | reasoning_pmat | PMAT24_A_RTCR | true | 0 |

## Domain Support

| Domain | Primary | Complete participants | Missing columns | Support passed |
|---|---:|---:|---|---:|
| attention_control | false | 0 | none | false |
| wm_list_sorting | true | 0 | none | false |
| wm_nback | true | 0 | none | false |
| reasoning_pmat | true | 0 | none | false |

## Boundary

- Participant-level HCP data in Git: false
- Outcome columns reused to construct same-domain K/C/V: false by config validation
- Model outcomes interpreted: false
- Stage 1-3 reports changed: false
