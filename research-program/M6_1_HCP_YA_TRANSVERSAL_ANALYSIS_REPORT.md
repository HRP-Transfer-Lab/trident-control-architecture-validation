# M6.1 HCP-YA Transversal K/C Candidate/V Analysis Report

**Status:** exploratory external-data analysis complete

**N:** 100 official HCP unrelated subjects

This is an exploratory HCP unrelated-subject test with N=100. It does not confirm g, cognitive control as a latent mechanism, vigilance as a latent mechanism, Trident-G, APC, PACE, W-specific capacity, bottlenecks, criticality or transfer.

## Summary

- Cohort mode: `hcp_100_unrelated`
- Split strategy: `participant_isolated_official_hcp_100_unrelated`
- Folds: 5
- Split seed: 20260822
- Bootstrap seed: 20260823
- Bootstrap iterations: 1000
- Participant-level outputs written: false

## Model Summary

| Domain | Model | Mean held-out log density | RMSE | MAE | Held-out R2 | Held-out r |
|---|---|---:|---:|---:|---:|---:|
| wm_list_sorting | M0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | M1_K | -3.830591 | 10.817086 | 8.457753 | 0.138802 | 0.375789 |
| wm_list_sorting | M2_K_plus_C | -3.833865 | 10.846228 | 8.476716 | 0.134156 | 0.370604 |
| wm_list_sorting | M3_K_plus_V | -3.839016 | 10.903141 | 8.478500 | 0.125045 | 0.361530 |
| wm_list_sorting | M4_K_plus_C_plus_V | -3.841611 | 10.927369 | 8.498331 | 0.121152 | 0.357526 |
| wm_list_sorting | M5_K_plus_C_plus_V_plus_C_by_V | -3.851215 | 11.015367 | 8.605126 | 0.106941 | 0.343189 |
| reasoning_pmat | M0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | M1_K | -2.914294 | 4.390524 | 3.665103 | 0.080920 | 0.308557 |
| reasoning_pmat | M2_K_plus_C | -2.926210 | 4.431026 | 3.712765 | 0.063884 | 0.290212 |
| reasoning_pmat | M3_K_plus_V | -2.925259 | 4.424935 | 3.691402 | 0.066457 | 0.305618 |
| reasoning_pmat | M4_K_plus_C_plus_V | -2.938279 | 4.468168 | 3.732065 | 0.048125 | 0.287847 |
| reasoning_pmat | M5_K_plus_C_plus_V_plus_C_by_V | -2.965675 | 4.554182 | 3.791988 | 0.011125 | 0.249794 |
| attention_control | M0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | M1_K | -3.678756 | 9.394136 | 7.491103 | 0.083290 | 0.291463 |
| attention_control | M2_K_plus_C | -3.571181 | 8.462624 | 6.727120 | 0.256077 | 0.508000 |
| attention_control | M3_K_plus_V | -3.683746 | 9.420447 | 7.516862 | 0.078148 | 0.287849 |
| attention_control | M4_K_plus_C_plus_V | -3.569093 | 8.375306 | 6.587478 | 0.271349 | 0.524697 |
| attention_control | M5_K_plus_C_plus_V_plus_C_by_V | -3.614871 | 8.624474 | 6.741408 | 0.227349 | 0.495215 |

## Registered Contrasts

| Domain | Contrast | Mean delta | 95% bootstrap CI | Fraction positive | Status |
|---|---|---:|---|---:|---|
| wm_list_sorting | K_increment | 0.076315 | [-0.018451, 0.176211] | 0.670000 | primary |
| wm_list_sorting | C_increment_beyond_K | -0.003274 | [-0.009557, 0.003201] | 0.480000 | primary |
| wm_list_sorting | V_increment_beyond_K | -0.008425 | [-0.027226, 0.010637] | 0.530000 | primary |
| wm_list_sorting | joint_CV_increment_beyond_K | -0.011020 | [-0.031647, 0.008813] | 0.490000 | primary |
| wm_list_sorting | interaction_increment | -0.009604 | [-0.019678, -0.000257] | 0.560000 | secondary_only |
| reasoning_pmat | K_increment | 0.037883 | [-0.089438, 0.149784] | 0.630000 | primary |
| reasoning_pmat | C_increment_beyond_K | -0.011916 | [-0.031178, 0.005694] | 0.480000 | primary |
| reasoning_pmat | V_increment_beyond_K | -0.010965 | [-0.064173, 0.028288] | 0.540000 | primary |
| reasoning_pmat | joint_CV_increment_beyond_K | -0.023985 | [-0.079253, 0.015599] | 0.490000 | primary |
| reasoning_pmat | interaction_increment | -0.027396 | [-0.060945, -0.001659] | 0.520000 | secondary_only |
| attention_control | K_increment | 0.063864 | [0.006459, 0.126840] | 0.670000 | primary |
| attention_control | C_increment_beyond_K | 0.107575 | [0.011809, 0.205351] | 0.670000 | primary |
| attention_control | V_increment_beyond_K | -0.004990 | [-0.028673, 0.017129] | 0.610000 | primary |
| attention_control | joint_CV_increment_beyond_K | 0.109663 | [-0.010062, 0.223714] | 0.710000 | primary |
| attention_control | interaction_increment | -0.045778 | [-0.107678, 0.000334] | 0.500000 | secondary_only |

## Domain-Weight Comparison

| Model/result | Mean held-out log density or delta | 95% bootstrap CI | Status |
|---|---:|---|---|
| common_slope | -1.370895 | [NA, NA] | nan |
| domain_specific_slope | -1.363669 | [NA, NA] | nan |
| domain_specific_minus_common | 0.007226 | [-0.028946, 0.040257] | secondary_low_precision |

## Descriptive Residual Diagnostics

These residual diagnostics are descriptive unexplained outcome variation only. They are not W, WM capacity, a layer-specific factor or a bottleneck variable.

| Domain | Model | Mean held-out log density | Diagnostic label |
|---|---|---:|---|
| wm_list_sorting | M0_intercept | -3.906906 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |
| wm_list_sorting | M4_K_plus_C_plus_V | -3.841611 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |
| reasoning_pmat | M0_intercept | -2.952177 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |
| reasoning_pmat | M4_K_plus_C_plus_V | -2.938279 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |
| attention_control | M0_intercept | -3.742620 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |
| attention_control | M4_K_plus_C_plus_V | -3.569093 | descriptive_heldout_unexplained_outcome_variation_not_latent_capacity |

## Diagnostic Fields

- PMAT24_A_RTCR available: true
- PMAT24_A_RTCR non-missing: 100
- PMAT RT used in reasoning composite: false
- WM_Task_2bk_Acc used: false
- Relational_Task_Acc used: false

## Gates

- M6.2 status: `blocked_pending_independent_second_WM_indicator`
- M6.3 status: `blocked_until_m6_2_establishes_defensible_independent_layer_specific_WM_candidate`
