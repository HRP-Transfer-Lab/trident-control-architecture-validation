# M6.1b HCP-YA Transversal Sensitivity Report

**Status:** exploratory post-M6.1 sensitivity complete

**N:** 100 official HCP unrelated subjects

This addendum was formulated after M6.1 outcome inspection. It is a robustness and demand-pattern diagnostic only. It does not confirm g, cognitive control, vigilance, binding, relational processing, WM updating, strategic policy variables, bottlenecks, Trident-G, APC, PACE, criticality or transfer.

## Summary

- Cohort mode: `hcp_100_unrelated`
- Split strategy: `participant_isolated_official_hcp_100_unrelated`
- Split seed: 20260822
- Bootstrap seed: 20260824
- Bootstrap iterations: 1000
- Participant-level outputs written: false
- Frozen M6.1 analysis changed: false

## K Robustness

| Domain | K variant | Mean delta log density | 95% bootstrap CI | Fraction positive |
|---|---|---:|---|---:|
| attention_control | speed_sequence_pair | 0.120896 | [0.022585, 0.218908] | 0.750000 |
| attention_control | no_picvocab | 0.101338 | [0.031755, 0.187187] | 0.770000 |
| attention_control | no_readeng | 0.082890 | [0.012444, 0.153864] | 0.710000 |
| attention_control | registered_full_k | 0.063864 | [0.007535, 0.122441] | 0.670000 |
| attention_control | no_picseq | 0.043995 | [-0.008178, 0.100365] | 0.640000 |
| attention_control | no_proc_speed | 0.011684 | [-0.022710, 0.048358] | 0.590000 |
| attention_control | language_crystallized_pair | 0.000126 | [-0.018657, 0.021518] | 0.500000 |
| reasoning_pmat | no_picvocab | 0.051203 | [-0.028687, 0.130528] | 0.630000 |
| reasoning_pmat | no_readeng | 0.042498 | [-0.051725, 0.127179] | 0.590000 |
| reasoning_pmat | registered_full_k | 0.037883 | [-0.104345, 0.149745] | 0.630000 |
| reasoning_pmat | speed_sequence_pair | 0.028461 | [-0.017695, 0.081094] | 0.620000 |
| reasoning_pmat | no_proc_speed | 0.027005 | [-0.137058, 0.149300] | 0.630000 |
| reasoning_pmat | no_picseq | 0.007330 | [-0.126350, 0.132340] | 0.650000 |
| reasoning_pmat | language_crystallized_pair | -0.007224 | [-0.167444, 0.118340] | 0.660000 |
| wm_list_sorting | no_proc_speed | 0.086577 | [0.003490, 0.169489] | 0.700000 |
| wm_list_sorting | registered_full_k | 0.076315 | [-0.025483, 0.172576] | 0.670000 |
| wm_list_sorting | no_readeng | 0.068756 | [-0.018087, 0.162082] | 0.680000 |
| wm_list_sorting | no_picvocab | 0.059950 | [-0.024735, 0.139535] | 0.640000 |
| wm_list_sorting | language_crystallized_pair | 0.055123 | [-0.024161, 0.131344] | 0.750000 |
| wm_list_sorting | no_picseq | 0.047818 | [-0.044664, 0.141287] | 0.680000 |
| wm_list_sorting | speed_sequence_pair | 0.032026 | [-0.026789, 0.096663] | 0.670000 |

## C/V Demand-Pattern Diagnostics

| Domain | Contrast | Mean delta log density | 95% bootstrap CI | Fraction positive |
|---|---|---:|---|---:|
| attention_control | C_increment_beyond_registered_K | 0.107575 | [0.009798, 0.201784] | 0.670000 |
| reasoning_pmat | C_increment_beyond_registered_K | -0.011916 | [-0.031852, 0.005425] | 0.480000 |
| wm_list_sorting | C_increment_beyond_registered_K | -0.003274 | [-0.009971, 0.002794] | 0.480000 |
| attention_control | V_increment_beyond_registered_K | -0.004990 | [-0.031319, 0.017312] | 0.610000 |
| reasoning_pmat | V_increment_beyond_registered_K | -0.010965 | [-0.058721, 0.027428] | 0.540000 |
| wm_list_sorting | V_increment_beyond_registered_K | -0.008425 | [-0.027756, 0.009044] | 0.530000 |
| attention_control | joint_CV_increment_beyond_registered_K | 0.109663 | [-0.007606, 0.216748] | 0.710000 |
| reasoning_pmat | joint_CV_increment_beyond_registered_K | -0.023985 | [-0.080162, 0.015745] | 0.490000 |
| wm_list_sorting | joint_CV_increment_beyond_registered_K | -0.011020 | [-0.031477, 0.009058] | 0.490000 |

## Model Descriptives

| Domain | K variant | Model | Mean held-out log density | RMSE | MAE | Held-out R2 | Held-out r |
|---|---|---|---:|---:|---:|---:|---:|
| wm_list_sorting | registered_full_k | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | registered_full_k | S1_K_variant | -3.830591 | 10.817086 | 8.457753 | 0.138802 | 0.375789 |
| wm_list_sorting | registered_full_k | S2_registered_K_plus_C | -3.833865 | 10.846228 | 8.476716 | 0.134156 | 0.370604 |
| wm_list_sorting | registered_full_k | S3_registered_K_plus_V | -3.839016 | 10.903141 | 8.478500 | 0.125045 | 0.361530 |
| wm_list_sorting | registered_full_k | S4_registered_K_plus_C_plus_V | -3.841611 | 10.927369 | 8.498331 | 0.121152 | 0.357526 |
| reasoning_pmat | registered_full_k | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | registered_full_k | S1_K_variant | -2.914294 | 4.390524 | 3.665103 | 0.080920 | 0.308557 |
| reasoning_pmat | registered_full_k | S2_registered_K_plus_C | -2.926210 | 4.431026 | 3.712765 | 0.063884 | 0.290212 |
| reasoning_pmat | registered_full_k | S3_registered_K_plus_V | -2.925259 | 4.424935 | 3.691402 | 0.066457 | 0.305618 |
| reasoning_pmat | registered_full_k | S4_registered_K_plus_C_plus_V | -2.938279 | 4.468168 | 3.732065 | 0.048125 | 0.287847 |
| attention_control | registered_full_k | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | registered_full_k | S1_K_variant | -3.678756 | 9.394136 | 7.491103 | 0.083290 | 0.291463 |
| attention_control | registered_full_k | S2_registered_K_plus_C | -3.571181 | 8.462624 | 6.727120 | 0.256077 | 0.508000 |
| attention_control | registered_full_k | S3_registered_K_plus_V | -3.683746 | 9.420447 | 7.516862 | 0.078148 | 0.287849 |
| attention_control | registered_full_k | S4_registered_K_plus_C_plus_V | -3.569093 | 8.375306 | 6.587478 | 0.271349 | 0.524697 |
| wm_list_sorting | no_proc_speed | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | no_proc_speed | S1_K_variant | -3.820329 | 10.773218 | 8.349071 | 0.145773 | 0.383568 |
| reasoning_pmat | no_proc_speed | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | no_proc_speed | S1_K_variant | -2.925172 | 4.418950 | 3.667735 | 0.068980 | 0.293889 |
| attention_control | no_proc_speed | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | no_proc_speed | S1_K_variant | -3.730936 | 9.849303 | 7.729922 | -0.007695 | 0.062962 |
| wm_list_sorting | no_picseq | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | no_picseq | S1_K_variant | -3.859089 | 11.118271 | 8.620735 | 0.090177 | 0.309128 |
| reasoning_pmat | no_picseq | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | no_picseq | S1_K_variant | -2.944847 | 4.513112 | 3.753806 | 0.028880 | 0.242972 |
| attention_control | no_picseq | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | no_picseq | S1_K_variant | -3.698625 | 9.535501 | 7.648196 | 0.055493 | 0.241814 |
| wm_list_sorting | no_readeng | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | no_readeng | S1_K_variant | -3.838150 | 10.941748 | 8.656667 | 0.118838 | 0.349235 |
| reasoning_pmat | no_readeng | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | no_readeng | S1_K_variant | -2.909679 | 4.407207 | 3.722540 | 0.073922 | 0.287271 |
| attention_control | no_readeng | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | no_readeng | S1_K_variant | -3.659730 | 9.253879 | 7.347320 | 0.110459 | 0.336825 |
| wm_list_sorting | no_picvocab | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | no_picvocab | S1_K_variant | -3.846957 | 11.047024 | 8.858042 | 0.101800 | 0.324492 |
| reasoning_pmat | no_picvocab | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | no_picvocab | S1_K_variant | -2.900974 | 4.375625 | 3.678897 | 0.087146 | 0.305222 |
| attention_control | no_picvocab | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | no_picvocab | S1_K_variant | -3.641282 | 9.059300 | 7.204428 | 0.147474 | 0.385521 |
| wm_list_sorting | language_crystallized_pair | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | language_crystallized_pair | S1_K_variant | -3.851783 | 11.128772 | 8.558739 | 0.088458 | 0.303027 |
| reasoning_pmat | language_crystallized_pair | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | language_crystallized_pair | S1_K_variant | -2.959401 | 4.564250 | 3.778021 | 0.006748 | 0.210097 |
| attention_control | language_crystallized_pair | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | language_crystallized_pair | S1_K_variant | -3.742494 | 9.934289 | 7.733794 | -0.025160 | -0.085952 |
| wm_list_sorting | speed_sequence_pair | S0_intercept | -3.906906 | 11.881880 | 9.483500 | -0.039089 | -0.263612 |
| wm_list_sorting | speed_sequence_pair | S1_K_variant | -3.874880 | 11.461770 | 9.215105 | 0.033091 | 0.203514 |
| reasoning_pmat | speed_sequence_pair | S0_intercept | -2.952177 | 4.616086 | 3.920000 | -0.015941 | -0.168345 |
| reasoning_pmat | speed_sequence_pair | S1_K_variant | -2.923717 | 4.483752 | 3.826473 | 0.041474 | 0.211285 |
| attention_control | speed_sequence_pair | S0_intercept | -3.742620 | 9.937746 | 7.697670 | -0.025874 | -0.214471 |
| attention_control | speed_sequence_pair | S1_K_variant | -3.621724 | 8.897729 | 6.923880 | 0.177613 | 0.424745 |

## Boundary

- `WM_Task_2bk_Acc` used: false
- `Relational_Task_Acc` used: false
- `PMAT24_A_RTCR` used: false
- Binding/relational/WM-updating operators tested: false
- Strategic A/T/PC tested: false
- Single-task residual converted to capacity: false
