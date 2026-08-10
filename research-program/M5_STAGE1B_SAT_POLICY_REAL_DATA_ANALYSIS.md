# M5 Stage 1B Speed-Accuracy Policy Candidate Real-Data Analysis

**Status:** exploratory Stage 1B policy-candidate analysis

**Formal claims allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, T_commit claim, Predictive Calibration claim, optimality claim, neural-criticality claim or cusp claim is made.

## Question

Is there a reproducible cross-task speed-accuracy policy dimension beyond K, C_signal and V?

## Timing

This hypothesis was formulated after Stage 1-3 review and was pre-registered before inspecting T_policy_candidate outcomes.

## Data-Support Preflight

- Input rows: 768
- Input participants: 466
- Input checksum: `sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0`
- Support gate passed: true

| Task | Accuracy column | Mean RT column | Complete rows | Complete participants |
|---|---|---|---:|---:|
| Stroop | stroop_accuracy | stroop_mean_rt_ms | 767 | 466 |
| Flanker | flanker_accuracy | flanker_mean_rt_ms | 768 | 466 |
| Stroop_Flanker_common | both | both | 767 | 466 |

## Pre-Registered Decision

Decision: **inconclusive**

cross-task and temporal evidence did not jointly satisfy support or disconfirmation rules

## Residual Policy Variance

| Diagnostic | Value |
|---|---:|
| residual_variance_fraction | 1.06484 |
| correlation_T_common_with_K | -0.14913 |
| correlation_T_common_with_C_signal | -0.10290 |
| correlation_T_common_with_V | 0.05529 |

## Cross-Task Transport

| Target | Contrast | Participant-mean delta log density | 95% CI | Positive-participant rate |
|---|---|---:|---:|---:|
| T_flanker | policy_minus_base | 0.14640 | [-0.02776, 0.41229] | 0.781 |
| T_flanker | speed_negative_control_minus_base | 0.13734 | [-0.02386, 0.38272] | 0.790 |
| T_flanker | accuracy_negative_control_minus_base | 0.14476 | [-0.03443, 0.41048] | 0.762 |
| T_stroop | policy_minus_base | 0.12601 | [-0.06689, 0.37061] | 0.785 |
| T_stroop | speed_negative_control_minus_base | 0.10657 | [-0.07584, 0.32128] | 0.792 |
| T_stroop | accuracy_negative_control_minus_base | 0.13847 | [-0.05147, 0.39711] | 0.783 |
| mean_cross_task_transport | policy_minus_base | 0.13620 | [-0.04909, 0.39244] | 0.788 |

## Next-Session Persistence

| Target | Contrast | Participant-mean delta log density | 95% CI | Positive-participant rate |
|---|---|---:|---:|---:|
| T_common_next | policy_persistence_minus_base | -0.21489 | [-0.91715, 0.16496] | 0.799 |
| T_common_next | speed_persistence_negative_control_minus_base | -0.20005 | [-0.87358, 0.16367] | 0.799 |
| T_common_next | accuracy_persistence_negative_control_minus_base | -0.21560 | [-0.91107, 0.16002] | 0.799 |

## Negative Controls

| Test family | Target | Contrast | Participant-mean delta log density | 95% CI |
|---|---|---|---:|---:|
| cross_task_transport | T_flanker | speed_negative_control_minus_base | 0.13734 | [-0.02386, 0.38272] |
| cross_task_transport | T_flanker | accuracy_negative_control_minus_base | 0.14476 | [-0.03443, 0.41048] |
| cross_task_transport | T_stroop | speed_negative_control_minus_base | 0.10657 | [-0.07584, 0.32128] |
| cross_task_transport | T_stroop | accuracy_negative_control_minus_base | 0.13847 | [-0.05147, 0.39711] |
| next_session_persistence | T_common_next | speed_persistence_negative_control_minus_base | -0.20005 | [-0.87358, 0.16367] |
| next_session_persistence | T_common_next | accuracy_persistence_negative_control_minus_base | -0.21560 | [-0.91107, 0.16002] |

## Boundary

- T_commit claim: false
- Predictive Calibration claim: false
- Optimality claim: false
- PACE/profile/probability/candidate columns used: false
- Dynamic-regime variables used: false
- Transfer outcomes used: false
- Criticality/cusp interpretation: false

This result only evaluates a behavioural cross-task speed-accuracy policy candidate in the paired public source.
