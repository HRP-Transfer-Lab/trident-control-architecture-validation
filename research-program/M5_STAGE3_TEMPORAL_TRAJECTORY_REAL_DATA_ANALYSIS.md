# M5 Stage 3 Temporal-Trajectory Real-Data Analysis

**Status:** exploratory Stage 3 temporal increment analysis

**Formal claims allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, adaptive/locked/scattered regime claim, neural-criticality claim or cusp claim is made.

## Question

Does prior within-person trajectory add predictive structure beyond static K/C/V?

## Design

The analysis uses complete online -> lab1 -> lab2 sequences from the frozen paired public session-level source. Lab1 static K/C_signal/V predicts lab2 behaviour. The temporal candidate adds online-to-lab1 within-person deltas.

No regime labels are estimated or interpreted. This is a trajectory-increment test only.

The upstream profile/probability/candidate columns are present in the source but are not used.

## Sample

- Input participant-session rows: 768
- Complete three-session sequences: 83
- Participants with complete sequences: 83
- Input checksum: `sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0`

## Pre-Registered Decision

Decision: **inconclusive**

temporal increment did not satisfy either the support or disconfirmation rule

## Primary Predictive Contrasts

| Target | Contrast | Mean delta log density | 95% CI | Positive-sequence rate |
|---|---:|---:|---:|---:|
| Y_behavior_next | continuous_autoregressive_delta_minus_static_K_C_V | 0.00631 | [-0.04249, 0.05773] | 0.446 |
| Y_behavior_next | delta_only_negative_control_minus_static_K_C_V | -0.01027 | [-0.29005, 0.32988] | 0.253 |
| K_next | continuous_autoregressive_delta_minus_static_K_C_V | 0.04200 | [-0.03788, 0.14588] | 0.590 |
| K_next | delta_only_negative_control_minus_static_K_C_V | -0.13216 | [-0.38461, 0.14110] | 0.241 |
| C_signal_next | continuous_autoregressive_delta_minus_static_K_C_V | -0.07205 | [-0.17936, -0.00324] | 0.422 |
| C_signal_next | delta_only_negative_control_minus_static_K_C_V | -0.04839 | [-0.13681, 0.02471] | 0.325 |
| V_next | continuous_autoregressive_delta_minus_static_K_C_V | -0.01189 | [-0.08046, 0.05532] | 0.518 |
| V_next | delta_only_negative_control_minus_static_K_C_V | -0.02371 | [-0.25330, 0.20991] | 0.253 |

## Delta Diagnostics

| Component | Train mean | Train SD | Held-out mean | Held-out SD |
|---|---:|---:|---:|---:|
| delta_C_signal | 0.3474 | 0.5520 | 0.3465 | 0.5355 |
| delta_K | 0.0725 | 0.5872 | 0.0772 | 0.5508 |
| delta_V | 0.2545 | 0.6899 | 0.2628 | 0.7017 |

## Mean Held-Out Log Density

| Target | Model | Mean held-out log density |
|---|---:|---:|
| C_signal_next | continuous_autoregressive_delta | -0.75148 |
| C_signal_next | delta_only_negative_control | -0.73730 |
| C_signal_next | static_K_C_V | -0.69924 |
| K_next | continuous_autoregressive_delta | -0.62725 |
| K_next | delta_only_negative_control | -0.84597 |
| K_next | static_K_C_V | -0.66508 |
| V_next | continuous_autoregressive_delta | -0.95248 |
| V_next | delta_only_negative_control | -0.99738 |
| V_next | static_K_C_V | -0.93347 |
| Y_behavior_next | continuous_autoregressive_delta | -0.45409 |
| Y_behavior_next | delta_only_negative_control | -0.53912 |
| Y_behavior_next | static_K_C_V | -0.45439 |

## Boundary

- Regime labels used: false
- Adaptive/locked/scattered regime claim: false
- Upstream PACE/profile/probability/candidate columns used: false
- Transfer outcomes used: false
- Criticality/cusp interpretation: false

This Stage 3 result only tests whether a simple prior within-person trajectory increment is useful beyond static K/C/V in the paired public source.
