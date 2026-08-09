# M5 Stage 2 Descriptive PACE-Expression Real-Data Analysis

**Status:** exploratory Stage 2 mechanism analysis

**Formal claims allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, neural-criticality claim or cusp claim is made.

## Question

Do regulated / brittle / compensatory / overloaded expressions add predictive structure beyond continuous K/C/V?

## Design

The analysis uses the frozen paired public session-level source. Adjacent current sessions predict next-session behaviour under participant-isolated 5-fold validation.

Continuous K/C_signal/V composites are fold-scaled from observed task features. Descriptive PACE-expression indicators are derived inside each training fold from training-fold medians and then applied to held-out participants. They are threshold descriptors, not latent classes.

The upstream profile/probability/candidate columns are present in the source but are not used.

## Sample

- Input participant-session rows: 768
- Complete adjacent session pairs: 282
- Participants with adjacent pairs: 199
- Input checksum: `sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0`

## Pre-Registered Decision

Decision: **pace_expression_increment_not_supported**

descriptive PACE increment failed the pre-registered held-out predictive criterion

## Primary Predictive Contrasts

| Target | Contrast | Mean delta log density | 95% CI | Positive-pair rate |
|---|---:|---:|---:|---:|
| Y_behavior_next | continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V | -0.03071 | [-0.05527, -0.00911] | 0.440 |
| Y_behavior_next | PACE_expression_only_negative_control_minus_continuous_K_C_V | -0.09537 | [-0.17313, -0.02458] | 0.301 |
| K_next | continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V | -0.03132 | [-0.05502, -0.00899] | 0.397 |
| K_next | PACE_expression_only_negative_control_minus_continuous_K_C_V | -0.02475 | [-0.17630, 0.22153] | 0.252 |
| C_signal_next | continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V | -0.04374 | [-0.06961, -0.02379] | 0.429 |
| C_signal_next | PACE_expression_only_negative_control_minus_continuous_K_C_V | -0.03563 | [-0.08913, 0.00871] | 0.394 |
| V_next | continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V | -0.00310 | [-0.02326, 0.01421] | 0.479 |
| V_next | PACE_expression_only_negative_control_minus_continuous_K_C_V | -0.08599 | [-0.16445, -0.01108] | 0.291 |

## Descriptive Expression Rates

| Expression | Train rate | Held-out rate |
|---|---:|---:|
| brittle | 0.111 | 0.122 |
| compensatory | 0.293 | 0.300 |
| none_of_four | 0.161 | 0.164 |
| overloaded | 0.206 | 0.200 |
| regulated | 0.229 | 0.215 |

## Mean Held-Out Log Density

| Target | Model | Mean held-out log density |
|---|---:|---:|
| C_signal_next | PACE_expression_only_negative_control | -0.72905 |
| C_signal_next | continuous_K_C_V | -0.69169 |
| C_signal_next | continuous_K_C_V_plus_descriptive_PACE_expression | -0.73540 |
| K_next | PACE_expression_only_negative_control | -1.15563 |
| K_next | continuous_K_C_V | -1.12622 |
| K_next | continuous_K_C_V_plus_descriptive_PACE_expression | -1.15668 |
| V_next | PACE_expression_only_negative_control | -0.93059 |
| V_next | continuous_K_C_V | -0.84021 |
| V_next | continuous_K_C_V_plus_descriptive_PACE_expression | -0.84333 |
| Y_behavior_next | PACE_expression_only_negative_control | -0.54687 |
| Y_behavior_next | continuous_K_C_V | -0.44734 |
| Y_behavior_next | continuous_K_C_V_plus_descriptive_PACE_expression | -0.47827 |

## Boundary

- Upstream PACE/profile/probability/candidate columns used: false
- PACE expression ontology claim: false
- Forced four-profile claim: false
- Dynamic-regime variables used: false
- Transfer outcomes used: false
- Criticality/cusp interpretation: false

This Stage 2 result only tests a descriptive nonlinear increment over continuous K/C/V in the paired public source.
