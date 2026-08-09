# M5 Stage 1 K / C_signal / Vigilance Real-Data Analysis

**Status:** exploratory Stage 1 mechanism analysis

**Formal claims allowed:** false

No Trident-G/APC/PACE validation claim, transfer claim, neural-criticality claim or cusp claim is made.

## Question

Can control and readiness be separated?

## Design

The analysis uses the frozen paired public session-level source only. Adjacent sessions are paired within participant, and current-session K/C_signal/V composites predict next-session K/C_signal/V targets under participant-isolated 5-fold validation.

Forbidden profile/probability/candidate columns are present in the source but are not used.

## Sample

- Input participant-session rows: 768
- Complete adjacent session pairs: 282
- Participants with adjacent pairs: 199
- Input checksum: `sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0`

## Pre-Registered Decision

Decision: **inconclusive**

evidence did not satisfy either the support or disconfirmation rule

## Residual Separation

- C_signal residual variance fraction after K: 0.7371
- V residual variance fraction after K: 0.9470
- Absolute residual C_signal/V correlation after K: 0.0894

## Primary Predictive Contrasts

| Target | Contrast | Mean delta log density | 95% CI | Positive-pair rate |
|---|---:|---:|---:|---:|
| C_signal_next | K_plus_C_signal_minus_K_only | 0.01807 | [-0.01332, 0.04591] | 0.699 |
| C_signal_next | K_plus_V_minus_K_only | -0.00082 | [-0.01162, 0.01119] | 0.514 |
| C_signal_next | K_plus_C_signal_plus_V_minus_K_only | 0.01524 | [-0.02034, 0.04289] | 0.670 |
| V_next | K_plus_C_signal_minus_K_only | 0.00042 | [-0.00485, 0.00692] | 0.411 |
| V_next | K_plus_V_minus_K_only | 0.10156 | [0.02830, 0.18087] | 0.798 |
| V_next | K_plus_C_signal_plus_V_minus_K_only | 0.10699 | [0.02948, 0.18344] | 0.787 |
| K_next | K_plus_C_signal_minus_K_only | -0.00659 | [-0.04273, 0.02126] | 0.560 |
| K_next | K_plus_V_minus_K_only | -0.00935 | [-0.03800, 0.01466] | 0.589 |
| K_next | K_plus_C_signal_plus_V_minus_K_only | -0.01872 | [-0.08911, 0.03161] | 0.613 |

## Mean Held-Out Log Density

| Target | Model | Mean held-out log density |
|---|---:|---:|
| C_signal_next | K_only | -0.70612 |
| C_signal_next | K_plus_C_signal | -0.68592 |
| C_signal_next | K_plus_C_signal_by_V | -0.69452 |
| C_signal_next | K_plus_C_signal_plus_V | -0.68910 |
| C_signal_next | K_plus_V | -0.70727 |
| K_next | K_only | -0.97175 |
| K_next | K_plus_C_signal | -0.97728 |
| K_next | K_plus_C_signal_by_V | -1.00769 |
| K_next | K_plus_C_signal_plus_V | -0.98895 |
| K_next | K_plus_V | -0.98107 |
| V_next | K_only | -0.94811 |
| V_next | K_plus_C_signal | -0.94776 |
| V_next | K_plus_C_signal_by_V | -0.83080 |
| V_next | K_plus_C_signal_plus_V | -0.84466 |
| V_next | K_plus_V | -0.85014 |

## Boundary

- PACE/profile/candidate columns used: false
- Dynamic-regime variables used: false
- Transfer outcomes used: false
- Criticality/cusp interpretation: false

This Stage 1 result may inform whether the programme should proceed to the descriptive PACE-expression increment, but it does not validate Trident-G or any latent ontology.
