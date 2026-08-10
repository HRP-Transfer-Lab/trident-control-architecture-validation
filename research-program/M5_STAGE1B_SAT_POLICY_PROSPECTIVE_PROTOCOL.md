# M5 Stage 1B Speed-Accuracy Policy Candidate Prospective Protocol

**Status:** pre-outcome registered amendment

This hypothesis was formulated after reviewing the completed Stage 1, Stage 2
and Stage 3 real-data analyses. It is prospective with respect to
`T_policy_candidate` outcomes, but it is not part of the original Stage 1-3
pre-registration.

Previous Stage 1-3 configurations, reports, outputs and conclusions remain
unchanged. Stage 4 behavioural-criticality testing remains blocked.

No Trident-G, APC, PACE ontology, transfer, Predictive Calibration,
T_commit, optimality, neural-criticality or cusp claim is authorised.

## Question

```text
Is there a reproducible cross-task speed-accuracy POLICY dimension beyond
K, C_signal and V?
```

## Boundary

`T_policy_candidate` is a behavioural speed-accuracy policy candidate only.

It is not:

```text
a diffusion-model decision boundary
T_commit as a confirmed mechanism
Predictive Calibration
an optimality measure
a PACE state
a Trident-G state
```

High `T_policy_candidate` means relatively accuracy-favouring/cautious
behaviour. Low `T_policy_candidate` means relatively speed-favouring/rapid
behaviour. Neither pole is intrinsically better.

## Data-Support Preflight

The frozen paired-session source was inspected for schema availability,
missingness and sample support only. No associations relevant to the policy
hypothesis were inspected during this preflight.

Input:

```text
../../.tmp-zone-replication/flow-zone-zone-validation/data/processed/paired_vigilance_session_features.parquet
```

Expected and observed checksum:

```text
sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0
```

Support result:

```text
input rows: 768
participants: 466
sessions: 768

stroop_accuracy nonmissing: 767
stroop_mean_rt_ms nonmissing: 767
flanker_accuracy nonmissing: 768
flanker_mean_rt_ms nonmissing: 768

complete Stroop speed+accuracy rows: 767; participants: 466
complete Flanker speed+accuracy rows: 768; participants: 466
complete two-task policy rows: 767; participants: 466
```

The primary support gate passes for Stroop and Flanker general speed and
accuracy observables. SART remains the independent vigilance/readiness source.

## Variable Definitions

The existing Stage 1 definitions of `K`, `C_signal` and `V` are retained.
They are not redefined to improve this test.

`T_policy_candidate` is constructed only from Stroop and Flanker general
accuracy and mean RT:

```text
accuracy_q = z_train(task accuracy)
speed_q    = -z_train(task mean RT)
T_q        = (accuracy_q - speed_q) / sqrt(2)
T_common   = mean(T_stroop, T_flanker)
```

Interference-cost variables are excluded from `T_policy_candidate` because
they define `C_signal`. Throughput is excluded because it deliberately combines
speed and accuracy. SART vigilance variables are excluded from the primary
policy score because they define `V`.

Task signs and weights are not fitted from data.

## Leakage Guard

For cross-task transport, the target task's own general speed, accuracy and
throughput components are withheld from the K predictor for that target. This
preserves the Stage 1 feature orientations while preventing the current target
from entering the same-row predictors.

For next-session persistence, only current-session predictors are used.

All scaling is fitted inside training folds only. Validation is
participant-isolated.

## Primary Test A: Cross-Task Transport

Participant-isolated 5-fold validation.

Flanker target:

```text
BASE:   T_flanker ~ K_without_Flanker_general + C_signal + V
POLICY: T_flanker ~ K_without_Flanker_general + C_signal + V + T_stroop
```

Stroop target:

```text
BASE:   T_stroop ~ K_without_Stroop_general + C_signal + V
POLICY: T_stroop ~ K_without_Stroop_general + C_signal + V + T_flanker
```

The primary metric is participant-isolated held-out predictive log density.
Directional contrasts are reported separately, and the primary cross-task
transport contrast is the prospectively defined mean of both directions.

## Primary Test B: Next-Session Policy Persistence

Using the adjacent-session structure from Stage 1:

```text
BASE:
T_common_next ~ K_current + C_signal_current + V_current

POLICY:
T_common_next ~ K_current + C_signal_current + V_current
                + T_common_current
```

This asks whether policy has temporal persistence beyond present capacity,
control and vigilance. It does not assume policy must be highly trait-like.

## Negative Controls

Where support permits, the policy transport and persistence contrasts are
compared against speed-only and accuracy-only predictors. These determine
whether apparent policy transport is merely generic speed or accuracy
persistence.

`T_current` prediction of next-session `Y_behavior` may be reported as
secondary only. Success is not defined by whether high or low T improves
performance.

## Decision Rule

Status values:

```text
policy_dimension_supported
policy_dimension_not_supported
inconclusive
```

Support requires:

```text
residual T_common variance after K/C_signal/V >= 0.20
mean cross-task policy contrast > 0
cross-task 95% bootstrap CI lower bound >= -0.005
next-session policy contrast > 0
next-session 95% bootstrap CI lower bound >= -0.005
no forbidden columns
no current-target or future-session leakage
```

If cross-task and temporal evidence disagree, the decision is inconclusive.

Allowed wording if supported:

```text
a reproducible cross-task speed-accuracy policy candidate exists beyond
K/C_signal/V in this source.
```

`T_commit` is not confirmed by this analysis.

## Machine-Readable Config

```text
config/public_stage1b_sat_policy_v1.yaml
```
