# M2.7 Empirical-Twin Pilot Protocol

**Status:** exploratory / informative pilot

No confirmatory M2.7 recovery claim is authorised. No Trident-G, APC, PACE,
neural criticality or cusp-validation claim is authorised.

## Frozen Inputs

```text
EMPIRICAL_BACKGROUND_CONTRACT_V1:
1e9c11e030dc0706c8941dfc08489bb6f63cac5d

STATIC_MODEL_SELECTION_CONTRACT_V2:
81d22f2a942afd1652c169935f058b1634922d30

Final pre-pilot generator:
9699427176961a3064e9db8da23a3b287a441b1b

Audit correction:
617549288c4ef44ae216473fcc822f4ad6856d3a
```

Dedicated repeat-only stability aggregate:

```text
paired_session_repeated_person_stability.csv
sha256:6d19f43aa2bef343a0e9e766f548be0cf1b64f66ece097f85267266ecd1e91bc
processed paired source checksum:
sha256:dd30838e96fbc1a80c42d0cd41161644b1563bccc3901b39ab9e9a1ded0065a0
```

The mixed all-participant `between_participant_fraction` from
`paired_session_variance_decomposition.csv` is not a primary
repeated-person-stability target.

## Matched-Support Task Decision

The Full ACDC background source contains eligible Stroop, Flanker and Simon
templates, while the independent repeated-session source contains Stroop,
Flanker and SART. The primary M2.7 pilot therefore uses the intersection,
Stroop and Flanker, so each task family has both window-level ACDC background
support and independently estimated repeated-session support. No cross-task
borrowing is permitted. Simon is reserved for a separately registered
support-asymmetry sensitivity if later scientifically warranted.

```text
Full ACDC contract-eligible templates:
Stroop:  34
Flanker: 12
Simon:    5
SART:     0

paired Simon repeated-session support: unavailable
primary pilot tasks: Stroop, Flanker
```

This is a prospective pilot-design decision, not a modification to
`EMPIRICAL_BACKGROUND_CONTRACT_V1`.

## Pilot Design

```text
worlds: ETW0, ETW1, ETW2, ETW3, ETW4
replicates/world: 10
total empirical twins: 50
templates/twin: 2
task composition/replicate: 1 Stroop + 1 Flanker
participants/template: 24
participants/twin: 48
sessions/participant: 2
participant-sessions/twin: 96
windows/session: 16
expected windows/twin: 1536
participant-isolated holdout fraction: 0.25
expected held-out participants/twin: approximately 12
process workers: 2
BLAS/OpenMP threads/worker: 1
session_order_context_trend: off
paired cross-task covariance: off
support-eligible covariance: on
formal_claims_allowed: false
```

The immutable pilot schedule is generated before the first model fit. It uses
pilot schedule seed `20260811`, which is distinct from the smoke seed. The first
ten pilot replicates select ten distinct Stroop templates and ten distinct
Flanker templates without replacement from the frozen eligible-template table.
For replicate `r`, the same template pair is used across ETW0-ETW4.

Schedule hash:
`sha256:c712eff7a52ad2601760d9fb15b253c46a299c803b73ed093c9e9372f271a81a`.

## Frozen Pilot Questions

Q1. Does ETW1/M1 remain distinguishable under frozen empirical background
realism?

Q2. Does ETW2/M2_EM_v1 remain distinguishable from M1?

Q3. Under ETW0, does empirical background create apparent multidimensionality?

Q4. Under continuous ETW0-ETW2 truth, does empirical background inflate false
selection of M3/M4?

Q5. Under strong discrete ETW3/ETW4 truth, do the discrete candidates remain
distinguishable?

Q6. Which world/model comparisons become practically ambiguous?

Q7. What are model-fit failure rates and M2_EM convergence/runtime behaviour?

Q8. What is the practical runtime per empirical twin?

No new hypotheses may be introduced after pilot outcomes are inspected.

## Outputs

```text
pilot_schedule.csv/json
pilot_manifest.json
unit checkpoints
model_scores.csv
participant_scores.csv
selection_summary.csv
paired_contrasts.csv
fit_diagnostics.csv
runtime_summary.csv
background_realism_summary.csv
M2_7_EMPIRICAL_TWIN_PILOT_REPORT.md
```

The pilot stops after the 10/world report. It does not choose confirmatory N,
retune empirical background, alter ETW truth, alter model definitions, add
APC/PACE/dynamic models or move beyond M2.7.
