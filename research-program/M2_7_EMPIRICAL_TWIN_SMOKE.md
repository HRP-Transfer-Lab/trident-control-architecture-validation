# M2.7 Empirical-Twin Smoke

**Status:** engineering/generator smoke only

No M2.7 architecture-recovery claim is made.

## Frozen Inputs

```text
EMPIRICAL_BACKGROUND_CONTRACT_V1:
1e9c11e030dc0706c8941dfc08489bb6f63cac5d

STATIC_MODEL_SELECTION_CONTRACT_V2:
81d22f2

Corrected generator commit:
efbdda4137d3839bfa23848de3c2496286ba4e3b
```

## Generator Configuration

```text
config: config/empirical_twin_v1.yaml
generator id: empirical_twin_v1
registry: m2_7_empirical_twin_world_registry_v1
master seed: 20260809
seed hierarchy: structural / background / missingness / trial count
source/task shift rule: template_mean - full_ACDC_feature_mean
session calibration: raw covariance scaled by m/(m-1) before participant centring
window calibration: diagonal AR innovation scaled to centred residual variance after fatigue
trial counts: truncated-normal approximation from mean, SD, p10, p50, p90
session_order_context_trend: off
paired cross-task session covariance: off
```

## Structural Worlds

| World | Known synthetic alignment | Structural source |
| --- | --- | --- |
| ETW0 | M0 probabilistic general performance | EW0 noise-free exact expectation |
| ETW1 | M1 continuous manifold | EW1 noise-free exact expectation |
| ETW2 | M2_EM_v1 | M2 self-check quadratic latent curve, noise-free expectation |
| ETW3 | M3 three-component mixture | EW3 noise-free exact expectation |
| ETW4 | M4 four-component mixture | EW4 noise-free exact expectation |

Historical M2.6 W0-W4 and EW0-EW4 definitions remain unchanged.

## Smoke Run

```text
command:
$env:PYTHONPATH='src'; python -m trident_validation.synthetic.empirical_twin_v1 --output-dir reports/generated/empirical_twin_v1_smoke_review_final

runtime: approximately 42 seconds
runs: 5
rows: 720
participants: 120
sessions: 240
windows: 720
participant-isolated split: true for all 5 worlds
train/test per world: 108 / 36 rows, 18 / 6 participants
models executed per world: 5
truth columns before fitting: stripped
truth columns after strip: 0
empirical participant IDs copied: 0
deterministic rerun: generated DataFrames equal
```

Tournament scoring was run only as an end-to-end engineering smoke. Model winners
or recovery rates were not inspected or interpreted.

## Support Audit

```text
estimated rows: 1065
support_eligible_covariance rows: 45
unsupported rows: 1560
implementation_approximation rows: 15
```

Unsupported rows were explicit, mainly:

```text
session_within_person_variance: 45
session_within_person_covariance: 1440
lag1_autocorrelation: 50
within_session_time_on_task_trend: 25
```

Mean/median RT borrowing is disabled. `median_rt_ms` repeated-session components
remain unsupported where the paired source has only `mean_rt_ms`.

## Component Checks

Large deterministic generator-only tests passed for:

```text
between-person variance recovery
participant-centred session covariance recovery
centred window AR + fatigue variance recovery
determinism
structural/background independence
forbidden feature-alias prevention
primary sensitivity switches off
```

Tiny-smoke positive-target median component ratios:

```text
between person: 0.901
session:        0.827
window:         0.657
```

The tiny smoke is not required to exactly recover targets.

## Lag-1 Diagnostics

Diagnostic-only lag summary:

```text
mean frozen target lag-1:                 0.340
mean existing per-sequence Pearson lag-1: -0.335
mean pooled window-residual lag-1:         0.152
proportion sequences length 3:             1.000
```

The three-window Pearson estimator remains unstable at smoke scale. The pooled
adjacent-pair statistic is diagnostic only and does not replace the frozen
empirical estimand.

## Gate Status

Engineering smoke passed for generation, manifesting, support/fallback audit,
participant-isolated split, truth stripping, deterministic rerun, and static V2
pipeline execution.

Residual smoke-scale diagnostics:

```text
lag-1 discrepancy: audit-estimator limitation plus finite-smoke noise
window ratio: finite-smoke/audit limitation; large deterministic test passed
```

Stop for scientific review before any 10-20/world M2.7 pilot.
