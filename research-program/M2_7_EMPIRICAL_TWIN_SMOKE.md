# M2.7 Empirical-Twin Smoke

**Status:** engineering/generator smoke only

No M2.7 architecture-recovery claim is made.

## Frozen Inputs

```text
EMPIRICAL_BACKGROUND_CONTRACT_V1:
1e9c11e030dc0706c8941dfc08489bb6f63cac5d

STATIC_MODEL_SELECTION_CONTRACT_V2:
81d22f2a942afd1652c169935f058b1634922d30

Final pre-pilot generator commit:
9699427176961a3064e9db8da23a3b287a441b1b
```

## Generator Configuration

```text
config: config/empirical_twin_v1.yaml
config hash: sha256:b40aaf8ba8d0a31e8fe92341f8f4e6486886f2f6baa9d783dcc606925f216a20
generator id: empirical_twin_v1
registry: m2_7_empirical_twin_world_registry_v1
master seed: 20260809
seed hierarchy: structural / background / missingness / trial count
source/task shift rule: template_mean - full_ACDC_feature_mean
session calibration: raw covariance scaled by m/(m-1) before participant centring
window calibration: full pairwise AR innovation covariance after fatigue, diagonal fallback if non-PSD
lag-1 calibration: deterministic grid calibration of internal phi to frozen Pearson estimand
trial counts: truncated-normal approximation from mean, SD, p10, p50, p90
session_order_context_trend: off
paired cross-task session covariance: off
windows per session: 16
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
$env:PYTHONPATH='src'; python -m trident_validation.synthetic.empirical_twin_v1 --output-dir reports/generated/empirical_twin_v1_final_temporal_covariance_smoke_v4

runtime: approximately 136 seconds
runs: 5
rows: 3840
participants: 120
participant-sessions: 240
windows per participant-session: 16
participant-isolated split: true for all 5 worlds
train/test per world: 576 / 192 rows, 18 / 6 participants
checkpoint/resume path flag: true for all 5 worlds
truth columns before fitting: 15 synthetic columns
truth columns before tournament fitting: stripped
deterministic rerun: generated DataFrames equal
```

Tournament scoring was run only as an end-to-end engineering smoke. Model winners
or recovery rates were not inspected or interpreted.

## Support Audit

```text
estimated rows: 1065
support_eligible_covariance rows: 60
internal_lag1_calibration calibrated rows: 75
unsupported rows: 1560
implementation_approximation rows: 15
```

Fallbacks were explicit:

```text
lag1_autocorrelation fixed_neutral_value: 50
session_within_person_covariance diagonal_variance_or_neutral: 1440
session_within_person_variance fixed_neutral_value: 45
within_session_time_on_task_trend fixed_neutral_value: 25
trial_count_distribution truncated_normal_from_summary_quantiles: 15
```

Mean/median RT borrowing remains disabled. `median_rt_ms` repeated-session
components remain unsupported where the paired source has only `mean_rt_ms`.

## Component Calibration

Large deterministic generator-only tests passed:

```text
between-person median variance ratio: 1.0059
session-within-person median variance ratio: 1.0064
window covariance status: support_eligible_covariance
window diagonal median ratio: 0.9990
window off-diagonal median absolute error: 0.00286
window off-diagonal max absolute error: 0.00762
```

Tiny-smoke component-isolated absolute-delta medians:

```text
between-person variance: 0.00963
session-within-person variance: 0.00000000146
window-within-session variance: 0.0000521
source/task shift: 0.0
missingness rate: 0.0
```

Tiny-smoke high-scale max deltas remain finite-smoke diagnostics, not acceptance
targets.

## Lag-1 Calibration

The generator records both the frozen empirical lag target and internal
calibrated phi. The frozen per-sequence Pearson estimator is not replaced by
the pooled adjacent-pair diagnostic.

Large deterministic calibration checks:

```text
target 0.35, T=12: internal phi 0.6138, realised frozen estimator 0.3475, delta 0.0025
target 0.62, T=16: internal phi 0.9108, realised frozen estimator 0.6162, delta 0.0038
```

Final smoke support-eligible Flanker lag targets and calibrated phi:

```text
accuracy:            target 0.5905, phi 0.8712, expected 0.5879
median_rt_ms:        target 0.6142, phi 0.9108, expected 0.6163
mean_response_speed: target 0.6510, phi 0.9504, expected 0.6488
rt_cv:               target 0.6001, phi 0.8910, expected 0.6017
throughput_proxy:    target 0.6058, phi 0.9009, expected 0.6088
```

Tiny-smoke realised frozen lag remains noisy with 16 sequences per
world/source/task cell:

```text
finite lag rows: 50
median absolute delta: 0.1729
p90 absolute delta: 0.4664
max absolute delta: 0.5843
```

This does not contradict the large deterministic calibration test.

## Repeated-Person Stability

Repeated-person stability is audited as an implied property of person plus
session components. It is not a third additive variance component.

The configured paired aggregate has no separate stability CSV, so the audit uses
the exact-feature paired variance-decomposition fraction when support is
estimated.

```text
finite target-realised rows: 30 / 75
median absolute delta: 0.0594
p90 absolute delta: 0.1218
max absolute delta: 0.3962
unsupported exact-feature rows: 45 / 75
```

## Gate Status

Engineering smoke passed for generation, manifesting, support/fallback audit,
participant-isolated split, truth stripping, deterministic rerun, calibrated
lag mapping, full support-eligible window covariance calibration, and static V2
pipeline execution.

Residual smoke-scale diagnostics:

```text
lag-1 target-realised smoke deltas are noisy at 16 sequences per cell;
between/window high-scale max deltas are finite-smoke diagnostics;
trial counts remain a documented quantile-summary approximation.
```

Stop for scientific review before any 10-20/world M2.7 pilot.
