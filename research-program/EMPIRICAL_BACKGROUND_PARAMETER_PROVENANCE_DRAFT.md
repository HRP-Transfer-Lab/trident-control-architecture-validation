# Empirical Background Parameter Provenance Draft

**Status:** draft for scientific review; not frozen  
**Stage:** M2.7 empirical-twin preflight  
**Purpose:** record which development source may inform each empirical-background quantity before `EMPIRICAL_BACKGROUND_CONTRACT_V1` is frozen.

## Boundary

These parameters describe empirical background realism for later known-truth synthetic worlds. They are not latent architecture labels and must not be used to decide which of M0/M1/M2_EM/M3/M4 should win.

No APC, PACE, Trident-state or brain-critical-zone inference is made from this table.

## Source Roles

| Parameter family | Primary source | Estimation level | Support rule | Current status | Used by empirical twins |
| --- | --- | --- | --- | --- | --- |
| `between_person_variance.<feature>` | Full ACDC | source x task | at least 2 participants with observed feature | supported across 57 templates | candidate yes |
| `between_person_covariance.<feature_pair>` | Full ACDC | source x task | at least 2 complete participant means | supported where feature pair observed | candidate yes |
| `source_task_shift.<feature>` | Full ACDC | source x task | observed template means | supported across 57 templates | candidate yes |
| `window_within_session_variance.<feature>` | Full ACDC | source x task | at least 1 session with repeated windows and observed feature | mostly supported | candidate yes |
| `within_session_covariance.<feature_pair>` | Full ACDC | source x task | at least 2 complete window residual units | supported where feature pair observed | candidate yes |
| `lag1_autocorrelation.<feature>` | Full ACDC | source x task x participant x session | within-task sequences with at least 3 varying windows | supported for 180 full-ACDC rows | candidate yes |
| `within_session_fatigue_slope.<feature>` | Full ACDC | source x task x participant x session | sessions with at least 2 varying windows | supported for 110 full-ACDC rows | candidate yes |
| `trial_count_distribution` | Full ACDC | source x task | valid trial counts observed | supported across 57 templates | candidate yes |
| `missingness_rate.<feature>` | Full ACDC | source x task | observed feature availability | supported across 57 templates | candidate yes |
| `session_within_person_variance.<feature>` | Paired Stroop-Flanker-SART | task x participant repeated session | at least 1 participant with 2+ sessions and observed feature | supported for observed paired task features | candidate yes |
| `session_within_person_covariance.<feature_pair>` | Paired Stroop-Flanker-SART | task x participant repeated session | at least 2 complete session-deviation units | supported where feature pair observed | candidate yes |
| `repeated_person_stability.<feature>` | Paired Stroop-Flanker-SART | task x repeated participant | repeat participants only; draft freeze threshold >= 30 repeat participants | explicitly estimated by paired adapter | candidate yes |
| `session_order_context_trend.<feature>` | Paired Stroop-Flanker-SART | task x repeated participant | ordered repeated sessions; draft freeze threshold >= 30 repeat participants | supported for observed paired task features | candidate review |
| `cross_task_session_covariance_raw.<feature>` | Paired Stroop-Flanker-SART | participant x session | at least 2 complete paired task sessions | supported where common feature exists | candidate review |
| `cross_task_session_covariance_within_person.<feature>` | Paired Stroop-Flanker-SART | repeat participant x session deviation | repeat participants only; at least 2 complete deviation rows | supported where common feature exists | candidate review |
| `within_session_window_variance_from_paired` | none | not applicable | paired source is session-level only | unsupported | no |
| `lag1_autocorrelation_from_paired` | none | not applicable | paired source is session-level only | unsupported | no |
| `within_session_fatigue_from_paired` | none | not applicable | paired source is session-level only | unsupported | no |

## Exclusion Rules

The paired adapter excludes profile/probability/candidate columns before estimation:

```text
control_profile
control_profile_probability
high_engagement_candidate
low_engagement_candidate
```

These columns are prior analysis outputs and must not become empirical-background inputs.

## Paired-Source Audit Link

The paired repeated-session provenance and estimator audit is recorded in:

```text
research-program/M2_7_PAIRED_SOURCE_PROVENANCE_AUDIT.md
```

The audit resolved two draft-contract issues:

```text
session variance/covariance now use repeat participants only
repeated-person stability now has an explicit estimator
```

It also renamed across-session practice as a session-order/context trend unless
the upstream source design review justifies a pure practice interpretation.

## Open Decisions Before Freeze

1. Whether sparse ACDC templates should remain separate or be partially pooled.
2. Whether cross-task paired-session covariance should be used directly or only as a diagnostic.
3. Whether bounded and full ACDC differences require robustness bands around variance fractions.
4. How unsupported quantities are represented in the empirical-twin generator.
5. Whether to include the paired source in `EMPIRICAL_BACKGROUND_CONTRACT_V1` now or after an independent adapter review.
