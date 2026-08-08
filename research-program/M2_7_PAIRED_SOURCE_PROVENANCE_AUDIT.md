# M2.7 Paired Source Provenance And Estimator Audit

**Status:** audit complete for scientific review; not a frozen contract  
**Stage:** M2.7 empirical-background calibration  
**Scope:** paired Stroop-Flanker-SART repeated-session source only

## Boundary

This audit estimates empirical background structure only. It does not fit
M0/M1/M2_EM/M3/M4, does not generate empirical twins and does not make APC,
PACE, Trident-state or brain-critical-zone claims.

## Local Source

```text
C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\.tmp-zone-replication\flow-zone-zone-validation\data\processed\paired_vigilance_session_features.parquet
```

Local processed-table SHA-256:

```text
DD30838E96FBC1A80C42D0CD41161644B1563BCCC3901B39AB9E9A1DED0065A0
```

Repository:

```text
https://github.com/HRP-Transfer-Lab/flow-zone-zone-validation.git
```

The local clone inspected during this audit was at:

```text
2d8d479befd73155d2215c1fef80a60cd27eaa5b
```

The paired-study manifest reports the processed feature table was generated at:

```text
e7e68aaaa18d56f46b53de6623c83f2ab5e532ed
```

Raw paired summary input recorded by the upstream manifest:

```text
data/raw/stroop_sart_flanker/STROOP_FLANKERS_SART_web_and_lab.xls
sha256: 884ba8bbae097e81f826fa247c2a0bb785302eb88173fbf88dfb5d3c76b8cd5b
```

Generation route:

```text
scripts/09_paired_vigilance_analysis.py
  reads the raw XLS summary
  reshape_paired_summary(...)
  build_session_features(...)
  writes data/processed/paired_vigilance_session_features.parquet

scripts/run_paired_study.ps1
  consumes the same processed feature table for downstream paired-study reports
```

## Session Ordering

The upstream `SESSION_SPECS` define three session labels:

```text
online
lab1
lab2
```

The adapter maps these as:

```text
online -> 1
lab1   -> 2
lab2   -> 3
```

The processed table audit found:

```text
session_type values observed: lab1; lab2; online
session_type mapped rows: 768
row-order fallback rows: 0
```

Because online and lab sessions differ in context, slopes are reported as
`session_order_context_trend`, not as pure practice effects. A future contract
may use them as practice-like realism only if the source design review
explicitly accepts that interpretation.

## Excluded Columns

The paired adapter excludes profile, probability, candidate, cluster and
latent-like columns before empirical-background estimation. The local audit
excluded:

```text
control_profile
control_profile_probability
high_engagement_candidate
low_engagement_candidate
```

These are prior analysis outputs and must not become latent-truth inputs for
M2.7.

## Support Summary

```text
input rows: 768
valid identity rows: 768
rows excluded for missing identity: 0
session-task rows emitted: 2303
participants: 466
repeat participants: 210
tasks: Flanker, SART, Stroop
```

Task support:

```text
Flanker: 768 session-task rows, 466 participants, 210 repeat participants, feature coverage min 1.000
SART:    768 session-task rows, 466 participants, 210 repeat participants, feature coverage min 0.982
Stroop:  767 session-task rows, 466 participants, 210 repeat participants, feature coverage min 1.000
```

All three task templates exceed the draft contract support threshold of 30
repeat participants.

## Estimator Audit

The paired estimator now separates components as follows:

```text
between-person variance/covariance:
  all participants with observed feature values

session-within-person variance/covariance:
  repeat participants only

repeated-person stability:
  repeat participants only

cross-task covariance:
  raw participant-session covariance
  within-person session-deviation covariance
```

The previous singleton-including session-deviation estimator was compared with
the repeat-only estimator on the observed paired feature rows:

```text
compared rows: 16
median old singleton-including session variance: 0.010489
median new repeat-only session variance:        0.015748
median new - old:                               0.005259
rows where new > old:                           16 / 16
```

This confirms the suspected downward bias: singleton participants contributed
zero deviations under the earlier calculation.

## Adapter Outputs

When supplied through `--paired-session-table`, the preflight now writes:

```text
paired_session_adapter_audit.csv
paired_session_support.csv
paired_session_feature_summary.csv
paired_session_variance_decomposition.csv
paired_session_covariance_between_person.csv
paired_session_covariance_session.csv
paired_session_repeated_person_stability.csv
paired_session_order_context_trend_summary.csv
paired_session_cross_task_covariance_raw.csv
paired_session_cross_task_covariance_within_person.csv
```

The paired source remains unsupported for:

```text
within-session/window variance
lag-1 window autocorrelation
within-session fatigue
trial-count/window missingness
```

## Recommendation

The paired source is suitable as the repeated-session development source for
`EMPIRICAL_BACKGROUND_CONTRACT_V1`, subject to scientific review of whether the
online/lab session-order trend should be used directly, treated as a context
effect, or excluded from generator parameters.

Do not freeze the empirical-background contract until that interpretation
decision is recorded.
