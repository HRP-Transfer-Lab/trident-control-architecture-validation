# M2.7 Empirical-Twin Preflight

**Status:** preflight scaffold  
**Static contract:** `static_tournament_v2`  
**Config:** `config/empirical_twin_preflight_v1.yaml`

## Purpose

M2.7 remains a known-truth experiment. Empirical data are used only to estimate nuisance structure:

```text
between-person variance
session/window variance
feature covariance
autocorrelation
practice/fatigue slopes
trial counts
missingness
source/task shifts
```

The empirical tables must not provide latent profile/state truth.

## V2 Model Space

Primary structural candidates:

```text
M0
M1
M2_EM_v1
```

Secondary discrete falsification candidates:

```text
M3
M4
```

The main preflight question is whether the nuisance extraction pipeline can produce stable, source/task-specific templates without changing the known synthetic truth definitions.

## Preflight Gates

1. Validate canonical window-table schema.
2. Confirm no synthetic or ground-truth columns are used as empirical nuisance inputs.
3. Estimate nuisance by source/task template.
4. Report missingness, feature covariance, variance decomposition, temporal autocorrelation and practice/fatigue slopes.
5. Run smoke only; no scientific model-recovery claim is allowed.

## Compute Rule

The preflight is allowed to run inside Codex only as a bounded smoke or pilot. Confirmatory M2.7 remains checkpointed, resumable and outside Codex unless explicitly requested.

## Smoke Result

The initial preflight smoke used the canonical synthetic fixture to validate the nuisance extraction pipeline only.

```text
input_mode: fallback_smoke_fixture
rows: 178
participants: 36
sources: 3
tasks: 4
source/task templates: 12
runtime: 4.355 seconds
formal_claims_allowed: false
```

Generated outputs:

```text
reports/generated/empirical_twin_preflight_v1/template_summary.csv
reports/generated/empirical_twin_preflight_v1/template_covariance.csv
reports/generated/empirical_twin_preflight_v1/variance_decomposition.csv
reports/generated/empirical_twin_preflight_v1/temporal_summary.csv
reports/generated/empirical_twin_preflight_v1/missingness_summary.csv
```

This validates the nuisance-extraction path. It does not test model recovery and should not be interpreted scientifically.
