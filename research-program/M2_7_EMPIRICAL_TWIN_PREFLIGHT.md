# M2.7 Empirical-Twin Preflight

**Status:** strengthened preflight scaffold; first bounded ACDC empirical-background preflight completed
**Static contract:** `static_tournament_v2`  
**Config:** `config/empirical_twin_preflight_v1.yaml`

## Purpose

M2.7 remains a known-truth experiment. Empirical data are used only to estimate empirical background structure:

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

For the M2.7 static architecture-recovery question, these empirical background
quantities function statistically as nuisance parameters because they are not
the known latent structural truth being adjudicated.

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

The main preflight question is whether the empirical-background extraction
pipeline can produce stable, source/task-specific templates without changing
the known synthetic truth definitions.

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

Generated smoke outputs:

```text
reports/generated/empirical_twin_preflight_v1/template_summary.csv
reports/generated/empirical_twin_preflight_v1/template_support.csv
reports/generated/empirical_twin_preflight_v1/variance_decomposition.csv
reports/generated/empirical_twin_preflight_v1/covariance_raw.csv
reports/generated/empirical_twin_preflight_v1/covariance_between_person.csv
reports/generated/empirical_twin_preflight_v1/covariance_session.csv
reports/generated/empirical_twin_preflight_v1/covariance_within_session.csv
reports/generated/empirical_twin_preflight_v1/temporal_summary.csv
reports/generated/empirical_twin_preflight_v1/missingness_summary.csv
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_summary.json
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_provenance.json
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_report.md
```

This validates the nuisance-extraction path. It does not test model recovery and should not be interpreted scientifically.

## Strengthened Empirical Background Decomposition

The preflight now separates empirical background structure into explicit support-checked
levels:

```text
source/task template support
between-person variance and covariance
session-within-participant variance and covariance
window-within-session variance and covariance
lag-1 autocorrelation within source x task x participant x session
across-session practice slopes
within-session time-on-task/fatigue slopes
missingness by source/task/feature
```

Sparse templates are not discarded or automatically pooled. Unsupported
components are reported as unsupported/NA with a support reason. Any later
pooling or shrinkage rule must be written into a versioned empirical-twin
contract before confirmatory generation.

## Real Empirical Run Command

Once a canonical empirical window table is available, run:

```powershell
$env:PYTHONPATH='src'
python -m trident_validation.synthetic.empirical_twin_preflight `
  --input-table C:\path\to\canonical_windows.csv `
  --no-fallback-fixture
```

Multiple tables can be supplied by repeating `--input-table`.

If the empirical source is the Flow Zone ACDC pipeline, regenerate or obtain:

```text
research/flow-zone-zone-validation/data/processed/cognitive_windows.parquet
```

That file is derived from the GitHub-hosted ACDC database release used by the Flow Zone validation pipeline. The preflight runner now adapts Flow Zone cognitive-window columns into the canonical M2.7 window schema before validation.

## ACDC Input Regeneration And First Real Preflight

Local search on the current workstation found the Flow Zone repository at:

```text
C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\research\flow-zone-zone-validation
```

Repository provenance:

```text
repository: HRP-Transfer-Lab/flow-zone-zone-validation
commit: 2d8d479befd73155d2215c1fef80a60cd27eaa5b
```

The intended ACDC empirical-background input was initially not present:

```text
data/processed/cognitive_windows.parquet
```

The local `data/processed/` directory contains only `.gitkeep`, consistent with
the upstream policy that raw, interim and processed participant-level data are
excluded from Git.

The existing Flow Zone command to regenerate the development ACDC window table
is:

```powershell
cd C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\research\flow-zone-zone-validation
.\scripts\run_pipeline.ps1 -SkipModels
```

or, stage by stage:

```powershell
Rscript scripts/00_download_acdc.R
Rscript scripts/01_inventory_acdc.R
Rscript scripts/02_extract_trials_acdc.R
.\.venv\Scripts\python.exe scripts/02b_build_pilot_subset.py
.\.venv\Scripts\python.exe scripts/03_build_cognitive_windows.py `
  --input data/interim/acdc_pilot_trial_extract.parquet
```

For a full-data formal calibration candidate, Flow Zone documents:

```powershell
.\scripts\run_pipeline.ps1 -AcdcTag "initial-release" -FullData -SkipModels
```

The public ACDC database source is `jstbcs/acdc-database`; the visible GitHub
release metadata showed `initial-release` as the latest release, published
2026-02-19. The Flow Zone downloader verifies the database release hash through
`acdcquery::check_acdc()` and records release tag, published time, SHA-256,
package versions and manifest metadata.

After restoring the Flow Zone Python and R environments, the bounded Flow Zone
pipeline generated:

```text
C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\research\flow-zone-zone-validation\data\processed\cognitive_windows.parquet
```

The first real ACDC empirical-background preflight then completed:

```text
input_mode: empirical_window_tables
rows: 7882
participants: 240
sessions: 240
sources: 12
tasks: 3
source/task templates: 12
runtime: 19.477 seconds
formal_claims_allowed: false
model_recovery_performed: false
```

Input provenance recorded by the preflight:

```text
input checksum:
sha256:f63344861fe984db85c7ff0816a9c8aca712a38d4af440260cffa604164d9d66

Flow Zone repository:
HRP-Transfer-Lab/flow-zone-zone-validation

Flow Zone commit:
2d8d479befd73155d2215c1fef80a60cd27eaa5b

validation repository commit:
64c9ae6d86c334a9e3ec74a923cba435812e87d0
```

Generated aggregate outputs:

```text
reports/generated/empirical_twin_preflight_v1/template_summary.csv
reports/generated/empirical_twin_preflight_v1/template_support.csv
reports/generated/empirical_twin_preflight_v1/feature_summary.csv
reports/generated/empirical_twin_preflight_v1/variance_decomposition.csv
reports/generated/empirical_twin_preflight_v1/covariance_raw.csv
reports/generated/empirical_twin_preflight_v1/covariance_between_person.csv
reports/generated/empirical_twin_preflight_v1/covariance_session.csv
reports/generated/empirical_twin_preflight_v1/covariance_within_session.csv
reports/generated/empirical_twin_preflight_v1/temporal_summary.csv
reports/generated/empirical_twin_preflight_v1/missingness_summary.csv
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_summary.json
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_provenance.json
reports/generated/empirical_twin_preflight_v1/empirical_twin_preflight_report.md
```

These generated reports are aggregate/local artefacts. The participant-level
ACDC table remains outside Git.

## First ACDC Support Findings

The bounded ACDC development extract is useful for source/task-specific
between-person and within-session empirical background structure, but it is not
a repeated-session dataset:

```text
n participants: 240
n sessions: 240
n repeat participants: 0 in every source/task template
```

Immediate support implications:

```text
between-person variance/covariance: supported
session-within-participant variance/covariance: unsupported in this bounded extract
window-within-session variance/covariance: mostly supported
lag-1 autocorrelation: mostly supported
across-session practice slopes: unsupported
within-session fatigue/time-on-task slopes: mostly supported
```

Template support:

```text
12 source/task templates
20 participants per template
65 to 1454 windows per template
feature coverage minimum: 0.981 to 1.000
```

The smallest template was source `3` / task `Simon` with 65 windows. This is
adequate for a first bounded plumbing run but should be reviewed before it is
treated as an independent empirical-background template.

Missingness was low. The largest observed missingness was in source `52` /
task `Stroop`, where `median_rt_ms`, `mean_response_speed`, `rt_cv`, and
`throughput_proxy` each had 28 missing rows:

```text
28 / 1454 = 0.0193
```

No M0/M1/M2_EM/M3/M4 model fitting or empirical-twin recovery was performed.
No APC, PACE, Trident-state or brain-critical-zone inference was made from
these empirical background estimates.

## Full ACDC Empirical Background Preflight

The full ACDC Flow Zone window table initially failed canonical validation
because five rows lacked `participant_id`. These rows were not assigned
synthetic identifiers. They were excluded into a local filtered input with an
explicit audit:

```text
audit:
C:\trident-runs\M2.7\inputs\acdc_full_identity_exclusion_audit.json

input rows: 117904
output rows: 117899
excluded rows with missing participant_id: 5
excluded source/task: dataset 59 / Stroop
```

The filtered participant-identity-valid full ACDC input was:

```text
C:\trident-runs\M2.7\inputs\acdc_full_identity_valid_cognitive_windows.parquet
```

The full ACDC empirical-background preflight completed:

```text
input_mode: empirical_window_tables
rows: 117899
participants: 9152
sessions: 9152
sources: 57
tasks: 3
source/task templates: 57
runtime: 125.798 seconds
formal_claims_allowed: false
model_recovery_performed: false
```

Input checksum recorded by the preflight:

```text
sha256:a2ff99c78517df555f9ba3636b0398f2c72cb60f43c2abf71905b6732b2a9172
```

Generated aggregate outputs:

```text
reports/generated/empirical_twin_preflight_v1_full_acdc/template_summary.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/template_support.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/feature_summary.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/variance_decomposition.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/covariance_raw.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/covariance_between_person.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/covariance_session.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/covariance_within_session.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/temporal_summary.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/missingness_summary.csv
reports/generated/empirical_twin_preflight_v1_full_acdc/empirical_twin_preflight_summary.json
reports/generated/empirical_twin_preflight_v1_full_acdc/empirical_twin_preflight_provenance.json
reports/generated/empirical_twin_preflight_v1_full_acdc/empirical_twin_preflight_report.md
```

Full ACDC support findings:

```text
templates: 57
minimum template size: 102 windows / 51 participants
median template size: 882 windows / 132 participants
maximum template size: 10583 windows
repeat participants: 0
```

All 57 templates had between-person support. None had repeated-session support.
Therefore ACDC should inform between-person, source/task, within-session/window,
lag-1 and missingness/background quantities, but not session-to-session variance
or across-session practice.

Temporal support rows:

```text
lag-1 estimated, practice unsupported, fatigue estimated: 110
lag-1 estimated, practice unsupported, fatigue unsupported: 70
lag-1 unsupported, practice unsupported, fatigue unsupported: 105
```

Missingness remained low. The largest core-feature missingness was:

```text
dataset 35 / Stroop / rt_cv:
4 / 102 = 0.0392
```

Bounded-to-full comparison:

```text
bounded templates: 12
full templates: 57

median between-person variance fraction:
bounded 0.771
full    0.920

median window-within-session variance fraction:
bounded 0.229
full    0.080

mean lag-1 autocorrelation:
bounded 0.406
full    0.406
```

The full run increases source/task coverage and precision. It also shows that
the bounded subset underweighted the between-person share of variance relative
to the full development extract. Lag-1 autocorrelation was stable on average.

No model recovery was run after the full ACDC preflight. The next scientific
decision is whether to build a separate paired Stroop-Flanker-SART adapter for
repeated-session empirical background quantities before freezing
`EMPIRICAL_BACKGROUND_CONTRACT_V1`.

## Paired Repeated-Session Adapter

A local temporary Flow Zone replication clone contains:

```text
C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\.tmp-zone-replication\flow-zone-zone-validation\data\processed\paired_vigilance_session_features.parquet
```

The paired-source provenance and estimator audit is recorded in:

```text
research-program/M2_7_PAIRED_SOURCE_PROVENANCE_AUDIT.md
```

Aggregate inspection found:

```text
rows: 768
level: session summary
tasks represented: Stroop, Flanker, SART
participant/session fields: present
profile/probability columns: present
processed table sha256: DD30838E96FBC1A80C42D0CD41161644B1563BCCC3901B39AB9E9A1DED0065A0
```

This table is not a canonical window table. A narrow paired-session adapter was
added to estimate repeated-session empirical background quantities only. It
does not create artificial window rows.

The adapter excludes profile, probability, cluster/candidate and latent-like
columns before estimation. The local audit reported:

```text
input rows: 768
rows with valid identity: 768
rows excluded for missing identity: 0
forbidden columns excluded: 4
excluded columns:
control_profile
control_profile_probability
high_engagement_candidate
low_engagement_candidate
session-level rows emitted: 2303
session_type values observed: lab1; lab2; online
session order fallback rows: 0
```

Paired support:

```text
Flanker:
session-task rows: 768
participants: 466
sessions: 768
repeat participants: 210
supported features: 5
feature coverage min: 1.000

SART:
session-task rows: 768
participants: 466
sessions: 768
repeat participants: 210
supported features: 6
feature coverage min: 0.982

Stroop:
session-task rows: 767
participants: 466
sessions: 767
repeat participants: 210
supported features: 5
feature coverage min: 1.000
```

The adapter estimates:

```text
between-person session-level variance/covariance
session-within-person variance/covariance using repeat participants only
repeated-person stability
session-order/context trends
raw cross-task covariance within participant-session
within-person session-deviation cross-task covariance
```

The session-order trend is not labelled as pure practice because online, lab1
and lab2 may mix learning with testing-context changes.

It explicitly leaves unsupported:

```text
within-session/window variance
lag-1 window autocorrelation
within-session fatigue
trial-count/window missingness
```

The paired adapter therefore fills the repeated-session timescale that ACDC
cannot identify, while ACDC remains primary for window-level and source/task
background structure.

The singleton-participant estimator audit confirmed that the earlier
session-deviation calculation understated session variance. Across the 16
observed paired feature rows, median session variance increased from `0.010489`
under the singleton-including calculation to `0.015748` under the repeat-only
calculation.

No M0/M1/M2_EM/M3/M4 model fitting or empirical-twin recovery was performed
with the paired source.

## Strengthened Smoke Result

A bounded synthetic-fixture smoke of the strengthened extractor completed:

```text
input_mode: fallback_smoke_fixture
rows: 178
participants: 36
sources: 3
tasks: 4
source/task templates: 12
runtime: 18.072 seconds
formal_claims_allowed: false
```

This confirms the strengthened nuisance-output path only. It is not an
empirical run and makes no scientific model-recovery, APC, PACE or Trident
claim.

The YAML config also accepts either plain paths:

```yaml
inputs:
  empirical_window_tables:
    - data/canonical_windows.csv
```

or path entries:

```yaml
inputs:
  empirical_window_tables:
    - path: data/canonical_windows.csv
      role: empirical_nuisance_template_source
```
