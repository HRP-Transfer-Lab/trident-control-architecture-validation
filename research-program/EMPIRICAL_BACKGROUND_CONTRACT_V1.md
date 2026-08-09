# Empirical Background Contract V1

**Status:** frozen M2.7 empirical-background parameter contract

**Stage:** M2.7 empirical-background calibration for later empirical-twin static recovery

**Static contract:** `STATIC_MODEL_SELECTION_CONTRACT_V2.md`

**Predecessors:** `M2_7_EMPIRICAL_TWIN_PREFLIGHT.md`, `M2_7_PAIRED_SOURCE_PROVENANCE_AUDIT.md`, `EMPIRICAL_BACKGROUND_PARAMETER_PROVENANCE_DRAFT.md`

## Boundary

This contract freezes which empirical-background quantities may be used to
parameterise later M2.7 known-truth synthetic worlds.

The empirical sources are used only for background realism:

```text
between-person variation
source/task shifts
window-level variation
lag-1 structure
within-session time-on-task trends
repeated-session variation
repeated-person stability
missingness and usable-window support
```

They must not provide latent truth and must not be used to fit or select
`M0`, `M1`, `M2_EM_v1`, `M3` or `M4` on empirical data.

This contract does not authorise empirical-twin generation, smoke recovery or
formal recovery. Stop for scientific review before any M2.7 empirical-twin
smoke/recovery run.

## Claims Boundary

No APC, PACE, Trident-state, brain-critical-zone, neural criticality or
participant-level diagnostic claim may be inferred from these estimates.

PACE/profile/probability/candidate/cluster/latent-like columns are excluded from
paired-source estimation and must remain excluded from all empirical-background
inputs.

## Frozen Source Roles

### Full ACDC Source

Primary source for source/task and window-level empirical background structure.

```text
preflight output:
reports/generated/empirical_twin_preflight_v1_full_acdc

rows: 117899
participants: 9152
sources: 57
tasks: 3
source/task templates: 57
minimum template size: 102 windows / 51 participants
repeat participants: 0
input checksum:
sha256:a2ff99c78517df555f9ba3636b0398f2c72cb60f43c2abf71905b6732b2a9172
```

ACDC may inform:

```text
source_task_shift
between_person_variance
between_person_covariance
window_within_session_variance
within_session_covariance
lag1_autocorrelation
within_session_time_on_task_trend
trial_count_distribution
missingness_rate
```

ACDC must not inform:

```text
session_within_person_variance
session_within_person_covariance
repeated_person_stability
pure across-session practice
```

### Paired Stroop-Flanker-SART Source

Primary source for repeated-session empirical background structure.

```text
local processed table:
C:\Users\admin\OneDrive\Documents\GitHub\trident-g-platform\.tmp-zone-replication\flow-zone-zone-validation\data\processed\paired_vigilance_session_features.parquet

processed-table sha256:
DD30838E96FBC1A80C42D0CD41161644B1563BCCC3901B39AB9E9A1DED0065A0

local source clone inspected at:
2d8d479befd73155d2215c1fef80a60cd27eaa5b

upstream paired-study manifest generation commit:
e7e68aaaa18d56f46b53de6623c83f2ab5e532ed

raw XLS summary hash recorded by upstream manifest:
sha256:884ba8bbae097e81f826fa247c2a0bb785302eb88173fbf88dfb5d3c76b8cd5b
```

Paired-source support:

```text
participants: 466
repeat participants: 210
tasks: Flanker, SART, Stroop
session-task rows emitted: 2303
profile/probability/candidate columns excluded: 4
```

The paired source may inform:

```text
session_within_person_variance
session_within_person_covariance
repeated_person_stability
```

The paired source may inform sensitivity-only quantities:

```text
session_order_context_trend
cross_task_session_covariance_raw
cross_task_session_covariance_within_person
```

The paired source must not inform:

```text
within_session_window_variance
lag1_window_autocorrelation
within_session_fatigue
trial_count_distribution
window_missingness
pure practice
```

## Session-Order/Context Decision

The online -> lab1 -> lab2 trend is carried as a sensitivity setting only.

It is not a primary generator session-order parameter and must not be labelled
as pure practice. The source design confounds repeated exposure with testing
context, so applying this slope directly in the primary generator would make a
context-specific design feature look like a general session-learning law.

Allowed use:

```text
parameter name: session_order_context_trend
role: sensitivity/context-effect realism only
label: session-order/context trend, not practice
reported separately from primary M2.7 recovery
```

Disallowed use:

```text
primary generator practice slope
pure session-learning parameter
latent APC/PACE/state transition proxy
post-hoc explanation for model recovery patterns
```

## Source-To-Parameter Mapping

| Parameter family | Primary source | Estimation level | M2.7 generator role |
| --- | --- | --- | --- |
| `source_task_shift.<feature>` | Full ACDC | source x task | primary |
| `between_person_variance.<feature>` | Full ACDC | source x task | primary |
| `between_person_covariance.<feature_pair>` | Full ACDC | source x task | primary if support-eligible |
| `window_within_session_variance.<feature>` | Full ACDC | source x task | primary |
| `within_session_covariance.<feature_pair>` | Full ACDC | source x task | primary if support-eligible |
| `lag1_autocorrelation.<feature>` | Full ACDC | source x task x participant x session | primary where estimated |
| `within_session_time_on_task_trend.<feature>` | Full ACDC | source x task x participant x session | primary where estimated |
| `trial_count_distribution` | Full ACDC | source x task | primary |
| `missingness_rate.<feature>` | Full ACDC | source x task | primary |
| `session_within_person_variance.<feature>` | Paired Stroop-Flanker-SART | task x repeated participant | primary for repeated-session worlds |
| `session_within_person_covariance.<feature_pair>` | Paired Stroop-Flanker-SART | task x repeated participant | primary if support-eligible |
| `repeated_person_stability.<feature>` | Paired Stroop-Flanker-SART | task x repeated participant | primary for repeated-session worlds |
| `session_order_context_trend.<feature>` | Paired Stroop-Flanker-SART | task x repeated participant | sensitivity only |
| `cross_task_session_covariance_raw.<feature>` | Paired Stroop-Flanker-SART | participant x session | sensitivity/diagnostic only |
| `cross_task_session_covariance_within_person.<feature>` | Paired Stroop-Flanker-SART | repeat participant x session deviation | sensitivity only |

## Support Thresholds

Extractor-level `estimated` rows are necessary but not sufficient for generator
use. A quantity is generator-eligible only if it also passes the thresholds
below.

### ACDC Source/Task Templates

Template-level generator eligibility:

```text
n_participants >= 50
n_rows >= 100
feature_coverage >= 0.95 for the relevant feature
```

Between-person variance:

```text
at least 30 participants with an observed participant mean
```

Between-person covariance:

```text
at least 30 complete participant means for the feature pair
```

Window-within-session variance:

```text
at least 30 sessions with repeated windows
at least 100 observed window residual rows
```

Within-session covariance:

```text
at least 100 complete within-session residual rows for the feature pair
```

Lag-1 autocorrelation:

```text
at least 30 usable within-participant-session sequences
sequence length >= 3
non-zero within-sequence variance
```

Within-session time-on-task trend:

```text
at least 30 usable sessions
at least 2 observed windows per usable session
non-zero within-session order variation
```

Missingness:

```text
observed by source x task x feature
structural missingness remains structural missingness
technical missingness remains separate where available
```

### Paired Session Source

Task-level repeated-session eligibility:

```text
n_repeat_participants >= 30
valid participant_id and session_id
no forbidden latent/profile-like input columns
```

Session-within-person variance:

```text
at least 30 repeat participants with 2+ observed sessions for the feature
repeat participants only
```

Session-within-person covariance:

```text
at least 30 complete repeat-participant session-deviation units for the feature pair
repeat participants only
```

Repeated-person stability:

```text
at least 30 repeat participants with 2+ observed sessions for the feature
repeat participants only
```

Session-order/context trend sensitivity:

```text
at least 30 repeat participants with ordered repeated sessions for the feature
reported as sensitivity only
```

Cross-task covariance sensitivity:

```text
raw covariance: at least 30 complete participant-session paired task units
within-person covariance: repeat participants only, at least 30 complete deviation units
```

## Pooling And Shrinkage Rules

No automatic pooling or statistical shrinkage is authorised for primary V1
generation.

Use template-specific estimates when a quantity is support-eligible.

If an ACDC source/task quantity is unsupported:

```text
1. keep the source/task template in the manifest;
2. mark the quantity as unsupported;
3. use a deterministic task-level fallback only if the generator requires the
   quantity for schema completion;
4. if no task-level fallback exists, use an all-ACDC fallback for the same
   feature family;
5. record the fallback source and reason in the generator manifest.
```

If a covariance matrix is not support-eligible or cannot be represented as a
valid covariance matrix without changing the scientific parameterisation:

```text
primary generator: use supported diagonal variances only
sensitivity generator: may use the repaired covariance only if the repair rule
is pre-registered before the run
```

Any hierarchical partial pooling, Bayesian shrinkage, Ledoit-Wolf shrinkage,
nearest-neighbour template borrowing or covariance repair beyond numerical
rounding requires a versioned amendment before empirical-twin generation.

## Unsupported Quantities

The following are unsupported in V1 and must not be silently imputed as
empirical estimates:

```text
ACDC repeated-session/session-to-session variance
ACDC repeated-person stability
ACDC pure across-session practice
paired within-session/window variance
paired lag-1 window autocorrelation
paired within-session fatigue/time-on-task
paired trial-count/window missingness
pure practice from online/lab ordering
APC parameters from empirical-background summaries
PACE/profile probabilities from empirical-background summaries
Trident states or brain-critical-zone quantities
```

Unsupported quantities may be represented in generator configuration only as:

```text
unsupported
disabled
fixed neutral value
pre-registered sensitivity setting
```

The representation must be explicit in the manifest.

## Sensitivity Settings

The following sensitivity settings are allowed after review and must be reported
separately from the primary M2.7 recovery result:

```text
session_order_context_trend enabled/disabled
paired cross-task raw covariance enabled/disabled
paired within-person cross-task session-deviation covariance enabled/disabled
diagonal-only covariance versus support-eligible empirical covariance
task-level fallback versus all-ACDC fallback for unsupported source/task quantities
```

Sensitivity settings must be fixed before inspecting any empirical-twin recovery
outputs.

## Output And Provenance Requirements

Every later empirical-twin run using this contract must record:

```text
contract id: EMPIRICAL_BACKGROUND_CONTRACT_V1
ACDC aggregate output directory and input checksum
paired-source aggregate output directory and processed-table checksum
support eligibility table
unsupported quantities and reasons
fallback quantities and sources
sensitivity settings
static model-selection contract id
git commit
config hash
random seed schedule
```

Participant-level empirical tables, fitted participant-level quantities and raw
or interim participant data remain outside Git.

## Review Gate

Before any M2.7 empirical-twin smoke or recovery run:

```text
1. review this contract;
2. confirm the sensitivity settings to include;
3. confirm whether primary generation should be diagonal-only or may use
   support-eligible covariance matrices;
4. create the generator configuration and manifest schema;
5. run only bounded smoke after review.
```
