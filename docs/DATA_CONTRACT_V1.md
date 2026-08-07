# Data Contract V1

## Purpose

Define the canonical analysis unit and minimum schema for the Trident control architecture validation programme. This contract allows different public datasets to be normalised into a common window/session representation without pretending that all tasks measure the same construct.

## Analysis unit

Primary unit:

```text
participant × session × task × valid block/window
```

No analytical window may cross participant, session, task or block boundaries.

## Required identity fields

```text
source_dataset
source_version
participant_id
session_id
task_id
block_id
window_id
window_start_trial
window_end_trial
n_trials_total
n_trials_valid
```

## Required provenance fields

```text
source_file_or_table
source_commit_or_release
source_hash_if_available
preprocessing_version
feature_version
```

## Core behavioural feature families

### Stable/background candidates

```text
accuracy
median_rt_ms
mean_response_speed
rt_cv
throughput_proxy
trial_count
practice_or_session_index
```

These features may contain stable participant differences. They must not be interpreted as state markers before normative decomposition.

### Dynamic/state-sensitive candidates

Where supported by the source task:

```text
conflict_cost_rt
conflict_cost_accuracy
post_error_adjustment
error_burstiness
lag1
lag2
roughness
permutation_entropy
difference_entropy
temporal_drift
sign_change_rate
large_update_rate
recovery_slope
```

Structurally unavailable features remain missing with an explicit availability flag; they are not imputed as if measured.

### Vigilance/readiness candidates

Where supported:

```text
vigilance_engagement
inhibitory_stability
reciprocal_rt
slow_tail_response_speed
lapse_rate
false_start_rate
vigilance_drift
```

## Task-design/context fields

Include all available design variables needed to avoid confusing task structure with cognitive state, for example:

```text
condition_mix
congruency_mix
switch_rate
lure_rate
difficulty_level
soa_or_foreperiod
time_on_task
response_mapping
input_device
timing_quality
browser_focus_flags
```

## Availability flags

Each feature family should have explicit availability fields or metadata. Example:

```text
has_conflict_cost
has_post_error
has_vigilance
has_switch_structure
has_confidence
has_change_point
```

## Normative outputs

For every modelled feature, the Stage 1 pipeline should emit:

```text
raw_value
population_standardised_value
expected_value_from_training_model
personal_or_hierarchical_deviation
prediction_interval_or_uncertainty
```

For previously unseen participants, personal deviation may be based only on the population/hierarchical prior until repeated observations exist.

## Splitting rules

- Train/test splits occur at participant level.
- Dataset-isolated evaluation occurs when enough sources are available.
- Session/window rows for one participant must never be divided between train and test.
- Scaling, imputation, feature selection, normative fitting and contrastive transforms are fitted on training data only.

## Missingness rules

- Preserve structural missingness.
- Do not impute a feature that a task cannot generate.
- Record technical missingness separately from structural absence.
- Report missingness by source, task and modelled feature.

## Output policy

Participant-level canonical tables are local/generated artefacts and should not be committed unless fully synthetic.

Safe Git outputs include:

```text
schema definitions
synthetic fixtures
aggregate validation summaries
run manifests without participant identifiers
```
