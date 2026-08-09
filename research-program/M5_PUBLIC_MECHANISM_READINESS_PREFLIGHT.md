# M5 Public Mechanism Readiness Preflight

**Status:** aggregate support preflight only

No mechanism model is fit. No confirmatory Trident-G, APC, PACE,
neural-criticality, cusp or transfer claim is made.

## Purpose

This preflight maps available public aggregate evidence onto the frozen
M3 variable registry before any real-data mechanism analysis.

## Variable Support Matrix

```csv
variable_id,support_status,acdc_task_support,paired_task_support,available_features,support_reason,model_fitting_allowed,real_transfer_outcomes_allowed,trident_validation_claim_allowed
K,supported,Flanker:12|Simon:5|Stroop:34,"Flanker:participants=466,repeat=210|SART:participants=466,repeat=210|Stroop:participants=466,repeat=210",accuracy|median_rt_ms|mean_response_speed|rt_cv|throughput_proxy|mean_rt_ms,ACDC supports between-person/static observed behaviour across eligible templates; paired source supports repeat/session stability for Stroop/Flanker/SART.,False,False,False
V,partial,none_for_SART,"SART:participants=466,repeat=210",anticipatory_rate|commission_rate|mean_rt_ms|omission_rate|pre_failure_speeding_ms|rt_cv,Paired SART supports session-level vigilance/readiness features with repeat participants; Full ACDC has no eligible SART window template.,False,False,False
C_signal,partial,Stroop|Flanker_core_only,"Flanker:participants=466,repeat=210|Stroop:participants=466,repeat=210",conflict_cost_accuracy|conflict_cost_rt,Paired Stroop/Flanker provide conflict-cost session summaries; Full ACDC core aggregates do not provide direct conflict-cost window features.,False,False,False
A_evidence,unsupported,none,none,none,"Current aggregate sources do not expose post-error, change-point or evidence-weighting observables required for primary Evidence updating analysis.",False,False,False
T_commit,partial_proxy_only,Flanker:12|Simon:5|Stroop:34,"Flanker:participants=466,repeat=210|SART:participants=466,repeat=210|Stroop:participants=466,repeat=210",median_rt_ms|mean_rt_ms|mean_response_speed|accuracy,"RT/speed/accuracy can describe decision timing, but no deadline or explicit speed-accuracy manipulation is available for primary Commit-threshold identification.",False,False,False
PC_calibration,unsupported,none,none,none,"No confidence, prediction-error or source-reliability observables are available in the current aggregate sources.",False,False,False
R_dynamic,partial_temporal_not_regime,Flanker:lag1_templates=13|Simon:lag1_templates=7|Stroop:lag1_templates=16,paired_source_session_summary_no_window_sequences,lag1|fatigue,"ACDC aggregate temporal support gives lag-1 and fatigue/time-on-task summaries, but this is not enough to identify dynamic regimes.",False,False,False
P_pace,descriptive_only,not_primary,not_primary,speed_accuracy_variability_patterns_only,PACE remains a descriptive phenotype until independently supported; no forced four-profile ontology is authorised.,False,False,False
Y_behavior,supported,Flanker:12|Simon:5|Stroop:34,"Flanker:participants=466,repeat=210|SART:participants=466,repeat=210|Stroop:participants=466,repeat=210",accuracy|rt|speed|variability|throughput|conflict_costs|vigilance_errors,Observed behaviour is available through ACDC core window features and paired session summaries.,False,False,False
Transfer_external,forbidden,not_applicable,not_applicable,none,Real wrapper-transfer outcomes remain external and must not be inspected or used until a later prospective prediction freeze.,False,False,False

```

## Interpretation Boundary

Supported means the current public aggregates can supply observables for
a prospective analysis design. It does not mean the latent variable is
real, validated or preferred. Partial support must remain explicitly
limited in any later model. Unsupported and forbidden variables must not
be silently estimated from proxies.