# Statistical Contract V1

## Purpose

Milestone 1.5 defines the shared scoring surface for the future M0-M5 model tournament. The contract is deliberately neutral: it must allow a null, one-factor, continuous, mixture, dynamic, source-dominated or otherwise simpler model to win.

## Universal Primary Score

The universal primary model-comparison metric is participant-isolated held-out predictive log density or log likelihood.

Reporting requirements:

- fit all preprocessing, imputation, scaling and model parameters on training participants only;
- score untouched participants or whole held-out datasets without refitting;
- report total held-out log likelihood;
- report mean held-out log density per valid canonical window;
- report mean held-out log density per observed feature value when feature availability differs across tasks.

The current primary scalar for automatic comparison is:

```text
heldout_log_density_mean_per_window
```

The observed-feature-normalised value is retained because structural missingness means some windows expose fewer modelled dimensions than others. Whether final preregistration should prefer window-normalised or observed-cell-normalised comparison remains unresolved.

## Secondary Metrics

Secondary metrics are reported where the model defines them:

- probability calibration for probabilistic class or state outputs;
- next-window or next-session prediction;
- participant bootstrap stability;
- whole-dataset transport;
- source and task predictability;
- posterior predictive checks.

These secondary metrics can overturn an apparently good density result if they reveal leakage, instability, source domination or poor transport.

## Diagnostics, Not Universal Scores

The following are diagnostics rather than universal primary scores:

- BIC and AIC;
- reconstruction error;
- adjusted Rand index;
- component entropy;
- component size;
- bootstrap component recovery;
- dynamic transition and dwell summaries.

They remain useful within model families, but they are not sufficient to select the tournament winner.

## Model Interface

Tournament models must declare:

- canonical feature requirements before fitting;
- deterministic random-state handling;
- `fit(...)` using training rows only;
- `score_samples(...)` returning held-out log density per canonical window or an explicitly equivalent density output;
- `predict_representation(...)` returning neutral numeric representations;
- serialisable metadata, including imputation and scaling provenance.

Discovery labels must remain neutral until post-hoc validation gates are passed.

## M0 Contract

Two M0 implementations are allowed at this stage:

- `M0GeneralPerformanceModel`: deterministic SVD/PCA development diagnostic with reconstruction error only;
- `M0ProbabilisticPCAModel`: formal one-factor probabilistic PCA baseline that returns held-out log density.

Only the probabilistic M0 participates in the formal primary tournament score.

## Normative Model Contract

Normative preprocessing is itself a model family:

- N0: raw/no-normalisation comparator;
- N1: simple median source/task/session residualiser;
- N2: later hierarchical location-scale model.

N1 is preserved as `N1_simple`; a later N2 may be compared against it, not silently substituted for it.

Strict prospective mode is mandatory for personal baselines. For participant session `t`, no observation from session `t` or a later session may contribute to that participant's personal baseline.

## Missingness Contract

Canonical data preserve structural missingness as `NaN`. A model may impute missing cells only inside a training-fitted adapter, and that adapter must not mutate the canonical table.

Model and run manifests must record:

- imputation strategy;
- scaling strategy;
- training feature means and scales;
- training observed, structural-missing and technical-missing counts;
- number of training rows and participant groups used to fit the adapter.

## Unresolved Choices

- Whether final model ranking should use mean log density per window or per observed feature value when tasks expose different feature sets.
- Whether probabilistic M0 should remain closed-form PPCA or move to factor analysis with feature-specific uniquenesses.
- How to harmonise density scores across models with different missing-data likelihood assumptions.
- Which calibration metric should become primary for future probabilistic state/profile outputs.
- How many bootstrap resamples are required for stability claims once M1-M5 are implemented.

