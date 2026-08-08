# M2.6 Formal Synthetic Recovery Interpretation

**Status:** Post-run scientific gate note, revised after adjudication  
**Run:** `formal_synthetic_recovery_v1`  
**Generated report:** `reports/generated/formal_synthetic_recovery_v1.md`  
**Manifest:** `manifests/local/formal_synthetic_recovery_v1_manifest.json`  
**Git commit at aggregation:** `8750a03aeb496e56448b89ebef4a930176852f94`

## Execution

The formal checkpoint run completed and aggregated successfully.

The aggregation emitted a pandas mixed-type warning while reading the compressed replicate audit. This was traced to baseline string labels mixed with numeric stress values during a manifest row-count readback, not to failed fits or stress mis-grouping. The aggregation readback has been patched to use a single-pass dtype inference.

The manifest records:

```text
replicate audit rows: 28000
baseline replicates per world: 200
stress replicates per world per level: 40
checkpoint directory: C:\trident-runs\M2.6\checkpoints
BLAS threads: 1
git status at start: clean
```

## Baseline Recovery Matrix

| True world | Aligned model | Recovery |
| --- | --- | ---: |
| W0 general performance | M0 probabilistic general performance | 0.000 |
| W1 continuous manifold | M1 continuous control manifold | 0.960 |
| W2 nonlinear vigilance | M2 nonlinear vigilance | 0.510 |
| W3 three-profile mixture | M3 three-profile mixture | 0.005 |
| W4 four-PACE mixture | M4 four-PACE mixture | 0.025 |

The primary positive result is strong recovery of W1 and moderate/ambiguous recovery of W2.

The primary negative result is poor aligned recovery of W0, W3 and W4.

The W0 failure is diagnostically different from the W3/W4 failures. W3/W4 may indicate that the conceptual profile worlds do not provide enough held-out evidence for static discrete classes. W0 is the simplest aligned structure and should not be ignored before adding empirical nuisance.

## False Discrete Structure

The pooled false-discrete rate under non-discrete W0-W2 worlds was:

```text
7 / 600 = 0.0117
95% CI [0.0057, 0.0239]
```

This is reassuring. The tournament rarely invents discrete profile structure when the generating truth is general, continuous, or nonlinear.

## Mixture Failure Pattern

W3 and W4 are not being rejected because the mixture fits always fail numerically.

For W3, the numerical best model was often the four-component mixture, but the frozen selection rule usually chose the simpler continuous model:

```text
W3 numerical best:
M4: 167 / 200
M1:  32 / 200
M2:   1 / 200

W3 selected:
M1: 158 / 200
M4:  38 / 200
M2:   3 / 200
M3:   1 / 200
```

For W4, the continuous model was the numerical best in most baseline replicates:

```text
W4 numerical best:
M1: 123 / 200
M4:  70 / 200
M2:   7 / 200

W4 selected:
M1: 195 / 200
M4:   5 / 200
```

Component diagnostics show non-collapsed mixture fits and moderate component recovery, but that is not enough to make the discrete models win under the registered held-out selection rule.

## Scientific Interpretation

M2.6 is a successful falsification experiment that exposed substantial limits in the current static adjudicator. It is not a clean model-recovery pass.

The current static tournament supports carrying forward:

```text
M1 continuous control manifold
M2 nonlinear vigilance, with ambiguity against M1
```

It does not support treating the current M3/M4 discrete PACE mixture models as identifiable latent architectures from these static behavioural summaries.

The PACE interpretation should therefore remain phenotypic or descriptive unless later designs add independent evidence that separates discrete classes from continuous regions of control space.

## Decision

Do not expand into new mechanistic dynamic models on the assumption that W3/W4 were validated.

Do not retune M0-M4 or W0-W4 in response to this result and then claim the same formal M2.6 gate.

Do not move directly to M2.7 until a small exact-model diagnostic resolves whether the M0, M2, M3 and M4 failures are caused by the statistical machinery or by mismatch between the conceptual synthetic worlds and the fitted likelihoods.

The immediate next step is M2.6b exact-model recovery:

```text
EW0 = data generated from exact M0 assumptions
EW1 = data generated from exact M1 assumptions
EW2 = data generated from exact M2 assumptions
EW3 = data generated from exact M3 assumptions
EW4 = data generated from exact M4 assumptions
```

Each EW dataset is then scored with the same M0-M4 held-out selection machinery.

M2.7 empirical-twin recovery should come after M2.6b, with the hypothesis space explicitly constrained by both M2.6 and M2.6b:

```text
primary candidate already supported: M1
M2: carry forward only with explicit ambiguity/diagnostic status
PACE mixture models: exploratory/descriptive diagnostics unless M2.6b shows clean exact-model recovery
claim boundary: no latent-class PACE claim from static M2.6 evidence
```

## M2.6b Exact-Model Diagnostic Status

A small M2.6b diagnostic runner has been added:

```text
config/exact_model_recovery_v1.yaml
src/trident_validation/synthetic/exact_model_diagnostic.py
tests/test_exact_model_diagnostic.py
```

The initial 25-task pilot used five exact replicates per EW world. It was not confirmatory, but it indicated the diagnostic was informative:

| Exact world | Aligned model | Pilot recovery |
| --- | --- | ---: |
| EW0 M0 exact | M0 | 0.40 |
| EW1 M1 exact | M1 | 1.00 |
| EW2 M2 exact | M2 | 0.00 |
| EW3 M3 exact | M3 | 1.00 |
| EW4 M4 exact | M4 | 1.00 |

This pilot suggests the static mixture fitting machinery can recover deliberately exact, well-separated mixtures. That strengthens the interpretation that the original W3/W4 failures are likely conceptual-world/scoring-evidence issues rather than a blanket inability to fit mixtures.

The EW0 and EW2 pilot results remain unresolved. EW0 is often numerically best under M1, and EW2 is numerically best under M1 in the pilot. These should be diagnosed before empirical-twin work.

The full default M2.6b run used 30 exact replicates per EW world:

| Exact world | Aligned model | Recovery | Interpretation |
| --- | --- | ---: | --- |
| EW0 M0 exact | M0 | 0.60 | moderate/ambiguous; M1 usually numerically best |
| EW1 M1 exact | M1 | 1.00 | strong |
| EW2 M2 exact | M2 | 0.00 | poor; M1 numerically best in 29/30 runs |
| EW3 M3 exact | M3 | 1.00 | strong |
| EW4 M4 exact | M4 | 1.00 | strong |

The full diagnostic materially changes the mixture interpretation: the static tournament can recover exact M3 and exact M4 mixture likelihoods when the data actually contain strong diagonal Gaussian mixture evidence. Therefore the original W3/W4 failures are not primarily a generic mixture-fitting failure.

EW0 remains partly unresolved. In EW0, M1 was numerically best in 29/30 runs, but the uncertainty/simplicity rule selected M0 in 18/30 runs. This suggests M0 and M1 are close under one-factor data, and M1's extra dimension can exploit finite-sample covariance enough to win numerically.

EW2 is the serious exact-model failure. M1 was numerically best in 29/30 EW2 runs, and M2 was never selected. Mean held-out log density per window under EW2 was approximately:

```text
M1: -1.632
M0: -1.965
M4: -2.027
M2: -2.424
M3: -2.503
```

This is not a conservative-selection artefact. M2 is losing numerically to M1 under the current exact EW2 generator and fitted M2 scoring path.

## M2 Self-Check Diagnostic

A targeted M2 self-check was added after the full M2.6b result. It generates data from a known M2 latent quadratic likelihood and compares:

```text
oracle M2 likelihood
fitted M2 with default 31-node quadrature
fitted M2 with 201-node quadrature
fitted M1
```

The 12-replicate diagnostic produced:

```text
oracle M2 beats M1 rate:                 1.00
fitted M2 default beats M1 rate:         0.00
fitted M2 high-quadrature beats M1 rate: 0.00
projection-search M2 beats M1 rate:      0.00
oracle-projection fitted M2 beats M1:    0.00

mean oracle M2 log density:              0.164
mean fitted M1 log density:             -0.308
mean fitted M2 high-quadrature density: -0.400
mean projection-search M2 density:      -0.406
mean oracle-projection M2 density:      -0.417
mean fitted M2 default density:         -1.939

mean fitted M2 projection alignment:      0.648
mean projection-search alignment:         0.687
```

This rules out the strongest concern that the M2-shaped diagnostic data are inherently indistinguishable from M1: the oracle M2 likelihood beats M1 in every self-check replicate.

It also shows that default M2 quadrature is too coarse for this narrow curved latent manifold, but quadrature alone is not the whole failure. Raising fitted M2 scoring to 201 nodes improves mean held-out density by more than 1.5 log-density units per window, but fitted M2 still remains below M1 on average.

An experimental projection-search M2 variant evaluated PCA directions plus deterministic random projections and selected the projection with best training likelihood. It did not beat M1 in any self-check replicate. A diagnostic oracle-projection fitted M2, which uses the oracle latent direction but still estimates the quadratic curve and residual variance from training data, also did not beat M1.

This makes a pure projection-search repair insufficient. The remaining failure is more likely in the fitted M2 likelihood formulation: coefficient/residual estimation, the Gaussian residual assumption around a narrow curved manifold, or a mismatch between the empirical training-standardised fitted model and the oracle generative density.

An experimental latent-curve EM M2 variant was then added to the same self-check. This variant keeps the same one-dimensional quadratic latent curve family, but estimates curve coefficients and residual variances by alternating posterior node responsibilities with weighted curve updates.

The same 12-replicate self-check produced:

```text
latent-curve EM M2 beats M1 rate:        1.00
mean fitted EM M2 log density:          -0.020
mean fitted EM M2 minus M1:              0.288
mean oracle minus fitted EM M2:          0.184
mean EM iterations:                     80.0
```

This is a material repair. The EW2 failure is not evidence that the nonlinear-vigilance architecture is intrinsically indistinguishable from M1. It is evidence that the original closed-form fitted M2 estimator is miscalibrated for the latent quadratic likelihood used in the exact diagnostic.

The EM fit still trails the oracle likelihood, and it reached the current iteration cap on average. That means it should be treated as a diagnostic repair candidate, not immediately substituted into the frozen M2.6 result.

An EW2-only exact-model diagnostic then appended the latent-curve EM M2 candidate to the standard M0-M4 scorer for 10 exact EW2 replicates. This was a diagnostic slice, not an amendment to the frozen M2.6b result.

```text
EW2 EM diagnostic runs:                  10
EM M2 numerical-best rate:               1.00
EM M2 selected rate:                     1.00
standard M2 selected rate:               0.00
mean EM M2 log density:                 -1.273
mean M1 log density:                    -1.745
mean standard M2 log density:           -2.459
mean EM M2 minus M1:                     0.472
mean EM M2 minus standard M2:            1.186
decision: em_m2_repairs_ew2_exact_selection
```

This confirms that the M2 exact-model failure is specifically a failure of the original closed-form fitted M2 estimator. When the same EW2 exact-world data are scored with a latent-curve EM estimator, the nonlinear-vigilance model cleanly recovers itself in this cheap diagnostic slice.

## M2.6c M2-EM Hardening Diagnostic

The latent-curve EM estimator was then treated as a new versioned diagnostic candidate rather than a silent replacement for the failed original M2 estimator.

A convergence grid was run on 12 exact EW2 datasets:

```text
max_iter: 80, 120, 200
quadrature points: 201
workers: 2 thread
```

The summary was:

| max_iter | EM beats M1 | mean EM-M1 | mean held-out density | convergence | final LL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80 | 1.00 | 0.470 | -1.212 | 0.00 | 0.0526 |
| 120 | 1.00 | 0.475 | -1.208 | 0.00 | 0.0052 |
| 200 | 1.00 | 0.474 | -1.208 | 0.67 | 0.0008 |

The strict likelihood tolerance is not usually met by 80 or 120 iterations, but held-out performance is already stable. Moving from 80 to 120 improves mean held-out density by only about `0.0048` per window. Moving from 120 to 200 improves training likelihood and convergence flags, but does not improve mean held-out density.

The efficient `M2_EM_v1` confirmation setting is therefore:

```text
M2_EM_v1
quadrature_points = 201
max_iter = 120
claim boundary = versioned diagnostic candidate, not retroactive M2.6 replacement
```

A 30-replicate EW2 exact confirmation was then run with this setting:

```text
EW2 EM confirmation runs:                30
EM M2 numerical-best rate:               1.00
EM M2 selected rate:                     0.967
standard M2 selected rate:               0.00
mean EM M2 log density:                 -1.136
mean M1 log density:                    -1.632
mean standard M2 log density:           -2.424
mean EM M2 minus M1:                     0.496
mean EM M2 minus standard M2:            1.288
decision: em_m2_repairs_ew2_exact_selection
```

The one non-EM selection was M4, even though EM M2 was the numerical best in all 30 runs. This is conservative against EM because the diagnostic selection rule treats the appended EM candidate as outside the registered M0-M4 simplicity order.

This confirms the repair at the same replicate count as the full M2.6b exact diagnostic. The next versioned tournament should include M2_EM_v1 explicitly in the model contract rather than appending it as an out-of-order diagnostic candidate.

## Static Model-Selection Contract V2

A draft V2 static model-selection contract has been added:

```text
research-program/STATIC_MODEL_SELECTION_CONTRACT_V2.md
```

The V2 contract treats M1 and M2_EM_v1 as same-tier structured continuous alternatives rather than forcing them into an arbitrary linear simplicity order:

| Tier | Models | Meaning |
| --- | --- | --- |
| 0 | M0 | broad general-factor baseline |
| 1 | M1, M2_EM_v1 | alternative structured continuous explanations |
| 2 | M3, M4 | secondary discrete falsification candidates |

The V2 selection rule compares participant-isolated held-out log density, uses paired uncertainty against the numerical best model, prefers lower structural tiers when practically tied, and reports same-tier ambiguity rather than manufacturing a winner by numeric model order.

## EW0/M1 Nesting Diagnostic

A small EW0 diagnostic was run to determine whether M1's numerical advantage under exact M0 data reflects a substantive second dimension.

```text
EW0/M1 nesting diagnostic runs:          12
bootstrap samples per run:               16
M1 numerical-best rate:                  1.00
mean M1 minus M0 held-out density:       0.052
mean paired M1 minus M0:                 0.051
mean second/first eigenvalue ratio:      0.044
mean second loading fraction:            0.123
mean bootstrap second-dim stability:     0.990
substantive second-dimension rate:       0.00
decision: m1_numerically_exploits_ew0_without_stable_second_dimension
```

This supports the nesting interpretation. M1 can exploit finite-sample covariance under exact M0 data, but the second dimension is too small by the current magnitude threshold to count as a substantive recovered dimension. Future V2 interpretation should therefore separate M1's marginal predictive gain from evidence for a real second continuous dimension.

## Efficient Next Work

Before running any new confirmatory computation:

1. Prepare the M2.7 empirical-twin preflight using the V2 static model space.
2. Treat M3/M4 as secondary discrete falsification candidates, not default PACE ontology.
3. Use a small M2.7 pilot to estimate runtime and check whether M0/M1/M2_EM_v1 are separable under empirical nuisance.
4. Run confirmatory M2.7 only after the preflight and pilot show useful signal.

The M2.6 result should be treated as frozen input to M2.7 design.
