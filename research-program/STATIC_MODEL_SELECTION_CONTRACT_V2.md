# Static Model-Selection Contract V2

**Status:** Draft contract for M2.7 preflight  
**Predecessor:** M2.6/M2.6b static tournament V1, frozen  
**Reason for V2:** `M2_closed_form_v1` failed exact nonlinear-vigilance recovery; `M2_EM_v1` repaired the estimator failure in M2.6c diagnostics.

## Model Space

| Model | Tier | Role |
| --- | ---: | --- |
| M0 probabilistic general performance | 0 | broad general-factor baseline |
| M1 continuous control manifold | 1 | structured continuous multidimensional manifold |
| M2_EM_v1 | 1 | structured continuous nonlinear readiness/manifold |
| M3 three-profile mixture | 2 | secondary discrete falsification candidate |
| M4 four-PACE-profile mixture | 2 | secondary discrete falsification candidate |

The original M2 remains in the historical record as:

```text
M2_closed_form_v1
status: failed estimator / frozen M2.6 model
```

`M2_EM_v1` is a new versioned candidate:

```text
quadrature_points = 201
max_iter = 120
strict optimiser convergence required for valid score = no
convergence metadata required = yes
```

## Selection Rule

1. Compare participant-isolated held-out log density per window.
2. Estimate paired participant-level uncertainty against the numerical best model.
3. If a lower-tier model is not meaningfully worse than the numerical best, prefer the lower tier.
4. Within the same tier, do not impose arbitrary numeric-order simplicity. If same-tier models are practically tied, report ambiguity.
5. M3/M4 remain in the tournament as discrete falsification candidates, not as default PACE ontology.

This means M1 and M2_EM_v1 are treated as alternative structured continuous explanations rather than as a simple nested sequence.

## M2_EM_v1 Evidence

M2.6c convergence diagnostic:

```text
80  iter: EM beats M1 1.00, mean EM-M1 0.470
120 iter: EM beats M1 1.00, mean EM-M1 0.475
200 iter: EM beats M1 1.00, mean EM-M1 0.474
```

The 120-iteration cap is the efficient setting: it captures the small gain over 80 without the extra optimisation cost of 200.

EW2 exact confirmation with `M2_EM_v1`:

```text
n = 30
EM numerical-best rate = 1.00
EM selected rate       = 0.967
mean EM-M1             = 0.496
```

## EW0/M1 Nesting Diagnostic

Small EW0 diagnostic:

```text
n = 12
M1 numerical-best rate                 = 1.00
mean M1-M0 held-out density            = 0.052
mean second/first eigenvalue ratio     = 0.044
mean second loading fraction           = 0.123
mean bootstrap second-dimension stability = 0.990
substantive second-dimension rate      = 0.00
decision = m1_numerically_exploits_ew0_without_stable_second_dimension
```

Interpretation: M1 can exploit finite-sample covariance under exact M0 data, but it does not recover a substantive second dimension by the current magnitude threshold. V2 should therefore treat M0/M1 ties as practical nesting/ambiguity unless M1 shows both credible predictive improvement and a non-negligible second dimension.

## M2.7 Implication

M2.7 should use the V2 static model space:

```text
Primary:
M0
M1
M2_EM_v1

Secondary discrete falsification:
M3
M4
```

The primary static question becomes:

> Can `M2_EM_v1` remain distinguishable from M1 once empirical-twin nuisance structure is introduced?
