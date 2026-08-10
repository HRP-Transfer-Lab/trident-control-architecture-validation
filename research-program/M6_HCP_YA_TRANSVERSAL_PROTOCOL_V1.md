# M6 HCP-YA Transversal K/C/V Protocol V1

**Status:** data-support preflight protocol

This protocol implements the next empirical strategy after the paired
Stroop/Flanker/SART public analyses. It does not fit HCP models yet.

No Trident-G, APC, PACE ontology, transfer, Predictive Calibration, T_commit,
dynamic-regime, neural-criticality or cusp claim is authorised.

## Question

```text
Are K/C/V genuinely transversal constraints on cognition, or are they mostly
features of attention/control tasks?
```

The first HCP-YA test separates two questions:

1. Do the same lower-level K/C/V coordinates predict different levels of the
   stack?
2. Do the weights of K/C/V change across attention/control, working memory and
   reasoning outcomes?

## Lower-Level Coordinates

The lower-level predictors are:

```text
K = broad non-target performance/general capacity indicators
C_signal = Flanker/interference-control candidate
V = sustained-attention/CPT candidate
```

These remain behavioural predictors, not validated Trident-G/APC variables.

## Held-Out Outcomes

The intended held-out outcome domains are:

```text
attention/control
working memory: List Sorting
working memory: 2-back task performance
reasoning: PMAT
```

The anti-circularity rule is hard:

```text
an outcome column must not be used to construct K/C/V for that same outcome
```

List Sorting cannot enter K if List Sorting is the WM outcome. PMAT cannot
enter K if PMAT is the reasoning outcome. The same rule applies to any
attention/control outcome.

## Primary Model Sequence

For each supported domain `Y`, the prospective model sequence is:

```text
Y ~ K
Y ~ K + C_signal
Y ~ K + V
Y ~ K + C_signal + V
Y ~ K + C_signal + V + C_signal x V
```

The primary score remains participant-isolated or family-isolated held-out
predictive log density.

## Domain Weight Test

If the data-support gate passes, the analysis will report whether coefficients
for K/C/V differ by domain:

```text
Y_domain = alpha_domain K + beta_domain C_signal + gamma_domain V + error
```

This is a transversal-weight test. It is not a claim that K/C/V are proven
mechanisms.

## Layer-Specific Residual Tests

Only after K/C/V domain prediction is evaluated, layer-specific terms may be
tested:

```text
WM ~ K + C + V
WM ~ K + C + V + W_specific

Reasoning ~ K + C + V
Reasoning ~ K + C + V + R_specific
```

`W_specific` and `R_specific` require independent indicators. The outcome
itself cannot be reused to define the layer-specific factor.

## Family Structure Safeguard

HCP-YA contains twins and siblings, so ordinary random participant folds are
not sufficient when family structure is present.

Preferred validation:

```text
family-isolated cross-validation
```

Acceptable first analysis without restricted family data:

```text
unrelated-only cohort
```

Ordinary participant-isolated folds are not allowed for the primary HCP-YA
test unless an explicit protocol amendment justifies them.

## Current Implementation

Machine-readable config:

```text
config/hcp_ya_transversal_v1.yaml
```

Runner:

```text
python -m trident_validation.mechanistic.hcp_transversal \
  --config config/hcp_ya_transversal_v1.yaml \
  --preflight-only
```

The runner currently performs data-support preflight only:

- verifies source availability and checksum when registered;
- checks required predictor/outcome columns;
- enforces anti-circularity between predictors and outcomes;
- checks unrelated/family-isolated validation feasibility;
- records missingness and complete-case support by domain;
- writes participant-free support outputs.

No HCP model fitting is authorised by this protocol until a support preflight
passes and a separate analysis-freeze commit is made.
