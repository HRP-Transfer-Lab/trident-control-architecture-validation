# M6 HCP-YA Transversal K/C Candidate/V Protocol V1

**Status:** M6.0a source-schema and construct-hardened data-support preflight protocol

This protocol implements the next empirical strategy after the paired
Stroop/Flanker/SART public analyses. It does not fit HCP models yet.

No Trident-G, APC, PACE ontology, transfer, Predictive Calibration, T_commit,
dynamic-regime, neural-criticality or cusp claim is authorised.

## Question

```text
Are K/C_candidate/V genuinely transversal constraints on cognition, or are they mostly
features of attention/control tasks?
```

The first HCP-YA test separates two questions:

1. Do the same lower-level K/C_candidate/V coordinates predict different levels of the
   stack?
2. Do the weights of K/C_candidate/V change across attention/control, working memory and
   reasoning outcomes?

## Lower-Level Coordinates

The lower-level predictors are:

```text
K = broad non-target performance/general capacity indicators
C_candidate = Flanker inhibitory-control/attention candidate
V = sustained-attention/CPT candidate
```

These remain behavioural predictors, not validated Trident-G/APC variables.
The HCP Flanker score is not called confirmed `C_signal` in this protocol.
Flanker is reserved for `C_candidate` and does not enter K.

Primary K excludes Flanker, Card Sort, List Sorting, 2-back, PMAT and HCP global
cognition composites such as CogTotalComp.

## Held-Out Outcomes

The intended held-out outcome domains are:

```text
attention/control
working memory: List Sorting
reasoning: PMAT
working memory optional/future: aggregate 2-back task performance, known issue pending
reasoning/relational optional/future: relational-processing behavioural summary, if available
```

List Sorting is the required primary working-memory outcome for the current
HCP support gate. The released aggregate `WM_Task_2bk_Acc` is optional and not
analysis-eligible because of an unresolved HCP Users concern about aggregate
WM task accuracy calculations. A 2-back outcome can be added later only through
an official HCP resolution or a prospective reconstruction from task/run/trial
files.

The anti-circularity rule is hard:

```text
an outcome column must not be used to construct K/C_candidate/V for that same outcome
```

List Sorting cannot enter K if List Sorting is the WM outcome. PMAT cannot
enter K if PMAT is the reasoning outcome. The same rule applies to any
attention/control outcome.

## Primary Model Sequence

For each supported domain `Y`, the prospective model sequence is:

```text
Y ~ K
Y ~ K + C_candidate
Y ~ K + V
Y ~ K + C_candidate + V
Y ~ K + C_candidate + V + C_candidate x V
```

The primary score remains participant-isolated or family-isolated held-out
predictive log density.

## Domain Weight Test

If the data-support gate passes, the analysis will report whether coefficients
for K/C_candidate/V differ by domain:

```text
Y_domain = alpha_domain K + beta_domain C_candidate + gamma_domain V + error
```

This is a transversal-weight test. It is not a claim that K/C/V are proven
mechanisms.

## Layer-Specific Residual Tests

Only after K/C/V domain prediction is evaluated, layer-specific terms may be
tested:

```text
WM ~ K + C_candidate + V
WM ~ K + C_candidate + V + W_specific

Reasoning ~ K + C_candidate + V
Reasoning ~ K + C_candidate + V + R_specific
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
- canonicalises registered HCP source columns before support checks;
- checks required predictor/outcome columns;
- enforces anti-circularity between predictors and outcomes;
- checks unrelated/family-isolated validation feasibility;
- records missingness and complete-case support by domain;
- writes participant-free support outputs.

No HCP model fitting is authorised by this protocol until a support preflight
passes and a separate analysis-freeze commit is made.
