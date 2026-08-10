# M6 HCP-YA Transversal K/C Candidate/V Analysis Protocol V1

**Status:** locked pending passed support preflight; M6.0a source-schema and construct hardened

This is the post-preflight analysis scaffold for the HCP-YA transversal K/C/V
test. It does not authorise model fitting yet.

No Trident-G, APC, PACE ontology, transfer, Predictive Calibration, T_commit,
dynamic-regime, neural-criticality or cusp claim is authorised.

## Target Question

```text
Can the same individual K/C_candidate/V coordinates transport across attention,
working memory and reasoning, while allowing the relative contribution of each
coordinate and genuinely layer-specific residual capacity to change with
representational level?
```

## Required Gate

Before this analysis can be run, the HCP-YA support preflight must pass:

```text
config/hcp_ya_transversal_v1.yaml
status = data_support_passed_ready_to_freeze_analysis
```

The current analysis config remains locked:

```text
config/hcp_ya_transversal_analysis_v1.yaml
model_fitting_enabled: false
```

A later analysis-freeze commit is required before any HCP outcome model is fit.

## Primary Transport Test

For each supported domain `Y`, fit the prospective sequence:

```text
Y ~ K
Y ~ K + C_candidate
Y ~ K + V
Y ~ K + C_candidate + V
Y ~ K + C_candidate + V + C_candidate x V
```

Primary score:

```text
family- or unrelated-isolated held-out predictive log density
```

Ordinary random participant folds are not allowed for HCP-YA primary analysis.

## Anti-Circularity

Outcome columns must not be reused to construct predictors for that same
domain.

Examples:

```text
List Sorting cannot enter K when List Sorting is the WM outcome.
PMAT cannot enter K when PMAT is the reasoning outcome.
Flanker cannot enter K because it defines C_candidate.
Card Sort cannot enter K because it is the held-out attention/control target.
HCP global cognition composites cannot enter K.
```

Outcome columns are also forbidden from overlapping with `C_candidate` or `V`.

## Domain Weight Test

If the primary transport analysis is run, report domain-specific coefficients:

```text
Y_domain = alpha_domain K + beta_domain C_candidate + gamma_domain V + error
```

This tests whether the K/C/V mixture changes by representational level. It is
not a mechanism-validation claim.

## Layer-Specific Residual Tests

Layer-specific tests are gated after primary transport:

```text
WM ~ K + C_candidate + V
WM ~ K + C_candidate + V + W_specific

Reasoning ~ K + C_candidate + V
Reasoning ~ K + C_candidate + V + R_specific
```

`W_specific` and `R_specific` remain blocked until independent indicators are
registered. The target outcome itself cannot define the layer-specific factor.
The optional HCP relational-processing behavioural summary is preflighted as a
candidate independent reasoning/relational indicator if available; it is not
required for the primary support gate.

## Interpretation Boundary

Possible result patterns are descriptive decision labels only:

```text
K_C_V_predict_all_domains
K_C_V_predict_all_domains_with_different_weights
K_only_transports_C_V_attention_local
layer_specific_residuals_remain
no_transversal_support
inconclusive
```

No confirmatory transversal architecture claim should be made from HCP-YA
alone. NKI or another instrument-changing replication is required for a stronger
transport claim.

## Current Command

This scaffold can only produce a plan/status report:

```powershell
$env:PYTHONPATH='src'

python -m trident_validation.mechanistic.hcp_transversal_analysis `
  --config config/hcp_ya_transversal_analysis_v1.yaml `
  --plan-only
```
