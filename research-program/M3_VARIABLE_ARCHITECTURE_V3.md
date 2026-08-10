# M3 Variable Architecture V3

**Status:** prospective working architecture for discriminative testing

V3 preserves the historical V2 registry and adds an ontology-neutral programme
architecture. It was formulated after M6.1 and does not change the frozen M6.1
analysis, protocol or numerical results.

The machine-readable registry is:

```text
config/hrp_stack_variable_registry_v3.yaml
```

## Operating / Capability-State System

This system asks:

```text
what resources or current conditions are available?
```

Variables:

```text
K
broad general-capacity / g-like candidate

C
cognitive-control-specific capability candidate relative to K

V
vigilance/readiness state candidate
```

For backwards compatibility the registry keeps `C_signal` as the observed
variable ID where existing machinery requires it. Historical `C_signal` results
must not be silently renamed as confirmed `C_control`.

The conceptual refinement is that C need not be another fully general ability
equivalent to K. The current hypothesis is that C may contain control-specific
variance beyond K and may be recruited across representational levels in
proportion to control demand.

M6.1 supports only the narrow statement that Flanker-derived C predicted
independent CardSort performance beyond K. It did not demonstrate broad C
transport to ListSort or PMAT.

V may be substantially state/session sensitive and is not required to behave
psychometrically like stable K.

## Strategic System

Strategic control `S` is the evidence-to-action policy system.

```text
A_evidence
evidence selection / weighting / updating policy

T_commit
commitment / stopping policy

PC_calibration
contextual metacontrol/calibration of A and T
```

`PC_calibration` concerns calibration of A and T according to uncertainty,
reliability, volatility, cost, urgency and expected value of further evidence.
It is not merely a third parallel strategy ability.

S/A/T/PC remain untested by HCP M6.1 and must not be inferred from ordinary
accuracy or RT.

## Ontology-Neutral Representational Programme

The representational programme asks:

```text
what organises specific variance after frozen K/C/V influences are accounted for?
```

V3 registers three competing organisational hypotheses:

```text
H_LAYER
representational constraints are primarily level/domain specific

H_OPERATOR
important specific capacities are cross-cutting representational/computational
operators recruited across levels according to task demand

H_HYBRID
both cross-cutting operators and local level-specific constraints exist
```

No one of these is preferred by registry fiat.

Candidate layers:

```text
L_attention
L_WM
L_predictive
L_reasoning
```

Candidate operators:

```text
OP_binding
OP_relational
OP_predictive
```

All are candidates only. Single-task estimation is forbidden.

## M6.2 Ontology-Neutral Freeze

M6.2 should not be described as directly testing whether `L_WM` is real.

The exact M6.2 question is:

```text
After frozen K/C/V influences are accounted for, do two independent WM tasks
share reliable held-out individual-difference variance?
```

If positive, the allowed interpretation is:

```text
WM_shared_specific_candidate
shared within-WM variance beyond K/C/V
```

This is not automatically `L_WM`, a WM layer, a WM bottleneck, relational
capacity or binding capacity. Shared ListSort plus 2-back residual variance
could arise from a WM-level constraint, a shared operator, another shared task
requirement or a mixture.

The planned M6.2 primary test remains bidirectional held-out residual transport:

```text
2Back ~ K + C + V
versus
2Back ~ K + C + V + cross-fitted ListSort residual

ListSort ~ K + C + V
versus
ListSort ~ K + C + V + cross-fitted 2Back residual
```

All residuals must be cross-fitted. The primary metric remains
participant-isolated held-out predictive log density. CardSort remains the
prospectively registered non-WM negative control.

M6.2 must not be fit until a valid independently reconstructed HCP 2-back
behavioural measure passes support for the same official 100 unrelated
subjects. `WM_Task_2bk_Acc` remains ineligible while its registered concern is
unresolved.

## Future Discriminative Design

Later datasets should cross representational level or task domain with operator
demand:

| Domain | Binding | Relational | Predictive |
|---|---|---|---|
| Attention/perception | X | X | optional |
| Working memory | X | X | X |
| Reasoning | X | X | X where feasible |

If layer structure dominates, different WM tasks should share specific residual
variance primarily because they are WM tasks.

If operator structure dominates, a relational WM task may share more specific
variance with relational reasoning than with a binding WM task.

If hybrid, both within-level and same-operator cross-level residual structure
should remain.

This is a later prospective model-comparison programme, not an analysis run.

## Future C-Control Design

C may be a cognitive-control-specific factor relative to K, with deployment
depending on task control demand. A stronger test requires multiple independent
control indicators and held-out tasks varying prospectively in control demand.

Desired later evidence:

```text
multiple control measures share variance after K is removed
and
that specific component increasingly predicts performance as control demand
increases, including at WM/reasoning levels
```

This architecture must not be inferred from Flanker/CardSort alone.

## Gates

```text
single-task residual -> specific capacity: forbidden
two same-domain tasks -> confirmed layer ontology: forbidden
domain-weight difference -> new latent factor: forbidden
K x specific or C x specific bottleneck interaction before independent
specific candidate exists: forbidden
strategy from generic RT/accuracy: forbidden
layer/operator/hybrid choice after inspecting target outcomes: forbidden
transfer outcome used to define predictor: forbidden
```
