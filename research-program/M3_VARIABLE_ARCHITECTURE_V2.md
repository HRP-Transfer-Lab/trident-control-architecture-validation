# M3 Variable Architecture V2

**Status:** versioned variable and identifiability contract for M4 design

M3 freezes the Trident/HRP Stack variables that later mechanistic synthetic
work may generate, estimate or hold out. This is not an empirical validation of
Trident-G.

No confirmatory Trident-G, APC, PACE, neural-criticality or cusp claim is
authorised by this document.

## Machine-Readable Registry

The canonical machine-readable registry is:

```text
config/hrp_stack_variable_registry_v2.yaml
```

Implementation code must consume this registry rather than retyping variable
names or causal-family identifiers.

## Variables

```text
K
stable capacity / baseline ability

V
vigilance / readiness

C_signal
Signal quality / representational recovery

A_evidence
Evidence accumulation and updating

T_commit
Commit threshold / decision timing

PC_calibration
prediction, confidence or source-reliability calibration

R_dynamic
temporary dynamic regime

P_pace
PACE phenotype or independently supported latent policy

Y_behavior
observable task behaviour

Transfer_external
external wrapper-transfer criterion for later prospective tests
```

## Core Rules

```text
Transfer outcomes must not define latent state.
PACE must not be forced into four profiles.
Vigilance must not be identified with PACE or Trident state.
A literal cusp is not a required mechanism.
Neural criticality is not inferred from behavioural data.
Wrapper-transfer predictions must be frozen before real transfer outcomes are
inspected.
```

## Causal Families For Synthetic Testing

M4 should implement known-truth generators corresponding to the registered
causal families:

```text
MECH0 general capacity only
MECH1 static continuous APC
MECH2 vigilance-gated APC
MECH3 profile-causes-control
MECH4 continuous APC causes apparent PACE
MECH5 dynamic regime plus APC
MECH6 adaptive-corridor dynamics
MECH7 lock/excess-update failure modes
```

The purpose is not to make a preferred family win. The purpose is to determine
which causal organisations are distinguishable at realistic sample sizes,
timescales and task coverage.

## Identifiability Gates

The minimum M4 known-truth gates are:

```text
K-only vs APC
APC vs vigilance-gated APC
profile-causes-control vs continuous APC causing apparent PACE
static APC vs dynamic regime
generic dynamics vs adaptive corridor
adaptive corridor vs lock/excess-update failure modes
```

If a distinction fails under known truth, it should not be interpreted strongly
in public cognitive data or Attention/WM Coach transfer data.

## Relationship To M2.7

M2.7 found that empirically realistic nuisance can make flexible
continuous/nonlinear static models attractive, while static M3/M4 profile
recovery is weak in the exploratory pilot. M3 carries this forward by treating
continuous and dynamic variables as the primary substrate for mechanistic
identifiability, while leaving PACE/profile structure as a hypothesis that must
earn its place.

## Relationship To M2.8

`M2_8_MECHANISTIC_SYNTHETIC_IDENTIFIABILITY_PROTOCOL.md` describes the next
known-truth synthetic work that will consume this registry.

M2.8 may use synthetic transfer criteria as held-out outcomes for
identifiability checks, but real wrapper-transfer outcomes remain external and
uninspected until a later prospective prediction freeze.

