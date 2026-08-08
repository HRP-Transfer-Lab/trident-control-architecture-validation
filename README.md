# Trident Control Architecture Validation

Public methodology and implementation repository for testing the Trident control architecture against simpler behavioural alternatives.

The target is not to assume that Trident states, PACE profiles or APC control modes are already validated. The programme uses synthetic known-truth experiments, existing cognitive-task databases and later prospective Attention Coach / WM Coach transfer tests to constrain which latent architectures are recoverable, predictive and useful for intervention design.

## Canonical research programme

The current programme-level roadmap is:

- [`research-program/RESEARCH_PROGRAM_V2.md`](research-program/RESEARCH_PROGRAM_V2.md)

This **supersedes the forward milestone sequencing** in the original V1 planning documents while preserving completed work, frozen evidence, data/statistical contracts and the current M0–M4 static recovery programme.

The detailed V1 scientific specification remains available at:

- [`docs/FLOW_ZONE_VALIDATION_V1.md`](docs/FLOW_ZONE_VALIDATION_V1.md)

It should now be read as the detailed methodological foundation and historical V1 specification rather than the canonical forward milestone plan.

## Research logic

```text
SYNTHETIC KNOWN-TRUTH DATA
What architectures can the methods distinguish?
        ↓
PUBLIC COGNITIVE DATABASES
Which distinguishable architectures best predict real behaviour?
        ↓
ATTENTION / WM COACH
Do the surviving models prospectively predict wrapper transfer
and differential response to intervention?
```

The current working variable architecture separates:

```text
stable person/layer capacity
+
vigilance/readiness
+
APC parameters:
Signal / Evidence / Commit / Predictive Calibration
+
dynamic learning regime
→
observed behaviour / PACE expression
→
prospective wrapper-transfer outcomes
```

PACE profiles are not assumed to be identical to Trident dynamical regimes, vigilance is not treated as another PACE profile, and a literal Trident cusp remains a candidate model rather than an imposed geometry.

## Revised milestone sequence

The forward programme is:

```text
M2.6  Formal static recovery
M2.7  Empirical-twin static recovery
M3    Variable Architecture V2
M4    Mechanistic synthetic tournament
M5    Public mechanism tournament
M6    Dynamic-regime tournament
M7    Integrated / contrastive validation
M8    App hypothesis freeze
M9    Attention / WM transport
M10   Personalised intervention experiment
```

The current hard gate is to **finish and freeze the formal M0–M4 / W0–W4 static recovery experiment without contaminating it with later hierarchy or dynamic-model changes**.

## Repository layout

```text
research-program/  Canonical programme roadmap and future programme-level contracts.
config/             Registered dataset, model, contrast and synthetic-world config.
docs/               Statistical, data and implementation contracts; historical V1 specification.
src/                Python package implementation.
studies/            Study-specific protocols, outputs and provenance.
synthetic/          Synthetic worlds, interventions and recovery tests.
reports/            Frozen reports produced by registered runs.
tests/              Unit and regression tests.
manifests/          Run manifests, hashes and analysis status records.
```

## Claims boundary

This repository tests candidate behavioural state/control structure and prospective transfer predictions. It does not establish brain states, diagnose individuals, validate clinical routing or directly measure neural criticality. Behavioural analogues of Trident-G or Zhang–Tang dynamics must remain explicitly labelled as behavioural/computational unless stronger independent neural evidence is available. See [`CLAIMS_BOUNDARY.md`](CLAIMS_BOUNDARY.md).
