# Trident Control Architecture Validation

Public methodology and implementation scaffold for testing the Trident control-state architecture against simpler behavioural alternatives.

This repository supersedes the earlier Flow Zone validation plan by making the model-selection problem explicit. The target is not to assume that Trident states, PACE profiles or APC control modes are already validated, but to test whether they add predictive and recoverable structure beyond simpler explanations.

## Scope

The V1 programme evaluates:

- normative trait-state decomposition;
- contrastive representation learning;
- static and dynamic model tournaments;
- synthetic model-recovery tests;
- state-gated APC mechanism models;
- prospective transport to limited Attention Coach data.

The primary specification is [docs/FLOW_ZONE_VALIDATION_V1.md](docs/FLOW_ZONE_VALIDATION_V1.md).

## Repository Layout

```text
config/      Registered dataset, model, contrast and synthetic-world config.
docs/        Methodology notes and implementation-facing design documents.
src/         Package areas for future implementation modules.
studies/     Study-specific protocols, outputs and provenance.
synthetic/   Synthetic worlds, interventions and recovery tests.
reports/     Frozen reports produced by registered runs.
tests/       Unit and regression tests for implementation modules.
manifests/   Run manifests, hashes and analysis status records.
```

## Current Status

This is an implementation-ready research specification. Code modules are intentionally scaffolded but not filled in until the Stage 0 evidence freeze and analysis contracts are confirmed.

## Claims Boundary

This repository tests candidate behavioural state/control structure. It does not establish brain states, diagnose individuals, validate clinical routing, or directly measure neural criticality. See [CLAIMS_BOUNDARY.md](CLAIMS_BOUNDARY.md).

