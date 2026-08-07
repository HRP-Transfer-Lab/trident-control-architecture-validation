# Model Architecture

## Conceptual Stack

The V1 architecture separates four levels that are often conflated:

- Readiness: whether stable activation is available.
- Trident state: the active dynamical landscape.
- APC mode: the control operation succeeding or failing.
- PACE profile: the observed behavioural control pattern.

## Proposed Hierarchy

```text
stable person baseline
  + current readiness
  -> latent Trident state trajectory
  -> state-gated APC mode parameters
  -> observed task behaviour
  -> PACE profile expression
```

PACE profiles are treated as observable behavioural expressions, not as synonyms for Trident states.

## Candidate Trident States

- Flat: low usable activation and weak task engagement.
- In the Zone: stable adaptive control with recoverable errors.
- Spun Out: unstable, high-variance control.
- Locked In: over-constrained persistence with reduced flexibility.

## Implementation Boundary

The architecture is a candidate explanatory model. It is evaluated against continuous, static-mixture and non-linear vigilance alternatives before being used for prospective transport.

