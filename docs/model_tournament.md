# Model Tournament

## Candidate Models

- M0: general performance baseline.
- M1: continuous control manifold.
- M2: non-linear vigilance or inverted-U model.
- M3: three-profile mixture.
- M4: four-PACE-profile mixture.
- M5: dynamic Trident state-space model.

## Scoring Requirements

All models must be evaluated with:

- participant-isolated splits;
- dataset-isolated transport where possible;
- calibration checks;
- sensitivity to feature sets and task duration;
- explicit complexity penalties.

## Decision Rule

Prefer the simplest model that gives stable out-of-sample prediction and passes preregistered robustness gates. M5 is supported only if transition, dwell, re-entry and prediction tests improve over M0-M4.

