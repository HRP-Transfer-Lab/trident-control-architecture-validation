# M2.8 K/APC Generator Smoke

**Status:** engineering generator smoke only

This smoke exercises the first M2.8 identifiability gate:

```text
K_only_vs_APC
MECH0: K -> Y_behavior
MECH1: K + C_signal + A_evidence + T_commit + PC_calibration -> Y_behavior
```

No confirmatory Trident-G, APC, PACE, neural-criticality, cusp or transfer
claim is made.

## Inputs

```text
M3 variable registry commit:
4073c76357954a386b30d2cad792e4c6bb4eb393

M2.8 scaffold commit:
68241545e5328464071218ede098dc4b7b6e16d1

M2.8 schedule hash:
sha256:95832283bed3c899ea743a9657ad567613662b5cf918fc13339fa89da9705808

Config:
config/mechanistic_identifiability_v1.yaml

Config hash:
sha256:f3f482c4c4ccaf83b5dccb6947844475930f518c3339d3e84f0c87ad13346c45

Registered K/APC scoring config:
config/mechanistic_k_apc_scoring_v1.yaml

Registered K/APC scoring config hash:
sha256:20c37197e586b9252c1ce3776a86d2e8e1e25656232da17af74524a55ac1b821
```

## Command

```powershell
$env:PYTHONPATH='src'
python -m trident_validation.mechanistic.k_apc_smoke `
  --config config/mechanistic_identifiability_v1.yaml `
  --scoring-config config/mechanistic_k_apc_scoring_v1.yaml `
  --output-dir reports/generated/m2_8_k_apc_smoke
```

## Engineering Result

```text
units: 4
generated rows: 6144
participants/unit: 48
sessions/participant: 2
blocks/session: 16
participant-isolated splits: yes
participant overlap: 0
truth columns in scoring frame: 0
real transfer outcomes inspected: false
model winner interpretation allowed: false
```

The smoke wrote local generated artefacts under:

```text
reports/generated/m2_8_k_apc_smoke
```

That directory is ignored by Git and contains generated windows, audits, model
scores, a local report and a manifest.

## Variable Audit

Mean realised truth variances across the two smoke replicates per family:

| Family | Variable | Mean variance | Mean observed truth rows |
|---|---|---:|---:|
| MECH0 | K | 0.884 | 1536 |
| MECH0 | C_signal | NA | 0 |
| MECH0 | A_evidence | NA | 0 |
| MECH0 | T_commit | NA | 0 |
| MECH0 | PC_calibration | NA | 0 |
| MECH1 | K | 0.804 | 1536 |
| MECH1 | C_signal | 1.040 | 1536 |
| MECH1 | A_evidence | 1.266 | 1536 |
| MECH1 | T_commit | 0.869 | 1536 |
| MECH1 | PC_calibration | 1.131 | 1536 |

This confirms the first-gate generator distinguishes capacity-only truth from
capacity-plus-APC truth at the latent-variable level.

## Registered First-Gate Scorer

The smoke now uses a registered first-gate engineering scoring contract:

```text
config/mechanistic_k_apc_scoring_v1.yaml
```

It compares:

```text
SCORE0_capacity_only_rank1
SCORE1_static_continuous_apc_rank5
```

The scorer estimates train-standardised low-rank Gaussian covariance models and
selects by:

```text
complexity_adjusted_heldout_log_density_mean_per_row
```

using a BIC-style parameter penalty per held-out row and a pre-specified
practical-equivalence margin with lower-tier tie break. This is still an
engineering smoke scorer, not the full M2.8 model tournament.

Smoke selection check:

| Truth family | Selected first-gate scorer |
|---|---:|
| MECH0 | 2/2 SCORE0_capacity_only_rank1 |
| MECH1 | 2/2 SCORE1_static_continuous_apc_rank5 |

This is a bounded engineering check that the registered scorer can separate
the two clean known-truth first-gate smoke cases. It is not a pilot recovery
rate and does not authorise real-data interpretation.

## Local Output Hashes

```text
generation_audit.csv:
sha256:dc0d46a0bd338f453ca4ebfae10811f748c81f90bb16532b5761dc1823f59a56

model_scores.csv:
sha256:32365d04469c5e76b42ca2e4811d4c288df93b568b742c443c10fb1e78f78f45

split_audit.csv:
sha256:ccf64620ce59de76da9ad0a85f9bbc02d8888e3c5e78b28c7ca1b9bbd00cf143

local smoke report:
sha256:78c2924b09bf019e7a9ad79afa9fdface4c644166b8fdbd0beb1d43070f8f3d1
```

## Interpretation Boundary

This smoke can be interpreted only as:

```text
the first M2.8 generator/split/truth-stripping/audit pathway works for
K_only_vs_APC, and the registered first-gate scorer can be run without truth
leakage.
```

It cannot be interpreted as:

```text
APC exists in real data;
K and APC are empirically distinguishable;
Trident-G is supported;
PACE is a natural latent ontology;
dynamic regimes are supported;
wrapper transfer is predictable;
a cusp or neural criticality is present.
```

## Next Step

The next pre-outcome step is a real-data readiness preflight: map available
public datasets to the M3 variable registry, declare which variables each task
family can and cannot support, and freeze the first public mechanism-analysis
protocol before fitting any real-data mechanism model.
