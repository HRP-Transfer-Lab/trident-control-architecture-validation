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
```

## Command

```powershell
$env:PYTHONPATH='src'
python -m trident_validation.mechanistic.k_apc_smoke `
  --config config/mechanistic_identifiability_v1.yaml `
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

## Placeholder Scorer

The smoke includes two placeholder engineering scorers:

```text
SCORE0_capacity_only_rank1
SCORE1_static_continuous_apc_rank5
```

These are not frozen M2.8 model definitions and do not implement the later
selection contract. They exist only to prove that truth stripping,
participant-isolated splitting and score-table plumbing work.

The rank-5 placeholder can dominate the rank-1 placeholder even under MECH0
because no frozen complexity/adjudication rule has been registered for this
temporary scorer. This is a scorer-scaffold limitation, not a scientific result
about APC identifiability.

## Local Output Hashes

```text
generation_audit.csv:
sha256:dc0d46a0bd338f453ca4ebfae10811f748c81f90bb16532b5761dc1823f59a56

model_scores.csv:
sha256:c90022333946a4b4934734c17ee12246cbea03bca3f21707d212757764382fe4

split_audit.csv:
sha256:ccf64620ce59de76da9ad0a85f9bbc02d8888e3c5e78b28c7ca1b9bbd00cf143

local smoke report:
sha256:4da6039faf219be1186f8c3370019bfc9aa4936079815e4154c6222ef33dad3b
```

## Interpretation Boundary

This smoke can be interpreted only as:

```text
the first M2.8 generator/split/truth-stripping/audit pathway works for
K_only_vs_APC.
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

The next pre-outcome step is to replace the placeholder rank scorer with a
registered first-gate M2.8 scoring contract that gives `K`-only and static APC
models a fair complexity-aware comparison before any larger M2.8 pilot.
