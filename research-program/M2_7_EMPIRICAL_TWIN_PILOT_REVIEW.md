# M2.7 Empirical-Twin Pilot Review

**Status:** exploratory pilot review, banked for scientific planning

This document reviews the completed M2.7 empirical-twin pilot. It does not
modify `EMPIRICAL_BACKGROUND_CONTRACT_V1`, `STATIC_MODEL_SELECTION_CONTRACT_V2`,
the ETW worlds, the pilot schedule, model definitions or selection rules.

No confirmatory architecture-recovery claim is made. No Trident-G, APC, PACE,
neural-criticality or cusp-validation claim is made.

## Frozen Inputs

```text
Pilot reporting and schedule commit:
eaeaeecd5b1f2955fe9167840a27eeb82b780e89

M2.7 pilot protocol commit:
45255968599d444d3f5c63e9739f7beb451274cd

Audit correction commit:
617549288c4ef44ae216473fcc822f4ad6856d3a

Frozen empirical-background contract:
1e9c11e030dc0706c8941dfc08489bb6f63cac5d

Frozen static V2 model-selection contract:
81d22f2a942afd1652c169935f058b1634922d30

Frozen scientific generator SHA recorded in schedule:
9699427176961a3064e9db8da23a3b287a441b1b

Pilot schedule hash:
sha256:4e90e0ea6f368f2430a5138dd69bcd2c03b34ad8bfad29e182a53df436099b3a
```

The schedule records `9699427...` as the frozen scientific generator baseline.
The manifest records the later validation-repository commit because pilot
runner, schedule and reporting support were added afterward. This distinction is
intentional provenance, not a reason to rewrite the generator identity.

## Output Provenance

```text
pilot_manifest.json:
reports/generated/empirical_twin_v1_pilot/pilot_manifest.json

pilot report:
reports/generated/empirical_twin_v1_pilot/M2_7_EMPIRICAL_TWIN_PILOT_REPORT.md

pilot report checksum:
sha256:3c32ac4a33e24798990a16a3b2bdb8edbb2b8cbf4e6a82471e835435c9e8bc21

model_scores.csv:
sha256:e5530a8db2d04f019e2cc03aba8fd75cd72ea928a68996e9d89b52b1c61c4c41

selection_summary.csv:
sha256:47d6448f7983cd187e6ea52f7ecdd0f2e4e330cdd2846bdb56dc2c5873b02f01

world_model_selection_summary.csv:
sha256:7ddace21297531fd8cb3c3e8a655dbdb965e6bef5149e24b644c55a647c3f030

false_discrete_pilot_summary.csv:
sha256:67458aee86427b69ee22d32eb4b47f9962de2f155675548a9af8309a8e473745

paired_contrasts.csv:
sha256:2b6a748c85e6198ce06da22207eb27ad6c0944f6fab4c9f741c712733cc17fdc

fit_diagnostics.csv:
sha256:8a6f4b013cb543b5d09759b108ba4450bf71f11d451cd7282d2a20409b4a6de4

background_realism_summary.csv:
sha256:bfddd5bff87afd834fe68c5f07472db71b8999da1141322e33a71f5e71f71ef5
```

## Engineering Result

```text
units planned: 50
units complete: 50
units failed: 0
formal_claims_allowed: false
model_recovery_claim_allowed: false
runtime_seconds: 881.288
```

All model fits completed. The pilot therefore passed the engineering gate for
checkpointed, participant-isolated empirical-twin execution.

## Exploratory Findings

These findings are pilot-scale only (`n = 10` replicates/world). Wilson
intervals in the output tables are descriptive and imprecise.

### ETW0

ETW0 is the broad general-performance world.

```text
selected M0:    1/10
selected M1:    1/10
selected M2_EM: 7/10
selected M4:    1/10
same-tier ambiguity: 8/10

M1 - M0 participant-isolated held-out density:
mean   0.811
median 0.682

M1 second/first eigenvalue ratio:
mean   0.359
median 0.328

M1 second loading fraction:
mean   0.351
median 0.343
```

Interpretation: realistic empirical background can make structured continuous
or nonlinear candidates attractive even under a general-performance known
truth. This is a specificity warning. A small or moderate M1 density advantage
must not be interpreted by itself as substantive multidimensionality.

### ETW1

ETW1 is the continuous manifold world.

```text
selected M0:    2/10
selected M1:    1/10
selected M2_EM: 7/10
same-tier ambiguity: 7/10

M1 - M0 held-out density:
mean   0.628
median 0.502

M1 - M2_EM held-out density:
mean  -0.184
median -0.239
```

Interpretation: continuous structure improves over M0, but the pilot does not
cleanly distinguish linear M1 from the more flexible M2_EM candidate. This is a
model-specificity issue, not evidence for Trident-G.

### ETW2

ETW2 is the M2_EM_v1-aligned nonlinear continuous world.

```text
selected M2_EM: 9/10
same-tier ambiguity: 3/10

M2_EM - M1 participant-isolated held-out density:
mean   0.784
median 0.658
```

Interpretation: the M2_EM-aligned nonlinear truth remains distinguishable in
this pilot. This supports using nonlinear continuous structure as a serious
candidate substrate in later mechanistic work.

Numerical diagnostic:

```text
M2_EM iterations:       120/120 in all ETW2 replicates
strict convergence:     0/10
iteration-cap flag:    10/10
mean final likelihood change: 0.00771
mean final parameter change:  0.000128
mean M2_EM runtime: 23.626 s
```

Under the frozen contract, strict optimiser convergence is not newly required
for valid scoring. Nevertheless, the iteration-cap pattern should be carried
forward as a numerical diagnostic before any larger confirmatory design.

### ETW3 and ETW4

ETW3 and ETW4 are known discrete-mixture worlds.

```text
ETW3 selected M3: 0/10
ETW4 selected M4: 1/10

ETW3 M3 - M1:
mean -0.344
median -0.427

ETW3 M3 - M2_EM:
mean -0.663
median -0.731

ETW4 M4 - M3:
mean 0.161
median 0.099

ETW4 M4 - M1:
mean -0.060
median -0.188

ETW4 M4 - M2_EM:
mean -0.185
median -0.350
```

Interpretation: static discrete profile recovery is weak under the frozen
empirical-background pilot. M3/M4 should not be treated as secure detectors of
profile truth in later work.

### False-Discrete Diagnostic

For continuous ETW0-ETW2 worlds:

```text
ETW0 selected M3/M4: 1/10
ETW1 selected M3/M4: 0/10
ETW2 selected M3/M4: 0/10
combined: 1/30 = 0.033
```

Interpretation: realistic background did not mainly produce spurious discrete
profile selection. The larger issue is not false discrete inflation, but the
ability of flexible continuous/nonlinear models to explain both null and
continuous worlds.

## Background Realism

The generated background audit remained broadly consistent with the calibrated
realism checks:

```text
source/task shifts: exact by construction
lag-1: small median discrepancies
missingness: near target at very low empirical rates
window and session variance: finite and sensible
trial counts: summary-quantile approximation, not exact distributional realism
repeated-person stability: diagnostic mismatch remains informative, not tuned
```

These diagnostics must not be used to retune empirical background because a
preferred model performs badly.

## Banked Scientific Constraint

M2.7 should now be banked as a modelling-substrate constraint:

```text
1. Empirical realism can make static continuous/nonlinear structure attractive.
2. ETW2/M2_EM_v1 is recoverable enough to remain a serious nonlinear candidate.
3. M1 vs M2_EM specificity is unresolved.
4. Static M3/M4 profiles are not robust profile detectors under this pilot.
5. False discrete selection under ETW0-ETW2 is low in this pilot.
6. The next programme should not keep asking only which static latent model wins.
```

This does not prove Trident-G. It constrains the next stage toward mechanistic
known-truth identifiability and then prospective transfer prediction.

## Allowed Follow-Up Before M2.8

Only a tightly bounded numerical check is warranted:

```text
Question:
Can M2_EM_v1 reduce iteration-cap frequency or report better convergence
diagnostics without changing model definitions, selection rules or M2.7
scientific conclusions?

Permitted:
implementation-level diagnostics
runtime/convergence profiling
clear reporting of cap-hit behaviour

Not permitted:
retuning M2.7 worlds
retuning empirical background
changing model definitions based on pilot winners
rerunning larger static recovery to obtain a preferred story
```

## Next Milestone

The next prospective milestone is:

```text
M2.8 mechanistic synthetic identifiability
```

Its question is not "which static latent model wins?" but:

> Can mechanistic transfer and adaptation variables be recovered and
> distinguished from simpler alternatives when the truth is known?

