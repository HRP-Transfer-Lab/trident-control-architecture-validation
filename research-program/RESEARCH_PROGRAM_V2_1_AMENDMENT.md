# Trident Control Architecture Validation — Research Programme V2.1 Amendment

**Status:** Canonical post-M2.6 amendment to milestone sequencing  
**Supersedes:** the M2.6→M2.7 transition and immediate implementation order in `RESEARCH_PROGRAM_V2.md`  
**Preserves:** the frozen M2.6 formal result, all M0–M4/W0–W4 definitions, seed schedules, scoring rules, and claims boundaries  
**Reason for amendment:** the completed M2.6 formal recovery experiment produced a mixed recovery pattern that requires a small exact-model diagnostic before empirical-twin recovery.

---

# 1. Frozen M2.6 result

M2.6 is a constraining result rather than a clean pass.

Baseline aligned recovery:

| True world | Aligned model | Recovery |
|---|---|---:|
| W0 general performance | M0 probabilistic general performance | 0.000 |
| W1 continuous manifold | M1 continuous control manifold | 0.960 |
| W2 nonlinear vigilance | M2 nonlinear vigilance | 0.510 |
| W3 three-profile mixture | M3 three-profile mixture | 0.005 |
| W4 four-PACE mixture | M4 four-PACE mixture | 0.025 |

Pooled false-discrete selection under W0–W2 was low:

```text
7 / 600 = 0.0117
95% CI [0.0057, 0.0239]
```

The current adjudicator therefore:

```text
strongly recognises the continuous M1/W1 structure;
shows moderate/ambiguous sensitivity to nonlinear M2/W2 structure;
rarely invents discrete profile structure under non-discrete truth;
does not currently recover M3/W3 or M4/W4 as aligned latent architectures;
has an unresolved M0/W0 failure that must be diagnosed before adding empirical nuisance complexity.
```

PACE categories therefore remain **phenotypic/descriptive** at this stage. Static M2.6 does not justify treating three- or four-profile PACE structure as independently identifiable latent classes.

Do not retune M0–M4 or W0–W4 and rerun M2.6 under the same version after seeing these results.

---

# 2. Why M2.6b is required

The M2.6 failures may arise from different causes that have different scientific implications:

```text
A. IDENTIFIABILITY FAILURE
The observed features/sample structure do not contain enough information
for the aligned model to distinguish itself.

B. MODEL MISSPECIFICATION
The fitted candidate family is too restrictive relative to the conceptual world.

C. GENERATOR–MODEL MISMATCH
A conceptual W-world is not an exact draw from the corresponding M-model likelihood.

D. SCORING / SELECTION FAILURE
The held-out scoring or simplicity rule systematically favours another family
even when the aligned model is literally true.

E. STRESS-ONLY FAILURE
Baseline discrimination works, but realistic nuisance conditions destroy it.
```

M2.6b is designed to distinguish A–D before M2.7 introduces realistic human-data nuisance structure.

---

# 3. M2.6b — Exact-model recovery diagnostic

## 3.1 Scientific question

> **When data are generated directly from the assumptions of each fitted model family, can the frozen tournament recover that model using the same held-out scoring and selection machinery?**

This is a diagnostic experiment, not a replacement for M2.6 and not evidence for Trident-G.

## 3.2 Exact worlds

Create versioned exact-model generators:

```text
EW0
exact data from M0 assumptions

EW1
exact data from M1 assumptions

EW2
exact data from M2 assumptions

EW3
exact data from M3 assumptions

EW4
exact data from M4 assumptions
```

The observation schema, participant isolation, scoring contract and neutral labels remain unchanged where possible.

Truth metadata must remain excluded from fitting and selection exactly as in M2.6.

## 3.3 Initial computational profile

Follow the computational budget contract.

```text
SMOKE
1–2 replicates/exact world
prove generator, scoring, manifests and outputs

PILOT
10–20 replicates/exact world
check whether diagnostic conclusions are already obvious

CONFIRMATORY DIAGNOSTIC
30–50 baseline replicates/exact world by default
no stress grid initially
checkpointed/resumable if runtime warrants it
```

Do not run a large stress matrix unless the exact baseline diagnostic creates a specific need for one.

## 3.4 Primary outputs

Required outputs:

```text
exact-world recovery matrix;
aligned recovery rate with uncertainty;
numerical-best versus selected model frequencies;
paired participant/window held-out score differences;
selection-rule override frequency;
fit-failure/degeneracy counts;
manifest and seed schedule.
```

For EW3/EW4 also retain component diagnostics, but component interpretability is secondary to model-family recovery.

---

# 4. M2.6b interpretation rules

## Case 1 — Exact recovery is strong

Example pattern:

```text
EW0→M0 strong
EW1→M1 strong
EW2→M2 strong
EW3→M3 strong
EW4→M4 strong
```

Interpretation:

> The fitting/scoring machinery can recognise the candidate models when their assumptions are literally true. Poor W0/W3/W4 recovery therefore reflects conceptual-world/model mismatch, insufficient observational information, or both rather than a basic adjudicator defect.

Consequences:

- M2.7 may proceed.
- M1/M2 remain the primary static candidates carried into empirical-twin work.
- M3/M4 remain descriptive/exploratory unless later mechanism-rich tasks provide independent class-separating evidence.
- The distinction between continuous APC space and discrete PACE classes becomes an explicit later identifiability question rather than an assumed taxonomy.

## Case 2 — Exact mixture recovery fails

Example:

```text
EW3→M3 poor
EW4→M4 poor
```

Interpretation:

> The current tournament cannot adjudicate discrete-versus-continuous profile structure even when discrete mixture assumptions are exactly true.

Consequences:

- Do not use current static M3/M4 selection to make latent-class claims.
- Diagnose fitting/scoring/selection before any later PACE-class inference.
- M2.7 may still proceed for M1/M2 if their exact recovery is adequate, but mixture conclusions remain gated.

## Case 3 — Exact M0 recovery fails

If EW0 does not recover M0 adequately:

> Treat this as a scoring/model-family diagnostic failure that must be understood before empirical-twin expansion.

Inspect, without retuning to a desired answer:

```text
likelihood normalisation;
held-out density comparability across families;
M0 effective flexibility relative to M1;
selection/tie rule;
feature adaptation and missingness handling;
parameter counting/regularisation where relevant.
```

Any scientific change to M0 or the scoring rule becomes a new version with new seeds; it does not rewrite M2.6.

## Case 4 — M2 remains ambiguous against M1

This is acceptable if exact recovery shows M2 is distinguishable in principle.

M2.7 should then explicitly quantify whether empirically realistic nuisance structure makes nonlinear readiness distinguishable from a simpler continuous manifold.

---

# 5. Revised M2.7 role

M2.7 remains the **empirical-twin static recovery** stage, but it begins only after M2.6b resolves the basic exact-model diagnostic.

Its scientific question is:

> **Do the static distinctions that are identifiable under exact-model conditions survive when known-truth synthetic data are given realistic cognitive-task nuisance structure?**

M2.7 should preserve the known latent truth while estimating nuisance structure from development public data, including where available:

```text
between-person variance;
session/block variance;
feature covariance;
autocorrelation;
practice and fatigue;
RT/accuracy distributions;
error clustering;
vigilance distributions;
task/source effects;
missingness;
unequal usable windows/trial counts.
```

### Primary M2.7 candidates after M2.6

Unless M2.6b changes the diagnostic interpretation:

```text
PRIMARY STATIC CANDIDATES
M1 continuous control manifold
M2 nonlinear vigilance/readiness

NULL / SIMPLE REFERENCE
M0, contingent on resolving EW0 exact recovery

PACE MIXTURES
M3/M4 retained as exploratory/descriptive diagnostics,
not as validated latent-class candidates
```

M2.7 must not be presented as another attempt to rescue W3/W4.

---

# 6. Revised milestone sequence

| Milestone | Purpose | Primary exit gate |
|---|---|---|
| **M2.6 — Formal static recovery** | Frozen W0–W4/M0–M4 known-truth recovery | Completed and frozen; mixed result retained without retuning |
| **M2.6b — Exact-model diagnostic** | Test whether the frozen adjudicator recovers models generated from their own exact assumptions | Distinguish machinery/scoring failure from conceptual-world/model mismatch; resolve M0 before empirical twins |
| **M2.7 — Empirical-twin static recovery** | Add empirically realistic nuisance structure to identifiable static truths | Determine whether M1/M2 and any other exact-recoverable distinctions survive realistic human-data messiness |
| **M3 — Variable Architecture V2** | Freeze `K, V, C, A, T, PC, R, P, Y, Transfer` and task identifiability map | Versioned variable and identifiability contract |
| **M4 — Mechanistic synthetic tournament** | Trait/state APC, vigilance relations, continuous APC vs PACE-layer alternatives | Identify which mechanistic distinctions are recoverable at realistic N/noise/task coverage |
| **M5 — Public mechanism tournament** | Fit only synthetic-surviving mechanism architectures to public cognitive data | Participant-isolated and dataset-transport evidence narrows model space |
| **M6 — Dynamic-regime tournament** | Continuous dynamics → generic regimes → adaptive corridor → MI-lock/entropy-excess → literal cusp | Temporal structure adds predictive value and surviving dynamics pass synthetic recovery |
| **M7 — Integrated/contrastive validation** | Integrate trait/state APC, vigilance and dynamics; add cPCA/CVQ contrasts | Stable interpretable cross-dataset representation |
| **M8 — App hypothesis freeze** | Freeze pre-swap model and quantitative transfer predictions | No post-swap information used in model definition |
| **M9 — Attention/WM transport** | Prospective test of dip/recovery/hysteresis/held-out/delayed predictions | Pre-swap model predicts transfer beyond capacity/prior performance |
| **M10 — Personalised intervention experiment** | Randomised matched/mismatched/neutral intervention test | Differential mechanism × intervention response justifies later routing research |

Dependency chain:

```text
M2.6 FROZEN
    ↓
M2.6b EXACT-MODEL DIAGNOSTIC
    ↓
M2.7 EMPIRICAL-TWIN
    ↓
M3 VARIABLE CONTRACT
    ↓
M4 MECHANISTIC SYNTHETIC
    ↓
M5 PUBLIC MECHANISMS
    ↓
M6 DYNAMIC REGIMES
    ↓
M7 INTEGRATION
    ↓
M8 APP FREEZE
    ↓
M9 APP TRANSPORT
    ↓
M10 INTERVENTION
```

---

# 7. Immediate implementation order

The current hard gate is M2.6b.

```text
1. Treat M2.6 outputs and interpretation as frozen evidence.
2. Do not modify M0–M4/W0–W4 in response to the result.
3. Implement EW0–EW4 exact-model generators as a separate versioned diagnostic.
4. Add exact-world recovery outputs and tests for truth non-leakage and reproducibility.
5. Run smoke.
6. Run a small pilot.
7. If informative, run only the bounded baseline confirmatory diagnostic needed to classify the failure pattern.
8. Freeze the M2.6b interpretation.
9. Only then finalise and run the M2.7 empirical-twin preflight.
```

Do not begin M3–M10 implementation while the M2.6b/M2.7 static-identifiability sequence is unresolved.

---

# 8. Scientific implication retained from M2.6

The current evidence favours a cautious working interpretation:

```text
continuous control structure
+
possible nonlinear readiness structure
+
PACE as a useful behavioural phenotype
```

rather than treating static PACE categories as established latent natural kinds.

This does **not** rule out meaningful discrete or metastable structure later in the programme. If such structure exists, it may emerge more strongly at the dynamic-regime level under perturbation, transition and recovery data than in static behavioural summaries.

The programme should therefore keep the following distinction explicit:

> **PACE may remain an interpretable phenotype of control configuration even if the underlying APC parameter space is continuous; genuinely discrete/metastable structure, if supported, must be established independently by later dynamic evidence.**
