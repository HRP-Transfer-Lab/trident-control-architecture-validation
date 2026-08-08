# Trident Control Architecture Validation — Research Programme V2

**Status:** Canonical research-programme and milestone plan  
**Supersedes:** the forward milestone sequencing in `docs/FLOW_ZONE_VALIDATION_V1.md`, `docs/CODEX_FIRST_STEPS.md`, and the earlier implementation-order section of `AGENTS.md`  
**Preserves:** completed work, frozen upstream evidence, current M0–M4 implementations, the statistical/data contracts, and the current formal static recovery study  
**Primary scientific aim:** use synthetic known-truth experiments and existing cognitive-task databases to constrain the space of plausible behavioural control architectures, then freeze prospective hypotheses and test them against Attention Coach and WM Coach transfer dynamics.

---

# 1. Programme logic

The programme now has three deliberately different evidence stages:

```text
SYNTHETIC DATA
What architectures can our methods actually distinguish
when the truth is known?
        ↓
PUBLIC COGNITIVE DATABASES
Which of those distinguishable architectures
best predicts real human behaviour?
        ↓
ATTENTION / WM COACH
Do the surviving models make correct prospective
predictions about wrapper transfer and intervention response?
```

The role of synthetic data is **model identifiability and falsification**, not demonstration of Trident-G.

The role of public cognitive databases is **model-space constraint and parameter estimation**, not confirmation of a preferred theory.

The role of Attention Coach and WM Coach is **prospective transport and intervention testing**, not rediscovery of the latent architecture in a small sample.

The guiding rule is:

> Synthetic data eliminate architectures we cannot reliably distinguish. Public databases eliminate distinguishable architectures that do not generalise. Wrapper-swap data then discriminate among the remaining mechanisms by requiring them to predict a controlled perturbation they were not fitted to.

---

# 2. Working variable framework

The research programme should begin from a neutral variable architecture rather than from fixed state labels.

```text
PERSON / LAYER CAPACITY
K_i
stable-ish ability, speed, representational capacity
           │
           ▼
VIGILANCE / READINESS
V_it
is sufficient stable activation available?
           │
           ├─────────────────────┐
           ▼                     ▼
APC PARAMETERS              DYNAMIC REGIME R_it
                            compiled / Ψ-like /
C = Signal                  MI-locked /
A = Evidence                entropy-excess
T = Commit                       │
PC = Calibration                 │
           │                     │
           └──────────┬──────────┘
                      ▼
                 BEHAVIOUR Y_it
                      │
                      ▼
                PACE EXPRESSION
 regulated / compensatory / brittle / overloaded

              [WRAPPER SWAP]
                      ↓
               TRANSFER OUTCOME
 dip / recovery / hysteresis /
 held-out / delayed survival
```

## 2.1 Stable person/layer capacity — `K`

This captures relatively enduring differences that should not be mislabelled as temporary control states:

```text
processing speed
baseline accuracy
representational capacity
attention-control capacity
working-memory capacity
stable response variability
layer-specific skill
```

A naturally slower participant is not automatically overloaded. Stable person/layer effects must be separated from within-person deviations before state-sensitive interpretation.

## 2.2 Vigilance/readiness — `V`

Vigilance/readiness asks whether sufficient stable activation is available to run the adaptive control process.

It is **not** assumed to be another PACE profile or identical to Trident engagement/criticality.

The paired Stroop–Flanker–SART evidence motivates treating vigilance as related but non-redundant with task-active control. The next programme should test whether vigilance is:

```text
independent of APC;
additive with APC;
a specific moderator of Signal quality;
a general gate on APC expression;
or nonlinear, with costs at both low and excessive mobilisation.
```

## 2.3 APC parameters

The primary mechanistic dimensions are:

```text
C_signal
Can the relevant representation be recovered under interference?

A_evidence
Can diagnostic evidence be accumulated, weighted and used to update the active model?

T_commit
Can sampling stop and action occur at an appropriate, context-sensitive threshold?

PC_calibration
Are prediction strength, confidence/precision and source reliability calibrated to environmental stability?
```

These are initially continuous parameters. They may contain both stable person biases and temporary session/block deviations.

## 2.4 Dynamic regime — `R`

The dynamic regime should initially be defined more neutrally than a literal cusp.

Candidate regime structure:

```text
COMPILED / SUBCRITICAL
low need for updating
stable Gc execution

Ψ-LIKE ADAPTIVE
intermittent exploration
+
strong task-relevant constraint
+
successful reconvergence

MI-LOCKED
insufficient reopening after diagnostic mismatch
hysteresis / overconstraint

ENTROPY-EXCESS
high updating/search
but weak task-relevance and convergence
```

A literal Trident cusp/bifurcation model remains a serious candidate, but it must compete with simpler continuous and generic dynamic models rather than define the regime in advance.

## 2.5 PACE expression

PACE should initially be treated as a behavioural/control phenotype rather than a proven natural-kind state:

```text
Regulated
Slow compensatory
Fast brittle
Globally overloaded
```

At least three causal architectures must remain open:

```text
A. APC → behaviour → apparent PACE phenotype

B. PACE latent policy → APC configuration → behaviour

C. dynamic regime × APC → PACE expression
```

The programme should determine whether PACE adds independent latent structure or is a useful coarse description of regions in continuous APC parameter space.

## 2.6 Behaviour — `Y`

Observable behaviour includes task-appropriate combinations of:

```text
accuracy
reaction time / response speed
throughput
RT variability
conflict costs
lapses and omissions
error burstiness
serial persistence
post-error behaviour
switch costs
change-point adaptation
confidence/calibration
recovery slope
```

No single task is expected to identify all latent variables.

## 2.7 Transfer outcome — external criterion

Wrapper transfer is **not** used to define the latent architecture.

It is a prospective external criterion:

```text
PRE-SWAP ONLY

K + V + APC + R + PACE probabilities
       │
       │ prediction frozen here
       ▼
=============================
        WRAPPER SWAP
=============================
       ▼
first-contact dip
recovery slope
trials/time to recovery
recovered asymptote
hysteresis / return asymmetry
held-out wrapper performance
delayed survival
```

This separation prevents circularity and makes the app data a much stronger model test.

---

# 3. Synthetic programme: constrain what can be inferred

The synthetic programme should now be organised around **identifiability gates**.

For every theoretical distinction, ask:

> If this distinction genuinely existed, could the available observations and modelling procedure recover it under participant-isolated validation?

Do not interpret real-data evidence for a distinction that fails synthetic recovery.

## Gate 1 — static architecture

Finish the existing M0–M4 / W0–W4 recovery programme unchanged.

Current competing structures:

```text
M0 / W0: one broad general-performance factor
M1 / W1: continuous multidimensional control manifold
M2 / W2: nonlinear vigilance/readiness structure
M3 / W3: three-profile mixture
M4 / W4: four-profile mixture
```

Primary purpose:

```text
Can the adjudicator distinguish one-factor,
continuous multidimensional,
nonlinear continuous,
three-component,
and four-component structure
without inventing discrete profiles under continuous truth?
```

Do not change M0–M4, W0–W4, scoring, seed schedule or registered recovery criteria after inspecting formal results. Any scientifically motivated revision becomes a versioned follow-up with new seeds.

## Gate 2 — trait versus state

Generate known-truth worlds where each APC parameter is alternatively:

```text
mostly person-like;
mixed person + session/block;
mostly block-state;
or pure noise.
```

For parameter θ:

```text
θ_isb = μ + u_i + s_is + b_isb + ε_isb
```

with known variance components.

Required recovery outputs:

```text
between-person variance recovery
session variance recovery
block variance recovery
ICC recovery
participant-level bias recovery
current-block deviation recovery
next-block prediction
```

Run this separately for Signal, Evidence, Commit and Calibration where the observation model provides identifiability.

## Gate 3 — vigilance relationships

Generate competing worlds:

```text
V0
vigilance independent of APC

V1
vigilance adds directly to performance

V2
vigilance specifically degrades Signal

V3
vigilance gates all APC mechanisms

V4
nonlinear readiness:
low and excessively high mobilisation are costly
```

The goal is to discover which of these relationships can be distinguished with realistic task batteries and sample sizes.

## Gate 4 — APC versus PACE

Construct worlds where:

```text
P0
PACE groups are genuine independent categories

P1
PACE groups are regions in continuous
Signal × Evidence × Commit × Calibration space

P2
stable APC person biases + temporary APC deviations
produce transient apparent PACE classes

P3
PACE adds a latent policy factor beyond APC
```

Required questions:

```text
Can continuous APC structure be falsely discretised?
Can true discrete PACE structure be recovered?
Does a PACE layer add predictive density beyond APC?
Can a mixture induced by task design be distinguished from a genuine participant/profile mixture?
```

## Gate 5 — dynamic regime

Only after the preceding gates are working, compare:

```text
R0
no dynamic regime; APC parameters vary continuously

R1
generic autoregressive continuous dynamics

R2
generic HMM / HSMM state dynamics

R3
continuous adaptive-learning corridor

R4
adaptive corridor with two off-critical failure directions:
MI-lock and entropy-excess

R5
literal Trident cusp / bifurcation model
```

The literal cusp is therefore a model candidate, not an assumption.

Required synthetic outputs include:

```text
state/regime recovery
latent-coordinate recovery
dwell-time recovery
transition recovery
re-entry probability
hysteresis recovery
false-state discovery
regime × APC interaction recovery
```

## Gate 6 — exact versus empirical-twin recovery

Every important architecture should eventually be tested at two levels.

### Exact / clean recovery

Generate data directly from a model's own assumptions.

Purpose:

```text
Does the fitting/scoring machinery work mathematically?
```

### Empirical-twin recovery

Estimate realistic nuisance structure from development cognitive datasets, including where possible:

```text
between-person variance
session variance
block variance
feature covariance
autocorrelation
practice
fatigue/time-on-task
RT and accuracy distributions
error clustering
vigilance distributions
task differences
source differences
missingness
usable windows per person
```

Preserve these nuisance properties while imposing different known latent truths.

Purpose:

```text
Can the model-selection machinery still identify the known architecture
when the data have realistic cognitive-task messiness?
```

A structure that works only under exact/clean recovery should not be trusted for real-data interpretation.

---

# 4. Public cognitive databases: constrain the surviving model space

Synthetic experiments establish what is identifiable. Public data determine what is empirically plausible.

The programme should deliberately combine task families because no one task identifies the complete architecture.

| Task/data family | Main model leverage |
|---|---|
| ACDC Stroop | stable vs dynamic control efficacy; brittle/overloaded structure; static model comparison |
| Paired Stroop–Flanker–SART | cross-task PACE structure; vigilance independence/coupling; trait-state decomposition |
| DMCC Stroop / AX-CPT / task switching / Sternberg | proactive/reactive control; context maintenance; demand sensitivity; mechanism separation |
| PVT / SART | vigilance/readiness and time-on-task dynamics |
| N-back / WM databases | representational Signal; lure handling; capacity × load; cross-layer transport |
| Task switching | reopening, persistence, switch/repeat dynamics, recovery |
| Reversal / change-point / probabilistic learning | Evidence updating, disconfirmation, prior precision, hysteresis |
| Deadline / speed–accuracy tasks | Commit threshold, urgency and selective slowing |
| Confidence/calibration datasets | Predictive Calibration, confidence–accuracy and source reliability |

Every imported dataset must declare:

```text
which latent parameters it can identify;
which it cannot identify;
which nuisance variables are available;
which time scales are observable;
which participant/session splits are valid.
```

Do not manufacture APC variables from tasks that cannot identify them.

---

# 5. Trait–state decomposition is a central target

The public-data programme should explicitly test whether APC dimensions contain both stable person biases and temporary deviations.

For each identifiable parameter:

```text
θ_isb
=
population mean
+
stable person bias
+
person × task/layer deviation
+
session state deviation
+
block state deviation
+
error
```

Primary questions:

```text
How much variance is stable between people?
How much varies within a person across sessions?
How much changes from block to block within one session?
Do stable APC biases recur across task families?
Do within-person deviations predict next-block behaviour beyond baseline capacity?
Do current deviations predict wrapper recovery beyond person-level propensity?
```

A useful end-state distinction is:

```text
stable transfer/control propensity
+
layer-specific propensity
+
current readiness/state
+
current APC deviation
```

---

# 6. Zhang–Tang-inspired dynamic measurements

Where sequential-learning data permit, the programme should estimate behavioural analogues of **updates**, not merely entropy of RT.

```text
prediction_t
      ↓
diagnostic evidence_t
      ↓
inferred change in belief, policy,
response tendency or latent parameter
      ↓
UPDATE Δ_t
```

Candidate dynamic features:

```text
update-magnitude distribution
large-update frequency
intervals between large updates
serial dependence
recovery/convergence after large updates
coupling of update magnitude to diagnostic/task-relevant evidence
```

The theoretical comparison is:

```text
Ψ-LIKE ADAPTIVE
intermittent / broad updating
+
updates remain coupled to useful evidence
+
eventual convergence
```

versus:

```text
MI LOCK
important diagnostic mismatch occurs
+
model does not sufficiently reopen
+
hysteresis / premature constraint
```

versus:

```text
ENTROPY EXCESS
many or large updates
+
weak task-relevance coupling
+
poor convergence
```

These are behavioural computational analogues of the Zhang–Tang entropy–mutual-information learning idea. They are **not direct measures of neural criticality**.

---

# 7. Public-data model tournament after synthetic constraint

The real-data tournament should progress hierarchically rather than jumping straight to a full Trident model.

Candidate sequence:

```text
A
capacity/general performance only

B
continuous APC dimensions only

C
PACE profiles only

D
vigilance/readiness + APC

E
dynamic regime only

F
APC + dynamic regime additive

G
APC × dynamic regime interaction

H
stable person APC biases
+
block/session APC deviations
+
independent vigilance
+
dynamic regime

I
H + independent PACE latent layer
```

Prefer the simplest architecture that survives:

```text
participant-isolated held-out prediction
dataset transport
calibration
bootstrap stability
source/task leakage checks
posterior predictive checks
synthetic recovery
```

The programme must allow a result such as:

```text
stable APC biases
+
block-varying APC deviations
+
independent vigilance
+
continuous adaptive-regime dynamics
```

without discrete Trident states or independent PACE categories.

That would remain a scientifically and interventionally useful result.

---

# 8. Contrastive methods after architecture constraint

Contrastive methods are secondary lenses, not substitutes for the architecture tournament.

Use them after the simpler architecture is constrained to ask what variation is enriched in specific phenomena.

Priority contrasts:

```text
state excursion vs same-person baseline
high vs low control demand
transition vs stable occupancy
low vs preserved vigilance
successful recovery vs persistent non-recovery
later: intervention vs neutral control
```

Use cPCA and CVQ before CVAE. Deep contrastive models remain gated by sample size, dataset isolation, synthetic recovery and reproducibility.

The contrastive question is:

> What multivariate variation is specifically enriched during the scientifically important condition, over and above stable person/task/background variation?

---

# 9. Prospective hypothesis freeze for Attention Coach and WM Coach

The app stage should begin only after a model survives synthetic and public-data gates.

Immediately before a wrapper swap, freeze estimates from pre-swap information only:

```text
K    personal/layer capacity
V    vigilance/readiness
C    Signal
A    Evidence
T    Commit
PC   Calibration
R    dynamic-regime probabilities
P    PACE probabilities if independently supported
```

Then predict:

```text
D       first-contact dip
κ       recovery slope
τ       trials/time to recovery
A∞      recovered asymptote
H       hysteresis / return asymmetry
G       held-out generalisation
Rdel    delayed survival
```

Candidate prospective hypotheses include:

| Pre-swap inference | Frozen transfer prediction |
|---|---|
| Adequate vigilance + regulated APC | Proportionate dip, steep recovery, low hysteresis, strong held-out/delayed survival |
| Low vigilance/readiness | Larger or slower recovery under demanding swaps; Signal/readiness support should help |
| Fast-brittle / low Commit threshold | Early post-swap errors with relatively little strategic slowing; possible false early recovery |
| Slow-compensatory / high Commit threshold | Accuracy relatively preserved but excessive latency/trials to criterion |
| MI-locked regime | Preserved source-wrapper performance but disproportionate swap cost, disconfirmation resistance and return asymmetry |
| Entropy-excess regime | Volatile post-swap behaviour, unstable recovery and poor reconvergence |
| Stable person APC bias | Similar transfer signature across multiple swaps/sessions/layers |
| Block-level APC deviation | Current recovery differs predictably from the person's usual transfer pattern |

The app data should not be used to redefine the public-data latent geometry after seeing the outcomes.

---

# 10. Intervention implications

Once prospective prediction works, move to randomised mechanism-targeted micro-interventions.

Candidate mapping:

```text
fast brittle
→ evidence-opening / disconfirmation prompt

slow compensatory
→ sufficiency / commit prompt

MI locked
→ selective entropy-opening / coherence-reversal pulse

entropy excess
→ constrain / stabilise intervention

low vigilance
→ readiness / Signal stabilisation before perturbation
```

Initially randomise rather than automatically route.

Preferred design:

```text
pre-swap mechanism inferred
        ↓
randomise
        │
        ├── matched targeted intervention
        ├── mismatched active intervention
        └── neutral transition
        ↓
wrapper swap
        ↓
recovery trajectory
```

The most informative result is an interaction:

```text
inferred mechanism
×
intervention
→
recovery improvement
```

rather than a simple intervention main effect.

This is the bridge from descriptive taxonomy to personalised transfer optimisation.

---

# 11. Revised milestones

The current formal static recovery remains the hard gate. Do not contaminate it by implementing the new hierarchy inside the current M0–M4 experiment.

| Milestone | Purpose | Primary exit gate |
|---|---|---|
| **M2.6 — Formal static recovery** | Finish current W0–W4 / M0–M4 experiment unchanged | Frozen recovery/confusion matrix, false-discrete rates and stress results; no post-result model retuning |
| **M2.7 — Empirical-twin static recovery** | Parameterise nuisance structure from development cognitive data and test whether static discrimination survives realistic variance/noise | Static recovery remains adequate under empirically realistic nuisance structure, or limitations are explicitly characterised |
| **M3 — Variable Architecture V2** | Freeze `K, V, Signal, Evidence, Commit, PC, regime, PACE, behaviour, transfer`; register competing causal/orderings | Versioned variable dictionary, identifiability map and model family registry |
| **M4 — Mechanistic synthetic tournament** | Test trait/state decomposition, vigilance relationships, APC→PACE vs independent PACE and recoverability | Synthetic recovery demonstrates which mechanistic distinctions are identifiable at realistic N/noise/task coverage |
| **M5 — Public mechanism tournament** | Fit surviving mechanistic architectures across ACDC, paired data, DMCC, WM/change-point/confidence/task-switching datasets | Participant-isolated and dataset-transport evidence narrows the candidate architecture without unacceptable source leakage |
| **M6 — Dynamic-regime tournament** | Compare continuous dynamics, generic states, adaptive Ψ corridor, MI-lock/entropy-excess and literal cusp/bifurcation | Temporal structure adds held-out predictive value and the surviving regime family passes synthetic recovery |
| **M7 — Integrated/contrastive validation** | Integrate trait/state APC + vigilance + dynamics; use cPCA/CVQ for excursions, transitions, vigilance and recovery contrasts | Stable interpretable representation transports across datasets; deep models remain gated |
| **M8 — App hypothesis freeze** | Translate surviving public model into explicit pre-swap quantitative predictions for Attention/WM Coach | Features, priors, model family, parameter mappings and primary transfer predictions frozen before outcome inspection |
| **M9 — Attention/WM transport** | Test frozen predictions prospectively against dip, recovery, hysteresis, held-out and delayed outcomes | Pre-swap model adds prospective prediction beyond baseline capacity and prior task performance |
| **M10 — Personalised intervention experiment** | Randomised targeted vs mismatched/neutral prompts or pulses; estimate mechanism/profile × treatment response | Differential response is prospectively demonstrated strongly enough to justify adaptive routing research |

---

# 12. Milestone dependencies and stop rules

```text
M2.6
  ↓
M2.7
  ↓
M3
  ↓
M4
  ↓
M5
  ↓
M6
  ↓
M7
  ↓
M8
  ↓
M9
  ↓
M10
```

The sequence is intentionally conservative.

Stop or revise when:

```text
synthetic recovery is poor;
false discrete discovery is high;
parameter recovery is inadequate;
a construct is not identifiable from the available task family;
source/task predictability dominates the latent representation;
dataset transport fails;
all candidate models fit poorly;
added hierarchy fails to improve held-out prediction;
app predictions do not transport.
```

A failed gate is an informative result. Do not move to a richer model simply because the richer model is theoretically preferred.

---

# 13. Immediate implementation order

The next implementation sequence is:

```text
1. Complete and freeze M2.6 formal static recovery.
2. Inspect recovery and confusion patterns without modifying the registered experiment.
3. Build M2.7 empirical nuisance-parameter extraction from development public datasets.
4. Run empirical-twin static recovery.
5. Write and freeze M3 variable/identifiability contracts.
6. Only then implement trait-state APC/vigilance synthetic worlds.
7. Add dynamic-regime models after mechanistic identifiability is established.
8. Do not use Attention Coach or WM Coach outcomes to tune the public-data architecture.
```

Do not start M6 dynamic/cusp modelling while M2.6 is incomplete.

---

# 14. Required new research-programme artefacts

As milestones advance, create versioned files under this folder rather than repeatedly rewriting one large specification.

Recommended structure:

```text
research-program/
  RESEARCH_PROGRAM_V2.md
  VARIABLE_ARCHITECTURE_V2.md
  IDENTIFIABILITY_CONTRACT_V1.md
  EMPIRICAL_TWIN_CONTRACT_V1.md
  DYNAMIC_REGIME_TOURNAMENT_V1.md
  APP_HYPOTHESIS_FREEZE_V1.md
  TRANSFER_OUTCOME_CONTRACT_V1.md
```

The present file is the canonical programme-level roadmap. Detailed statistical and implementation contracts should remain in `docs/`, `config/`, `src/`, `tests/`, `reports/` and `manifests/` as appropriate.

---

# 15. Claims boundary

This programme can test whether behavioural data support:

```text
stable and state-like control parameters;
separable readiness/vigilance;
continuous or discrete control phenotypes;
dynamic learning regimes;
state-gated APC effects;
prospective prediction of wrapper recovery;
differential response to targeted interventions.
```

It must not claim from behavioural data alone that it has directly measured:

```text
neural criticality;
F, G or F★;
a literal neural cusp;
FPCN-A/B balance;
clinical diagnoses;
brain-state categories.
```

Zhang–Tang-inspired heavy-tailed behavioural update analyses are behavioural computational analogues unless and until independent neural evidence establishes a stronger mapping.

---

# 16. End-state scientific question

The programme is no longer primarily asking:

> Can we recover four behavioural profiles?

It is asking:

> **What combination of stable capacity, readiness, adaptive-control parameters and dynamic learning regime best explains how people sample evidence, update, commit, recover from perturbation and transfer an invariant across changed wrappers — and can that model prospectively identify which intervention will improve transfer for a particular person in a particular state?**

That is the organising question for the next generation of Trident control-architecture validation.
