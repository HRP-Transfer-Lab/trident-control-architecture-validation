# Trident-G Empirical Computational Test Overview

**Status:** Research-programme overview  
**Purpose:** Explain how the validation programme can provide a strong empirical and computational test of Trident-G while allowing important parts of the theory to be revised or rejected.  
**Read with:** `RESEARCH_PROGRAM_V2.md`, `COMPUTATIONAL_BUDGET_AND_EXECUTION_V1.md`, `docs/FLOW_ZONE_VALIDATION_V1.md`, and `AGENTS.md`.

---

# 1. Why this programme is a strong test of Trident-G

The programme should not ask:

> Can we find behavioural patterns that look like Trident-G?

It should ask:

> Which latent architecture best explains individual differences, within-person change, vigilance coupling, learning dynamics and future transfer behaviour after simpler alternatives have been given a fair chance to win?

This makes the programme useful even when a specific Trident-G claim fails.

The core logic is:

```text
THEORY
Trident-G / APC / PACE candidate structure
        ↓
SYNTHETIC KNOWN-TRUTH TESTS
Which distinctions are statistically identifiable?
        ↓
PUBLIC COGNITIVE DATABASES
Which identifiable architectures generalise to real behaviour?
        ↓
PROSPECTIVE APP PREDICTIONS
Do frozen pre-swap estimates predict wrapper dip and recovery?
        ↓
CAUSAL INTERVENTION TESTS
Does a mechanism-matched intervention improve recovery selectively?
```

A favourable result is not required at every stage. The scientific value comes from progressively constraining the theory.

---

# 2. Computational variables rather than fixed labels

The programme translates broad theoretical constructs into candidate variables that can be compared, decomposed and tested.

```text
K
stable person / layer capacity

V
vigilance / readiness

C
Signal quality / representational recovery

A
Evidence accumulation / updating

T
Commit threshold / decision timing

PC
Predictive Calibration

R
dynamic learning regime

P
PACE phenotype if independently supported

Y
observable behaviour

Transfer
external wrapper-swap outcome
```

The working architecture is:

```text
PERSON / LAYER CAPACITY
K_i
        │
        ▼
VIGILANCE / READINESS
V_it
        │
        ├──────────────────────────┐
        ▼                          ▼
APC PARAMETERS               DYNAMIC REGIME R_it
C = Signal                   compiled / Ψ-like /
A = Evidence                 MI-locked /
T = Commit                   entropy-excess
PC = Calibration                  │
        │                          │
        └────────────┬─────────────┘
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

The causal ordering between APC, PACE and dynamic regime is not assumed in advance. It is a model-comparison question.

---

# 3. Synthetic data provide falsifiability and identifiability

Synthetic experiments are not used to demonstrate Trident-G. Their role is to determine whether the available data and modelling machinery can distinguish candidate truths when the truth is known.

The programme should test whether it can distinguish, where relevant:

```text
one broad performance factor
vs
continuous control dimensions
vs
nonlinear vigilance/readiness
vs
three- or four-profile mixtures
vs
stable + state-varying APC parameters
vs
continuous temporal dynamics
vs
generic hidden regimes
vs
adaptive Ψ-like / off-critical regime structure
vs
literal cusp/bifurcation dynamics
```

Important consequences follow.

If a distinction cannot be recovered under known-truth simulation, it should not be interpreted confidently in real data.

If a simpler continuous model is repeatedly misclassified as a four-profile model, profile claims must be weakened.

If the literal cusp cannot be distinguished from ordinary continuous or hidden-state dynamics, the cusp should not be treated as an empirically established part of Trident-G.

Synthetic failure therefore refines the theory rather than merely blocking analysis.

---

# 4. Public databases constrain what is empirically plausible

Public cognitive datasets then test the subset of architectures that survived synthetic recovery.

Different task families have different inferential roles:

```text
Stroop / Flanker
→ task-active efficacy, response policy, conflict control

SART / PVT
→ vigilance / readiness

WM / n-back
→ representational Signal, lure resistance, capacity × load

reversal / change-point learning
→ Evidence updating, disconfirmation, hysteresis

deadline / speed–accuracy tasks
→ Commit threshold and urgency

confidence tasks
→ Predictive Calibration

task switching / perturbation
→ reopening, persistence, transition and recovery
```

No single dataset is expected to identify the complete architecture.

The programme should instead ask whether the same latent dimensions recur across task families and whether the richer architecture improves participant-isolated and dataset-isolated prediction over simpler models.

---

# 5. Vigilance is already an example of theory being constrained by data

The paired Stroop–Flanker–SART study shows why this approach is useful.

Vigilance and task-active control are related, but not interchangeable. In the current paired results:

```text
session-level association between SART engagement and task-active efficacy ≈ .370
within-person association ≈ .156
```

Every task-active PACE candidate contained both preserved- and low-engagement sessions.

This constrains the theory away from:

```text
vigilance = PACE profile
```

or:

```text
vigilance = Trident state by definition
```

and towards the testable hypothesis:

> Vigilance/readiness gates or moderates the ability to deploy adaptive control, while APC parameters and dynamic-regime variables describe how control unfolds once the system is engaged.

This is the desired pattern for the whole programme: empirical results should progressively sharpen the theoretical architecture.

---

# 6. The Zhang–Tang component can become computationally explicit

A key opportunity is to turn the entropy–mutual-information idea into testable behavioural dynamics rather than using "criticality" as a broad label.

Where sequential-learning data permit, estimate behavioural analogues of updating:

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

Candidate measures include:

```text
update-magnitude distribution
large-update frequency
intervals between large updates
serial dependence
convergence after large updates
coupling of updates to diagnostic/task-relevant evidence
```

The central comparison becomes:

```text
COMPILED / Gc-LIKE
small or thin updating
stable low-uncertainty execution
```

```text
Ψ-LIKE ADAPTIVE
intermittent / broad updating
+
preserved coupling to diagnostic evidence
+
successful reconvergence
```

```text
MI-LOCKED
important mismatch occurs
+
insufficient reopening
+
hysteresis / overconstraint
```

```text
ENTROPY-EXCESS
many or large updates
+
weak task-relevance coupling
+
poor convergence
```

These are behavioural computational analogues of the Zhang–Tang entropy–mutual-information learning principle. They are not direct measurements of neural criticality.

---

# 7. Wrapper swaps provide the decisive prospective test

The strongest Trident-G test may come from Attention Coach and WM Coach because wrapper change creates a controlled perturbation.

The public-data stage should first constrain and freeze the model.

Immediately before a wrapper swap, estimate from pre-swap information only:

```text
K    person / layer capacity
V    vigilance / readiness
C    Signal
A    Evidence
T    Commit
PC   Predictive Calibration
R    dynamic-regime probabilities
P    PACE probabilities if independently supported
```

Then freeze the prediction before observing:

```text
first-contact dip
recovery slope
trials or time to recovery
recovered asymptote
hysteresis / return asymmetry
held-out wrapper performance
delayed survival
```

Candidate prospective predictions include:

```text
Ψ-like + regulated APC
→ proportionate dip
→ fast coherent recovery
→ low hysteresis
→ strong portability
```

```text
MI-locked
→ good familiar-wrapper performance
→ disproportionate wrapper-swap cost
→ persistence of the old solution
→ hysteresis / slow reopening
```

```text
entropy-excess
→ volatile post-swap behaviour
→ repeated or excessive updates
→ weak reconvergence
```

```text
low vigilance / readiness
→ insufficient mobilisation or Signal stability
→ slow or weak reconstruction even without strong MI-lock
```

This is substantially stronger than discovering clusters in the same data used to evaluate transfer.

---

# 8. Intervention tests can provide causal leverage

If the pre-swap model predicts a specific failure mode, the next step is not immediately to route users automatically.

First randomise interventions that should differentially affect the inferred mechanism.

Examples:

```text
fast brittle / premature Commit
→ evidence-opening prompt

slow compensatory / excessive Commit threshold
→ sufficiency / commitment prompt

MI-lock / overconstraint
→ entropy-opening or coherence-reversal intervention

entropy-excess
→ stabilisation / constraint intervention

low vigilance
→ readiness or Signal-stabilisation intervention
```

The key causal test is:

```text
inferred mechanism
×
intervention condition
→
wrapper-recovery improvement
```

A selective mechanism × intervention interaction would provide stronger evidence than a simple average intervention effect.

---

# 9. How the programme can revise Trident-G

The programme should be designed so that specific theoretical commitments can fail independently.

Possible outcomes include:

```text
literal cusp fails
but adaptive/off-critical dynamic regions survive
```

```text
discrete Trident states fail
but continuous dynamic-regime structure survives
```

```text
independent PACE categories fail
but PACE remains a useful phenotype of continuous APC space
```

```text
vigilance is not a Trident state
but strongly moderates access to adaptive control
```

```text
stable APC person biases coexist with strong block-level state deviations
```

```text
wrapper recovery is predicted mainly by APC × current regime
rather than static capacity alone
```

These are all theoretically informative outcomes.

A theory that survives only after losing unsupported pieces is computationally stronger, not weaker.

---

# 10. What computational depth would mean here

If successful, the programme moves Trident-G from a broad conceptual dynamical theory towards an empirical computational model with explicit:

1. **latent variables** — `K, V, C, A, T, PC, R, P`;
2. **timescales** — stable person biases, session states and block-level deviations;
3. **individual-difference parameters** — recurring APC and transfer propensities;
4. **state/regime dynamics** — continuous or discrete, whichever wins model comparison;
5. **observable signatures** — RT, accuracy, variability, evidence updating, confidence and transition measures;
6. **prospective predictions** — pre-swap estimates predicting unseen recovery trajectories;
7. **controlled perturbations** — wrapper swaps as system-identification events;
8. **causal interventions** — mechanism-matched prompts or transition manipulations;
9. **transfer criteria** — held-out wrapper and delayed survival rather than trained-task gain alone;
10. **formal failure conditions** — explicit outcomes under which parts of Trident-G are revised or rejected.

The desired end state is therefore not merely a richer taxonomy.

It is a computational architecture capable of specifying:

```text
what varies between people
what changes within a person
what regime the system is currently in
which control mechanism is limiting adaptation
what should happen under perturbation
and which intervention should selectively improve recovery
```

---

# 11. Claim boundary

This programme tests candidate behavioural and computational analogues of Trident-G.

It does not, by behavioural modelling alone, establish:

```text
direct neural criticality
literal F, G or F★ values
FPCN-A/B identities
brain-level entropy–MI quantities
a literal cusp catastrophe
diagnostic cognitive states
clinical treatment routing
```

Those stronger claims require independent neural, physiological or intervention evidence.

The appropriate claim is:

> The programme tests whether a Trident-G-inspired architecture of readiness, adaptive control, learning regime and transfer provides recoverable and prospectively useful structure beyond simpler behavioural alternatives.

---

# 12. Programme principle

The central principle is:

> **Synthetic data constrain what can be inferred. Public databases constrain what is plausible. Controlled wrapper perturbations test what predicts transfer. Randomised interventions test whether the inferred mechanisms are causally useful.**

If Trident-G survives this sequence, it gains genuine computational and empirical depth. If particular parts fail, the surviving theory should be revised around the structures that remain identifiable, predictive and interventionally useful.
