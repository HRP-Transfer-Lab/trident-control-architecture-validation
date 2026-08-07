# Flow Zone Validation V1

## Contrastive computational model tournament for Trident state, PACE profiles and APC mechanisms

**Status:** implementation-ready research specification  
**Target repository:** `HRP-Transfer-Lab/trident-control-architecture-validation`  
**Scope:** public-data model development → synthetic model recovery → prospective transport to limited Attention Coach data.  
**Claims boundary:** tests candidate behavioural state/control structure. It does not establish brain states, diagnose individuals, validate clinical routing, or directly measure neural criticality.

---

## 1. Scientific target

The next validation cycle should not ask only whether four clusters can be recovered. The stronger question is:

> Which latent architecture best explains individual heterogeneity, within-person state change, persistence, transitions, vigilance coupling and future behaviour across cognitive-control datasets?

The target architecture distinguishes four levels:

```text
READINESS
Is sufficient stable activation available?

TRIDENT STATE
What dynamical landscape is active?
Flat / In the Zone / Spun Out / Locked In

APC CONTROL MODES
Which control operation is succeeding or failing?
Signal / Evidence / Commit
with Predictive Calibration cross-cutting

PACE PROFILE
What joint behavioural control pattern is expressed?
Regulated / Slow compensatory / Fast brittle / Globally overloaded
```

The V1 hierarchy is:

```text
stable person baseline
        +
current readiness
        ↓
latent Trident state trajectory
        ↓
state-gated APC mode parameters
        ↓
observed task behaviour
        ↓
PACE profile expression
```

PACE profiles are not assumed to be identical to Trident states.

---

## 2. Existing evidence V1 must explain

### ACDC Stroop-focused pilot

Current work favours three task-active profiles: efficient regulated, inefficient overloaded and fast brittle. The four-component solution failed its registered component-size gate. The analysis also shows a mixed trait–state structure: stable between-person differences coexist with substantial within-person movement.

### Paired Stroop–Flanker–SART study

The independent paired study favours a four-component full-covariance GMM, provisionally interpretable as slow compensatory, regulated, globally overloaded and fast brittle. SART vigilance is associated with those profiles but is not interchangeable with them. Repeat-session results again support mixed trait–state structure.

These findings create a model-selection problem, not a contradiction to be hidden.

---

## 3. Contrastive methodological extension

The 2026 Biological Psychiatry review *A Contrastive Framework for Modeling Brain Heterogeneity in Precision Mental Health* motivates a shift from modelling dominant average variation to modelling variation enriched in a scientifically defined target relative to a background condition.

For Flow Zone:

> Build the scientific contrast into the representation-learning problem.

V1 adds, in increasing order of complexity:

| Method | Role |
|---|---|
| Normative modelling | Separate expected person/task performance from current deviation |
| Contrastive PCA (cPCA) | Interpretable target-enriched linear dimensions |
| Contrastive variance quotient (CVQ) | Target variance penalised by background/test–retest variance |
| Contrastive VAE (CVAE) | Non-linear separation of shared person/task variation from salient state variation |
| Graph contrastive learning | Later modelling of task-invariant versus state-specific transition topology |

Deep models cannot replace simpler models merely because they fit discovery data better. They must pass participant-isolated and dataset-isolated validation plus synthetic ground-truth recovery.

---

## 4. Formal model tournament

### M0 — General performance

```text
one broad performance factor
+ stable person differences
+ measurement error
```

### M1 — Continuous control manifold

One or two continuous dimensions, initially:

```text
control efficacy × instability/response-policy
```

### M2 — Non-linear vigilance / inverted-U

```text
readiness or mobilisation → nonlinear performance curve
```

### M3 — Three-profile mixture

```text
regulated / overloaded / brittle
```

### M4 — Four-PACE-profile mixture

```text
regulated / slow compensatory / fast brittle / globally overloaded
```

### M5 — Dynamic Trident state-space

The primary Trident model is a dynamic latent space, not a four-cluster GMM.

```text
z_t = [engagement_or_mobilisation_t,
       explore_exploit_tilt_t]

z_(t+1) ~ p(z_(t+1) | z_t,
                        task demand,
                        conflict,
                        error history,
                        vigilance,
                        context)
```

Four theoretical regions are applied only after fitting and validating the latent geometry:

```text
Flat / In the Zone / Spun Out / Locked In
```

Behavioural data are not treated as direct measurements of F, G, F★, neural criticality or the literal cusp potential.

---

## 5. Stage 0 — Freeze existing evidence

Before V1 modelling:

1. Freeze current ACDC and paired-study manifests.
2. Reproduce registered outputs without V1 transformations.
3. Treat current clusters as prior evidence, not training labels.
4. Keep neutral cluster identifiers during discovery.
5. Preserve null, continuous, source-dominated, three-profile and four-profile outcomes as legitimate results.

---

## 6. Stage 1 — Normative trait–state decomposition

For feature `y` in person `i`, session/window `t`:

```text
y_it = expected_person_i
     + expected_task_dataset_context_it
     + state_deviation_it
     + residual_error_it
```

Recommended implementation: robust hierarchical location-scale modelling with task, condition composition, trial count, practice/session order, source dataset and device/timing covariates where available; participant intercepts and slopes where repeated data support them.

Output both:

```text
population-standardised feature
personal-deviation feature
```

The personal-deviation representation becomes the principal input to state-sensitive contrastive analyses.

Required outputs per feature:

- stable person variance;
- session/window variance;
- task/source variance;
- residual variance;
- ICC/reliability where meaningful;
- incremental prediction of next window/session from personal deviation beyond raw population score.

---

## 7. Stage 2 — Latent-architecture tournament

Fit M0–M5 using identical participant-isolated folds and, where feasible, dataset-isolated folds.

Primary outcomes:

```text
held-out log predictive density / ELPD
next-window or next-session prediction
probability calibration
participant-bootstrap stability
whole-dataset transport
source/task predictability
posterior predictive checks
```

Mixture models additionally report BIC/AIC, posterior entropy, component size, bootstrap ARI and held-out component recovery.

Dynamic models additionally report dwell-time distributions, transition matrix stability, transition entropy, re-entry probability, recovery time and next-state prediction.

Required comparisons:

```text
M0 vs M1: does multidimensional control add beyond general performance?
M1 vs M2: is a nonlinear readiness curve sufficient?
M1/M2 vs M3/M4: do profiles improve out-of-sample prediction?
M3 vs M4: is slow compensation reliably separable?
M0-M4 vs M5: does explicit temporal state structure add predictive value?
```

Prefer the simplest model that survives the registered validation gates.

---

## 8. Stage 3 — Contrastive decomposition

All transformations are fitted inside training folds only.

### Contrast A — state deviation versus personal background

```text
TARGET: large within-person deviation windows/sessions
BACKGROUND: same-person baseline-like windows/sessions
```

Question: what multivariate structure is enriched during state excursions once stable individual differences are discounted?

### Contrast B — high versus low control demand

```text
TARGET: high conflict/switch/lure/ambiguity/perturbation
BACKGROUND: matched lower-demand periods
```

Question: what dimensions become specifically important when adaptive control is required?

### Contrast C — transition versus stable occupancy

```text
TARGET: windows around a latent state/profile transition
BACKGROUND: matched windows with stable occupancy
```

Question: what distinguishes reconfiguration from merely being in a high- or low-performing state?

### Contrast D — low versus preserved vigilance

```text
TARGET: low/unstable SART or PVT vigilance
BACKGROUND: preserved vigilance, preferably within person
```

Question: which control dimensions are specifically enriched when readiness is poor and which remain independent of vigilance?

### Contrast E — recovery versus non-recovery

Where perturbation/error/switch/reversal/repeated sessions allow it:

```text
TARGET: successful recovery/re-entry trajectories
BACKGROUND: matched trajectories with persistent impairment/no re-entry
```

Question: what latent structure characterises recoverability rather than static performance?

---

## 9. cPCA and CVQ

### cPCA

For each contrast:

1. fit cPCA inside each training fold;
2. evaluate a pre-specified contrast-strength grid;
3. retain dimensions only if stable across bootstrap resamples and reasonable backgrounds;
4. project untouched participants/datasets using the frozen transformation;
5. test external prediction.

Interpretation is always conditional on the selected background.

### CVQ

Use especially when paired/repeated observations estimate natural variability. Favour dimensions with high target variance relative to background/test–retest variance.

Potential contrasts:

```text
low vs preserved vigilance
switch vs repeat
perturbation vs stable continuation
transition vs stable occupancy
later: intervention vs neutral control
```

A CVQ dimension is target-enriched heterogeneity, not automatically a causal mechanism.

---

## 10. CVAE gate

Run CVAE only if:

1. several hundred independent participants remain after QC;
2. at least one whole dataset can remain untouched;
3. all splits are participant-level;
4. synthetic experiments show acceptable disentanglement/subtype recovery at the effective sample size;
5. source leakage is low;
6. repeated retraining gives reproducible latent solutions.

Intended decomposition:

```text
SHARED LATENT
stable person differences
general speed/capacity
task familiarity
dataset/device structure

SALIENT LATENT
current state displacement
vigilance-related deviation
persistence/rigidity
instability/volatility
recovery dynamics
```

No Trident label is attached without independent behavioural/dynamic validation.

---

## 11. Stage 4 — Dynamic topology

Trident-G is a theory about movement through a landscape, so temporal topology is a primary test.

Candidate models:

```text
hierarchical HMM
hidden semi-Markov model
switching linear dynamical model
continuous latent state-space model for M5
```

Pre-registered directional signatures:

### In-Zone-like

```text
high re-entry probability
context-sensitive switching
moderate dwell time
successful recovery after conflict/error/perturbation
```

### Locked-In-like

```text
high self-transition probability
increased persistence/hysteresis
weak reopening after diagnostic mismatch
reduced transition diversity
```

### Spun-Out-like

```text
high local volatility
high transition entropy
unstable evidence use
poor reconvergence
```

### Flat-like

```text
low mobilisation/vigilance when maladaptive
low update/reconfiguration
persistent routine under changed demand
```

Flat is never inferred solely from slow RT; efficient routine may be adaptive.

### Later graph contrastive extension

After stable regions are recoverable:

```text
nodes = learned latent states/regions
edges = transition probabilities
node attributes = state feature distributions
edge attributes = conflict/switch/error/vigilance/recovery context
```

Graph contrastive learning can then test task-invariant versus context-specific transition topology.

---

## 12. Stage 5 — APC mode-mechanism layer

Once the state architecture is constrained, test how state affects behaviour through APC modes.

Candidate parameter families:

```text
C_signal = representational/evidence quality
A_evidence = accumulation/updating/change-point response
T_commit = threshold/urgency/stopping/deadline adaptation
PC = prediction strength/precision/confidence/source reliability
```

Do not estimate parameters from tasks that cannot identify them.

Compare:

```text
A. general performance/capacity only
B. APC modes only
C. Zone/state only
D. additive state + APC
E. state-gated APC: Zone × Signal/Evidence/Commit/PC
```

The strongest Trident/APC result is Model E outperforming state-only and mode-only models in held-out prediction.

Candidate discriminating predictions:

```text
LOCKED-IN:
adequate signal may coexist with underweighting of disconfirmation
and hysteretic/inflexible commitment.

SPUN-OUT:
unstable signal/evidence, labile updating and erratic/delayed commitment.

SLOW COMPENSATORY:
preserved evidence quality, continued sampling/checking,
threshold too high for delay cost.

FAST BRITTLE:
adequate easy-trial signal, insufficient diagnostic evidence use,
threshold too low under ambiguity/conflict.
```

---

## 13. Public datasets by mechanism

| Dataset/task family | V1 contribution |
|---|---|
| ACDC Stroop/Flanker/Simon | conflict control, state discovery, stable vs dynamic heterogeneity |
| Paired Stroop-Flanker-SART | profile replication, vigilance separation, trait-state decomposition |
| DMCC Stroop/task switching/AX-CPT/Sternberg | proactive/reactive control, switching, context maintenance, WM-control transport |
| COG-BCI PVT/Flanker/N-back/MATB | vigilance, workload, cross-task and cognitive-autonomic bridge |
| Large task-switching datasets | persistence, switch/repeat dynamics, recovery, wrapper-like changes |
| Reversal/change-point/probabilistic-learning datasets | Evidence Control, learning rate, prior precision, change-point adaptation |
| Confidence datasets | Predictive Calibration and confidence–accuracy relations |

Each imported dataset must declare which APC parameters it can and cannot identify.

---

## 14. Stage 6 — Synthetic model-recovery programme

Synthetic experiments are mandatory before interpreting states/mechanisms.

Generate six worlds:

```text
WORLD 0: one general-performance factor
WORLD 1: continuous 1–2D control manifold
WORLD 2: nonlinear inverted-U readiness
WORLD 3: three-profile mixture
WORLD 4: four-PACE-profile mixture
WORLD 5: dynamic continuous Trident-like state space
```

Every world should include realistic stable participant differences, task method variance, practice, fatigue/time-on-task, device noise, missing trials, unequal trial counts, source shifts and within-person fluctuations.

Add mechanism variants with known truth:

```text
weak signal/evidence quality
miscalibrated prior/precision
slow/excessive accumulation
low commitment threshold
high commitment threshold
poor state-transition flexibility
poor perturbation recovery
```

Required recovery outputs:

```text
model-selection confusion matrix
false discovery of states under continuous truth
recovery of true profile number
latent-coordinate recovery
parameter recovery
state-transition recovery
probability calibration
whole-participant classification error
whole-dataset transport
```

The pipeline fails if it reliably invents a four-state solution under continuous or one-factor ground truth.

---

## 15. Synthetic intervention design

After mechanisms survive recovery, simulate manipulations that maximise expected discrimination.

| Manipulation | Primary target |
|---|---|
| signal strength/distractor ratio | signal quality vs threshold compensation |
| source reliability then reversal | prior precision, updating, hysteresis |
| repeated vs genuinely new evidence | accumulation/evidence independence |
| deadline/error-cost/reversibility | commitment threshold/urgency |
| task switch/wrapper perturbation | reconstruction vs surface skill |
| prolonged time-on-task | vigilance/state deterioration |
| error/near-miss perturbation | reopening vs persistence |
| delayed re-check | consolidation/recovery vs priming |

Attention Coach experiments should preferentially use manipulations where surviving models make clearly different quantitative predictions.

---

## 16. Stage 7 — Limited Attention Coach transport test

Do not use a small Attention Coach sample to rediscover the latent architecture.

Correct sequence:

```text
public data
→ constrained model
→ synthetic recovery
→ frozen predictions
→ Attention Coach transport test
```

Freeze before testing:

```text
feature definitions
normative model
latent transformation
model family
parameter priors
contrast definitions
primary predictions
```

The app sample may estimate participant-specific intercepts/baselines and pre-specified random effects but should not redefine the public-data geometry after outcome inspection.

Primary repeated within-person outcomes:

```text
next-block accuracy/throughput
next-block variability
wrapper first-contact dip
trials/time to recovery
recovered asymptote
return-to-base retention
held-out wrapper performance
fresh delayed re-check
```

Use hierarchical partial pooling.

Primary tests:

```text
1. Does current state probability predict the next block?
2. Does personal deviation add beyond population score and baseline capacity?
3. Do APC parameters add beyond state alone?
4. Does a targeted manipulation selectively alter the predicted parameter?
5. Does parameter change predict wrapper recovery?
6. Does the effect survive a fresh delayed probe?
```

---

## 17. Falsification and decision rules

### Prefer a simpler model when

- continuous models match/exceed mixtures/dynamic models on held-out prediction;
- clusters disappear under participant/source control;
- labels are unstable;
- transition structure is indistinguishable from a simple continuous autoregressive process;
- four-region interpretation depends strongly on arbitrary hyperparameters/backgrounds.

### PACE supported, Trident topology not supported

Possible if three/four behavioural profiles replicate and predict outcomes but temporal Trident topology adds no value.

### Stronger support for Trident state geometry requires

1. M5 improves participant- and dataset-isolated prediction over M0–M4.
2. latent regions/trajectories replicate across task families.
3. transition/dwell/re-entry signatures are stable and directionally consistent with preregistered theory.
4. vigilance/readiness is related but not redundant.
5. contrastive dimensions survive reasonable alternative backgrounds.
6. synthetic null worlds do not readily generate the same structure.
7. the frozen model predicts new Attention Coach behaviour.

### Stronger support for state-gated APC requires

```text
state × mode model > state-only and mode-only models
```

on held-out prediction plus selective experimental perturbation of the corresponding parameter.

---

## 18. Provisional preregistration gates

General:

- participant-level train/test separation mandatory;
- at least one whole source dataset held out where feasible;
- preprocessing/imputation/dimensionality reduction fitted inside training folds;
- source/task predictability explicitly reported;
- failed/null gates retained.

Discrete mixture solution should normally require:

```text
smallest component >= 5% of valid independent units
bootstrap median ARI >= 0.60
acceptable posterior entropy
replication of broad centroid ordering
out-of-sample gain over continuous baseline
```

Contrastive dimension retained only if:

```text
stable under participant bootstrap
stable across reasonable contrast strengths
not mainly source/task identity
predictive in untouched participants
preferably replicated in an untouched dataset
```

CVAE interpretation requires synthetic recovery, repeated-training reproducibility, participant-isolated testing, whole-dataset validation and shared-vs-salient nuisance checks.

---

## 19. Required outputs

Tables:

```text
model_tournament_summary.csv
model_tournament_heldout_metrics.csv
trait_state_variance_decomposition.csv
normative_feature_reliability.csv
contrast_registry.csv
cpca_component_stability.csv
cvq_component_stability.csv
state_transition_matrices.csv
state_dwell_reentry_metrics.csv
source_predictability.csv
synthetic_model_recovery.csv
synthetic_parameter_recovery.csv
synthetic_false_zone_discovery.csv
attention_coach_frozen_predictions.csv
attention_coach_transport_metrics.csv
```

Reports:

```text
reports/flow_zone_v1_model_tournament.md
reports/flow_zone_v1_contrastive_results.md
reports/flow_zone_v1_dynamic_topology.md
reports/flow_zone_v1_synthetic_recovery.md
reports/flow_zone_v1_attention_coach_transport.md
```

---

## 20. Proposed implementation files

Do not replace the existing ACDC or paired-study runners.

```text
config/model_tournament_v1.json

scripts/09_build_normative_residuals.py
scripts/10_fit_zone_model_tournament.py
scripts/11_run_contrastive_cpca_cvq.py
scripts/12_fit_dynamic_state_models.py
scripts/13_run_synthetic_model_recovery.py
scripts/14_run_mode_mechanism_models.py
scripts/15_attention_coach_transport.py

tests/test_normative_residuals.py
tests/test_model_tournament_splits.py
tests/test_contrastive_no_leakage.py
tests/test_dynamic_state_recovery.py
tests/test_synthetic_model_recovery.py
```

CVAE and graph-contrastive modules remain off by default until their gates are passed.

---

## 21. Implementation order

### Phase A — now

```text
1. Freeze existing ACDC and paired outputs.
2. Implement normative trait–state residuals.
3. Implement M0–M4 tournament with current datasets.
4. Implement cPCA and CVQ registered contrasts.
5. Add participant- and dataset-isolated scoring.
```

### Phase B

```text
6. Fit M5 dynamic state-space models.
7. Add transition/dwell/re-entry tests.
8. Run synthetic six-world model recovery.
9. Add public datasets needed for Evidence/Commit/PC identification.
```

### Phase C — gated

```text
10. Train CVAE only if sample/recovery gates pass.
11. Add graph contrastive learning only after transition graphs are stable.
12. Freeze surviving model and transport it to Attention Coach.
```

---

## 22. Claim ladder

| Evidence stage | Defensible interpretation |
|---|---|
| Existing mixture replication | Public control-task data contain reproducible behavioural heterogeneity |
| Normative trait–state decomposition | Stable individual differences can be separated from session/window deviations |
| Contrastive dimensions replicate | Specific demand/state conditions expose structure beyond common background variation |
| Dynamic model wins | Temporal state structure adds prediction beyond static profiles/continuous baselines |
| Trident transition predictions hold | Behaviour is consistent with proposed state-space topology |
| APC state-gating wins | Signal/Evidence/Commit mechanisms interact with latent state as predicted |
| Synthetic recovery succeeds | Candidate mechanisms are distinguishable at realistic data quality/sample sizes |
| Frozen model predicts Attention Coach | Public-data model transports prospectively to a new task system |
| Targeted intervention alters predicted parameter | Preliminary mechanism-specific intervention evidence |
| Wrapper/delayed survival | Preliminary portability/consolidation evidence |

---

## 23. Research boundary

V1 must be capable of weakening or rejecting the proposed structure. A scientifically successful outcome may conclude that:

```text
one continuous manifold is sufficient;
three behavioural profiles replicate but four do not;
four PACE profiles are useful but Trident state topology is unsupported;
Trident state topology is continuous rather than categorical;
or
a dynamic Trident-like state space adds genuine predictive value.
```

The preferred result is whichever survives external prediction, synthetic falsification and prospective transport.
