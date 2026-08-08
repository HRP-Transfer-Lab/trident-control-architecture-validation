# AGENTS.md

## Purpose

This repository is a falsifiable computational validation programme for the Trident control architecture. It compares simpler behavioural explanations against richer Trident/PACE/APC models using public cognitive-control data, synthetic ground-truth experiments, and later prospective Attention Coach / WM Coach transport tests.

Codex should optimise for **scientific validity, provenance, leakage prevention and reproducibility**, not for producing a favourable Trident result.

## Read first

Before changing analysis code, read in this order:

1. `CLAIMS_BOUNDARY.md`
2. `research-program/RESEARCH_PROGRAM_V2.md`
3. `docs/FLOW_ZONE_VALIDATION_V1.md`
4. `docs/DATA_CONTRACT_V1.md`
5. `docs/STATISTICAL_CONTRACT_V1.md`
6. `config/model_tournament.yaml`
7. `config/datasets.yaml`
8. `config/contrasts.yaml`
9. `config/synthetic_worlds.yaml`
10. `config/upstream_evidence.yaml`

`research-program/RESEARCH_PROGRAM_V2.md` is the **canonical forward research programme and milestone plan**. The V1 documents remain important methodological foundations and historical specifications, but their forward milestone ordering is superseded where it conflicts with Research Programme V2.

## Hard scientific rules

1. **Never force four states or profiles.** Null, continuous, three-profile, four-profile and source-dominated results are all legitimate.
2. **Keep discovery labels neutral.** Do not rename statistical components as Flat, In Zone, Spun Out, Locked In, Regulated, Slow Compensatory, Fast Brittle or Globally Overloaded until the relevant post-hoc validation gate is passed.
3. **Trident state and PACE profile are different levels.** Do not treat the four PACE behavioural profiles as identical to the four Trident dynamical regions.
4. **Vigilance/readiness is not automatically another control profile.** Treat SART/PVT vigilance as a related input/moderator unless model comparison supports otherwise.
5. **No raw score shortcut.** A naturally slow participant is not 'overloaded' by definition. Separate stable person effects from session/window deviations before state-sensitive modelling.
6. **No leakage.** Participant, session and source structure must be respected. All imputation, scaling, feature selection, normative transformations, cPCA/CVQ fitting and hyperparameter selection occur inside training data only.
7. **Participant-isolated validation is mandatory.** Do not randomly split windows/trials from the same participant across train and test.
8. **Dataset-isolated validation is required when feasible.** At least one whole source should remain untouched for external transport tests once enough datasets are available.
9. **No current-outcome leakage into priors.** Predictive-calibration variables must use only information available before the current observation.
10. **Deep models are gated.** Do not implement or interpret CVAE or graph contrastive learning until their registered gates pass.
11. **Synthetic falsification is mandatory.** The pipeline must show that it does not invent discrete Trident-like structure when data are generated from one-factor or continuous worlds.
12. **Do not infer neural criticality from behavioural dynamics.** Behavioural state-space and Zhang–Tang-inspired update results are analogues/tests of theory, not direct measurements of F, G, F★, a neural cusp or brain criticality.
13. **Do not modify upstream evidence repositories from this project.** Treat them as pinned evidence sources.
14. **No participant-level data in Git.** Raw, interim and processed participant data, fitted participant-level latent states, and private Attention Coach / WM Coach data stay outside Git.
15. **Do not hard-code the PACE→APC causal ordering.** APC→behaviour→PACE, PACE→APC→behaviour and regime×APC→PACE are competing architectures unless later evidence constrains them.
16. **Do not hard-code a literal cusp.** A Trident cusp/bifurcation model must compete with continuous and generic dynamic-regime alternatives.
17. **Transfer is an external prospective criterion.** Wrapper dip/recovery outcomes must not be used to define the same latent states that are then claimed to predict them.

## Coding rules

- Target Python 3.12.
- Use a `src/` package layout with package name `trident_validation`.
- Prefer pure functions for feature transforms and model evaluation.
- All randomness must use explicit seeds from config/manifests.
- Every formal run writes a machine-readable manifest containing:
  - Git commit;
  - config hashes;
  - upstream evidence commit(s);
  - package versions;
  - random seeds;
  - input checksums where permitted;
  - split definition;
  - model identifiers;
  - output checksums.
- Use deterministic tests wherever possible.
- Unit tests must cover grouping/leakage before model-quality tests.
- Keep model-specific code behind a common interface so competing models can be scored on identical folds.
- Do not silently drop failed models, degenerate components or null results from reports.
- Use British English in documentation and report prose.

## Canonical variable architecture

Future modelling should keep the following constructs separable unless model comparison supports a simpler or different structure:

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
observed behaviour

Transfer
external wrapper-swap outcome
```

The programme should explicitly test stable person biases, person×task/layer effects, session deviations and block-level deviations rather than collapsing them into one score.

## Revised implementation and milestone order

Do **not** jump from the static tournament directly to M5/cusp modelling.

The canonical sequence is now:

### M2.6 — Formal static recovery

- Finish the registered W0–W4 / M0–M4 formal recovery experiment unchanged.
- Preserve the current scoring contract, seed schedule, split logic and model definitions.
- Freeze recovery/confusion matrices, false-discrete rates, stress results and manifests.
- Do not retune models after seeing formal results.

### M2.7 — Empirical-twin static recovery

- Estimate realistic nuisance structure from development public cognitive datasets.
- Parameterise synthetic worlds with realistic between-person variance, session/block variance, covariance, autocorrelation, practice, fatigue, RT/accuracy distributions, error clustering, vigilance, source effects and missingness where available.
- Re-test static recovery without changing the known latent truth.

### M3 — Variable Architecture V2

- Freeze the working variable dictionary: `K, V, C, A, T, PC, R, P, Y, Transfer`.
- Register which datasets/tasks can and cannot identify each variable.
- Register competing causal/orderings rather than assuming PACE/APC hierarchy.

### M4 — Mechanistic synthetic tournament

- Test trait-versus-state recovery for APC parameters.
- Test competing vigilance relationships.
- Test APC-continuous versus independent PACE-category architectures.
- Establish which mechanistic distinctions are identifiable at realistic sample sizes and noise levels.

### M5 — Public mechanism tournament

- Fit only the mechanistic architectures that survived synthetic recovery.
- Use ACDC and paired control-vigilance as frozen prior evidence/development sources according to their contracts.
- Add mechanism-informative public task families such as DMCC, WM/n-back, task-switching, reversal/change-point, deadline/speed–accuracy and confidence datasets as registered.
- Require participant-isolated validation and whole-dataset transport when feasible.

### M6 — Dynamic-regime tournament

Compare, in increasing richness:

```text
R0 no dynamic regime
R1 continuous autoregressive dynamics
R2 generic HMM / HSMM state dynamics
R3 continuous adaptive-learning corridor
R4 adaptive corridor + MI-lock / entropy-excess failure directions
R5 literal Trident cusp / bifurcation model
```

Do not interpret a literal cusp unless it wins against simpler alternatives and passes synthetic recovery.

### M7 — Integrated / contrastive validation

- Integrate the surviving trait-state APC, vigilance and dynamic-regime structure.
- Add cPCA/CVQ for registered contrasts such as excursion vs baseline, transition vs stable occupancy, low vs preserved vigilance, and recovery vs non-recovery.
- Keep CVAE and graph contrastive learning gated.

### M8 — App hypothesis freeze

Before inspecting post-swap outcomes, freeze:

```text
feature definitions
normative model
model family
parameter priors
pre-swap latent estimates
primary transfer predictions
```

### M9 — Attention / WM transport

Test prospective predictions for:

```text
first-contact dip
recovery slope
trials/time to recovery
recovered asymptote
hysteresis / return asymmetry
held-out wrapper performance
delayed survival
```

App data may estimate pre-specified participant random effects/baselines, but must not redefine the public-data architecture after outcome inspection.

### M10 — Personalised intervention experiment

- Randomise matched targeted, mismatched active and/or neutral interventions where feasible.
- Test whether inferred mechanism × intervention predicts differential recovery improvement.
- Do not implement adaptive routing until prospective differential-response evidence justifies it.

## Immediate hard gate

**Do not start M3–M10 implementation while M2.6 formal static recovery is incomplete.**

The next practical sequence is:

```text
complete M2.6
→ freeze results
→ inspect recovery/confusion patterns
→ build M2.7 empirical nuisance calibration
→ run empirical-twin static recovery
→ then freeze M3 variable/identifiability contracts
```

## Definition of done for the current gate

M2.6 is complete only when:

- the formal synthetic run is reproducible from a clean tree;
- participant-isolated splits are verified;
- no truth columns enter model fitting or selection;
- every registered W0–W4 world is evaluated against every registered M0–M4 model;
- model recovery/confusion is reported with uncertainty;
- false discrete discovery is reported for continuous/non-mixture truths;
- stress-condition recovery is reported;
- full manifests/config hashes/seeds are recorded;
- failures and ambiguous recoveries remain visible in the report;
- no scientific parameter or model definition has been changed in response to the formal result.
