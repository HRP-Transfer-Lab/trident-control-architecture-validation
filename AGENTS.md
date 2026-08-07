# AGENTS.md

## Purpose

This repository is a falsifiable computational validation programme for the Trident control architecture. It compares simpler behavioural explanations against richer Trident/PACE/APC models using public cognitive-control data, synthetic ground-truth experiments, and later prospective Attention Coach transport tests.

Codex should optimise for **scientific validity, provenance, leakage prevention and reproducibility**, not for producing a favourable Trident result.

## Read first

Before changing analysis code, read in this order:

1. `CLAIMS_BOUNDARY.md`
2. `docs/FLOW_ZONE_VALIDATION_V1.md`
3. `docs/DATA_CONTRACT_V1.md`
4. `config/model_tournament.yaml`
5. `config/contrasts.yaml`
6. `config/synthetic_worlds.yaml`
7. `config/upstream_evidence.yaml`

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
12. **Do not infer neural criticality from behavioural dynamics.** Behavioural state-space results are analogues/tests of theory, not direct measurements of F, G, F★, a neural cusp or brain criticality.
13. **Do not modify upstream evidence repositories from this project.** Treat them as pinned evidence sources.
14. **No participant-level data in Git.** Raw, interim and processed participant data, fitted participant-level latent states, and private Attention Coach data stay outside Git.

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
- Keep model-specific code behind a common interface so M0–M5 can be scored on identical folds.
- Do not silently drop failed models, degenerate components or null results from reports.
- Use British English in documentation and report prose.

## Initial implementation order

Do **not** start with M5, HMMs, CVAE or graph models.

### Phase A0 — evidence and contracts

1. Implement upstream evidence registry validation.
2. Implement the canonical window-feature schema.
3. Add local-only data path conventions and hash/provenance helpers.
4. Add participant-isolated and dataset-isolated split utilities with tests.

### Phase A1 — normative trait-state decomposition

5. Implement robust baseline models that separate:
   - source/task/context effects;
   - stable participant effects;
   - session/window deviation.
6. Emit both population-standardised and personal-deviation features.
7. Test whether personal deviation adds next-window/session prediction beyond raw scores.

### Phase A2 — simple model tournament

8. Implement M0–M4 behind a shared fit/predict/score interface.
9. Use the same folds and scoring contract for every model.
10. Generate a neutral model-tournament report.

### Phase A3 — first contrastive analyses

11. Implement cPCA and CVQ only after normative residuals and split safety are tested.
12. Start with registered contrasts that can be constructed cleanly from existing public data.

Only then proceed to dynamic M5 and synthetic six-world model recovery.

## Definition of done for the first Codex milestone

The first milestone is complete when:

- `pytest` passes;
- upstream evidence commits are pinned and validated;
- a canonical synthetic window table passes schema validation;
- participant-grouped folds are reproducible and demonstrably leakage-free;
- a baseline normative residualiser can be fitted on training participants and applied to untouched participants without refitting;
- a run manifest is generated;
- no Trident/PACE interpretive labels are required for the pipeline to run.
