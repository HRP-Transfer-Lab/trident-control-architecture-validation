# Codex First Steps

## Goal of the first implementation milestone

Build a safe, testable foundation for the V1 model tournament **without yet fitting the Trident model**.

The first milestone should establish:

1. provenance;
2. canonical data contracts;
3. leakage-safe splitting;
4. normative trait-state decomposition;
5. reproducible baseline model interfaces.

Do not start with HMMs, CVAE, graph contrastive learning or Attention Coach transport.

---

## Step 1 — create the Python package scaffold

Use Python 3.12 and a `src` layout.

Recommended structure:

```text
pyproject.toml
src/
  trident_validation/
    __init__.py
    config.py
    provenance.py
    schema.py
    splits.py
    normative.py
    metrics.py
    models/
      __init__.py
      base.py
      m0_general.py
      m1_continuous.py
      m2_nonlinear.py
      m3_mixture3.py
      m4_mixture4.py
    contrastive/
      __init__.py
    synthetic/
      __init__.py

tests/
  test_schema.py
  test_splits.py
  test_provenance.py
  test_normative.py
```

Use minimal dependencies initially. Suggested first-pass packages:

```text
numpy
pandas
scipy
scikit-learn
statsmodels
pyyaml
pydantic
pyarrow
pytest
```

Do not add PyTorch until the CVAE gate is reached.

---

## Step 2 — implement config and provenance validation

Read:

```text
config/model_tournament.yaml
config/datasets.yaml
config/contrasts.yaml
config/synthetic_worlds.yaml
config/upstream_evidence.yaml
```

Implement:

```python
load_yaml_config(path)
validate_upstream_registry(config)
get_git_commit()
hash_file(path)
write_run_manifest(...)
```

The upstream registry validation should verify that required fields are present and should fail loudly if the local study is run against an unregistered upstream evidence commit.

Do not automatically pull or mutate the upstream repositories.

---

## Step 3 — implement the canonical window schema

Translate `docs/DATA_CONTRACT_V1.md` into a Pydantic or Pandera-like validation layer. Prefer Pydantic plus explicit DataFrame checks unless another validation library is already justified.

Required behaviours:

- reject duplicate `source_dataset + participant_id + session_id + task_id + block_id + window_id` keys;
- require positive valid trial counts;
- require `n_trials_valid <= n_trials_total`;
- preserve structural missingness;
- require source/provenance metadata;
- do not require task-specific features when availability flags are false.

Create a fully synthetic fixture with at least:

```text
3 datasets
40 participants per dataset
2 sessions per participant
2-4 windows per session
```

This fixture is for testing infrastructure only, not for validating the theory.

---

## Step 4 — implement leakage-safe split objects

Create deterministic split utilities with explicit seeds.

Required splitters:

```python
participant_group_kfold(...)
dataset_holdout(...)
participant_train_test_split(...)
```

Tests must prove that:

```text
no participant appears in both train and test;
all windows/sessions for a participant remain together;
dataset holdout removes every row from the held-out source;
re-running with the same seed returns identical splits.
```

This is a hard gate. Do not proceed to model-quality work until these tests pass.

---

## Step 5 — implement Stage 1 normative decomposition

Start simple and transparent.

First implementation target:

```text
feature ~ source_dataset + task_id + session/practice terms + participant random/intercept effect
```

If a full mixed-effects location-scale model is too heavy for the first pass, implement a two-level baseline that preserves the scientific contract:

1. fit source/task/context expectation on training participants only;
2. estimate training-participant baseline offsets where repeated observations exist;
3. for unseen test participants, use the population/hierarchical prior rather than leaking their full history;
4. when evaluating longitudinal personalisation, use only prior sessions to estimate an individual's personal baseline.

Return columns such as:

```text
feature_raw
feature_expected
feature_population_z
feature_deviation
feature_uncertainty
```

Important: if a test participant has no prior sessions, `feature_deviation` must not use future or same-session outcomes to estimate their personal mean.

---

## Step 6 — add a common model interface

Create an abstract/lightweight interface:

```python
fit(X_train, y=None, groups=None)
predict(X_test)
predict_proba(X_test)  # where meaningful
score_holdout(...)
get_model_metadata()
```

For unsupervised models, `predict` should mean projection/component assignment based on a model fitted to training data, not refitting on test data.

Implement M0 first.

### M0 minimum

A general-performance latent baseline can initially be:

```text
standardised core behavioural features
→ one-factor/PCA representation
→ held-out density or reconstruction/predictive score
```

The exact likelihood implementation can be refined later. What matters initially is having a deterministic baseline under the same folds as later models.

Do not attach cognitive labels to M0 components.

---

## Step 7 — implement the first automated manifest

Every test run and formal run should be able to emit something like:

```json
{
  "protocol": "flow-zone-validation-v1",
  "git_commit": "...",
  "configs": {"model_tournament": "sha256:..."},
  "upstream": {"flow-zone-zone-validation": "2d8d..."},
  "python": "3.12.x",
  "seed": 20260807,
  "split": "participant_group_kfold",
  "models": ["M0_general_performance"]
}
```

Generated manifests should contain no participant identifiers.

---

## Step 8 — stop and review before M1–M4

The first milestone PR should contain only:

- package scaffold;
- configuration loading;
- evidence registry validation;
- canonical schema;
- synthetic infrastructure fixture;
- leakage-safe splits;
- normative residualisation baseline;
- M0 interface/baseline;
- tests;
- manifest writer.

Do not add cPCA/CVQ yet unless the above is complete and reviewed.

---

# Acceptance tests for milestone 1

Run:

```bash
python -m pytest -q
```

The milestone passes only if tests demonstrate:

```text
schema failures are caught;
participant leakage is impossible through the public split API;
dataset holdout is exact;
normative preprocessing fitted on training data can transform untouched test data;
future sessions are not used to construct a person's earlier personal baseline;
run manifests are deterministic except for expected timestamp fields;
M0 can fit and score the synthetic fixture end-to-end.
```

---

# Prompt to give Codex in VS Code

Use this as the first implementation prompt:

> Read `AGENTS.md`, `CLAIMS_BOUNDARY.md`, `docs/FLOW_ZONE_VALIDATION_V1.md`, `docs/DATA_CONTRACT_V1.md`, `docs/CODEX_FIRST_STEPS.md`, and the YAML files in `config/` before making changes. Implement Milestone 1 only. Create a Python 3.12 `src/trident_validation` package with config/provenance handling, canonical window-schema validation, deterministic participant-isolated and dataset-holdout split utilities, a leakage-safe first-pass normative trait-state residualiser, an M0 general-performance model behind a common model interface, run-manifest generation, a realistic fully synthetic infrastructure fixture, and tests. Do not implement M1–M5, cPCA, CVQ, CVAE, HMMs, graph models or Attention Coach transport yet. Preserve structural missingness, never split a participant across train/test, never estimate personal baselines using future sessions, keep all discovery labels neutral, and make all formal randomness explicit and reproducible. Run the full test suite and summarise any scientific or implementation decisions that remain ambiguous rather than silently choosing them.

