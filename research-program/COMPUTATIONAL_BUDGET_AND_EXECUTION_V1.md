# Computational Budget and Execution Strategy V1

**Status:** Canonical execution guidance for the Trident Control Architecture Validation research programme  
**Applies to:** M2.6 onward  
**Companion to:** `research-program/RESEARCH_PROGRAM_V2.md`  
**Purpose:** Reduce unnecessary compute and avoid long blocking Codex/VS Code runs without weakening scientific validity, reproducibility or falsifiability.

---

# 1. Core principle

The research programme should spend compute where competing explanations are genuinely difficult to distinguish.

Do **not** begin a new experiment by exhaustively fitting every model across every replicate, stress level and diagnostic output at maximum scale.

Use a staged execution policy:

```text
SMOKE
Is the code correct and end-to-end?
        ↓
PILOT
Is the experiment informative and are obvious failures visible?
        ↓
CONFIRMATORY
Estimate the registered scientific quantity with adequate precision.
```

The compute strategy must never be used to hide ambiguous or unfavourable scientific results.

---

# 2. Three run profiles

| Profile | Purpose | Typical scale | Runtime target | Codex may launch? |
|---|---|---:|---:|---|
| **Smoke** | Verify code, schemas, manifests, leakage guards and end-to-end output | 1–2 replicates/world; reduced N if permitted by fixture | <2 minutes | Yes |
| **Pilot** | Detect obvious model confusion, performance bottlenecks and execution failures | ~10–20 replicates/world | <5–10 minutes | Yes, if bounded and observable |
| **Confirmatory** | Produce the registered scientific estimate | Sequential/versioned batches or fixed registered schedule | Run as resumable jobs, ideally <10–15 minutes per shard | No blocking Codex run |

A formal confirmatory run should normally be launched from a terminal or job runner and monitored through progress/checkpoint files. Codex should prepare, test and report the command rather than remain blocked waiting for a long run.

---

# 3. Hard execution rules

1. **No invisible long runs.** Any job expected to exceed 2 minutes must report progress at least every 10–20 seconds.
2. **No blocking Codex formal runs above 5 minutes.** Codex should stop after smoke/pilot validation and provide the formal-run command.
3. **Every formal experiment must be resumable.** Completed replicate units must not be recomputed after interruption.
4. **Every formal experiment must have a preflight.** Before launch, report expected datasets, model fits, workers, estimated runtime and approximate disk usage.
5. **Checkpoint at the smallest scientifically independent unit.** For synthetic studies this is normally `world × replicate × condition`.
6. **Use atomic writes.** A partially written replicate must never be mistaken for a completed replicate.
7. **Preserve deterministic seed schedules.** Resume and parallel execution must use exactly the same child seeds as serial execution.
8. **Bound parallelism.** Start with 2–4 process workers unless benchmark evidence justifies more.
9. **Prevent thread oversubscription.** Worker processes should normally set numerical-library threads to 1 (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
10. **Formal results must be reproducible from a clean tree.** Operational optimisation may not silently alter the scientific model, data generator, split policy or scoring contract.

---

# 4. Separate scientific design from numerical implementation

The following are **engineering optimisations** and can be introduced without changing the scientific experiment, provided numerical equivalence is tested:

```text
vectorising likelihood calculations
removing duplicate likelihood passes
batching rows and quadrature nodes
reusing split-level preprocessing
caching immutable design matrices
bounded multiprocessing across independent replicates
checkpoint/resume
primary-only scoring during bulk simulations
post-hoc diagnostic expansion only where registered
progress logging
preflight/runtime estimation
```

The following are **scientific changes** and require an explicit versioned protocol amendment before results are inspected:

```text
changing replicate counts or stopping rules
changing synthetic-world parameters
changing sample sizes
changing stress grids
changing model definitions
changing priors or hyperparameters
changing feature sets
changing train/test split rules
changing the primary scoring metric
changing recovery thresholds
```

Do not disguise a scientific change as a compute optimisation.

---

# 5. Immediate optimisation targets in M0–M4

## 5.1 Vectorise M3/M4 mixture fitting

The present static Gaussian-mixture implementation uses Python loops over rows/components during the E-step and then performs another near-duplicate pass to obtain total log likelihood.

Target implementation:

```text
rows × components × features
→ vectorised component log densities
→ one log-sum-exp normaliser
→ responsibilities + row log likelihoods from the same calculation
```

Acceptance rule:

```text
old and optimised implementations
must agree within a registered numerical tolerance
on deterministic fixtures.
```

Do not alter `n_components`, EM convergence rules or scientific model form merely to gain speed.

## 5.2 Batch M2 quadrature

M2 currently evaluates a 31-point quadrature for held-out rows. Preserve the 31-point scientific specification but evaluate rows/nodes in arrays rather than nested Python loops where possible.

Acceptance rule:

```text
same deterministic fixture
→ same held-out log-density values within tolerance
→ materially reduced runtime
```

## 5.3 Reuse split-level preprocessing

Where scientifically equivalent, fit the common training feature adapter once per split and pass immutable transformed training/test arrays to compatible models.

Do not permit a shared adapter to erase legitimate model-specific missingness or likelihood assumptions.

## 5.4 Primary-only bulk scoring

For large recovery studies, the default bulk replicate should calculate only what is needed for registered selection and recovery:

```text
primary held-out log density
model winner / uncertainty
replicate metadata
```

Compute expensive representation/diagnostic outputs only where required, for example:

```text
M3/W3 and M4/W4 component recovery
selected audit replicates
final aggregate reports
registered diagnostic subsets
```

This is allowed only if the omitted diagnostics are not part of the registered selection rule.

---

# 6. Sequential precision for future experiments

For **new, not-yet-run experiments**, prefer pre-registered sequential precision to arbitrary large fixed replicate counts when statistically appropriate.

Illustrative pattern:

```text
minimum replicates = 30
batch size = 20
target confidence-interval half-width = 0.10
maximum replicates = 150
```

Execution:

```text
30 replicates
→ estimate target + CI
→ precision adequate?
   yes → stop
   no  → +20 replicates
          → reassess
```

Rules:

- Freeze the stopping rule **before** inspecting confirmatory results.
- Use a maximum replicate cap.
- Report the realised replicate count and stopping reason.
- Do not stop because a preferred model is winning.
- Stop only because the pre-specified uncertainty/precision criterion is met.

**M2.6 exception:** the currently registered formal M2.6 design remains unchanged unless it is explicitly cancelled and replaced by a new version before formal results are inspected. M2.6 may be accelerated by engineering optimisation, sharding and resume support without altering its scientific schedule.

---

# 7. Adaptive stress testing for future milestones

Do not automatically run a full dense grid of every nuisance parameter.

Prefer boundary-finding designs where scientifically appropriate:

```text
BASELINE
   ↓
WORST PLAUSIBLE CONDITION
   ↓
recovery still adequate?
   yes → no further levels needed for this factor
   no  → evaluate midpoint/intermediate level
          → locate degradation boundary
```

Examples:

```text
missingness
observation noise
source shift
participant count
windows/session
latent separation
```

Rules:

- Stress boundaries and candidate levels must be specified before the confirmatory run.
- The procedure must not selectively search only conditions favourable to a model.
- Report all tested conditions, including failures.
- Use one-factor-at-a-time stress tests unless an interaction is itself the registered scientific question.

---

# 8. Compute gating across the research programme

The V2 programme is deliberately hierarchical. Richer models should run only after simpler alternatives fail the relevant gate.

## M2.6 — Formal static recovery

```text
optimise numerics
→ checkpoint/resume
→ run registered W0–W4 × M0–M4 schedule
→ freeze recovery/confusion results
```

Do not change the registered scientific design merely because runtime is inconvenient.

## M2.7 — Empirical-twin static recovery

Start with:

```text
baseline empirical nuisance calibration
+
pre-specified worst plausible nuisance level
```

Add intermediate levels only when needed to locate a recovery boundary.

## M4 — Mechanistic synthetic tournament

Run one identifiability question at a time.

Preferred structure:

```text
trait vs state
THEN
vigilance relationships
THEN
APC vs PACE
```

Do not fit every hierarchical architecture to every synthetic world in one giant job.

## M5 — Public mechanism tournament

Fit only architectures that survived synthetic identifiability.

Do not spend compute on mechanisms that the available task cannot identify.

## M6 — Dynamic-regime tournament

Use increasing complexity:

```text
R0 static/no regime
→ R1 continuous AR
→ R2 HMM/HSMM
→ R3 adaptive corridor
→ R4 MI-lock + entropy-excess corridor
→ R5 literal cusp
```

If a simpler model survives the registered predictive/transport gates and richer models add no meaningful held-out value, stop escalation.

## M7 — Contrastive validation

```text
cPCA/CVQ first
→ only then CVAE if its gate is met
→ graph contrastive models only after stable transition topology exists
```

## M9/M10 — App transport/intervention

Use a small number of frozen hypotheses. Do not run a large exploratory model search on the limited app sample.

---

# 9. Deterministic sharding and resume contract

A confirmatory synthetic run should have a pre-generated immutable schedule:

```text
run_id
world_id
condition_id
replicate_id
child_seed
split_seed
model_seed(s)
```

Each scheduled unit has one terminal status:

```text
pending
running
complete
failed
```

A completed unit stores:

```text
primary scores
selection result
minimum registered diagnostics
runtime
software/version metadata
checksum
```

Resume behaviour:

```text
read immutable schedule
→ verify config hashes
→ skip checksum-valid completed units
→ rerun only pending/failed units
```

Serial and parallel execution must aggregate to the same scientific result within deterministic/numerical tolerance.

---

# 10. Progress and preflight output

Every long-running command should begin with a preflight summary similar to:

```text
Protocol: formal_synthetic_recovery_v1
Worlds: 5
Replicates: 200/world
Conditions: 1 baseline + registered stress cells
Models: 5
Expected synthetic datasets: ...
Expected model fits: ...
Workers: 4
BLAS threads/worker: 1
Estimated runtime: ...
Checkpoint path: ...
Resume supported: yes
```

During execution print, at least every 10–20 seconds:

```text
M2.6 | W2 | baseline
42/80 scheduled units complete
210/400 model fits complete
elapsed 03:18
ETA 02:57
failures 0
```

A silent console for >30 minutes is an execution defect even if computation is technically still progressing.

---

# 11. Runtime benchmarking gate

Before a confirmatory run:

1. benchmark one representative replicate of each expensive model;
2. benchmark one complete world replicate across all candidate models;
3. estimate total runtime from the immutable schedule;
4. identify the dominant cost centre;
5. optimise only after verifying numerical equivalence;
6. rerun benchmark;
7. launch formal shards only when runtime is operationally acceptable.

Keep benchmark results separate from scientific outcome reports.

---

# 12. Recommended default worker policy

Initial local default:

```text
process workers: 2
BLAS/OpenMP threads per worker: 1
```

Increase to 3–4 only after benchmarking shows improved throughput without memory pressure or UI freezing.

Avoid nested parallelism.

The scientific unit of parallelisation should normally be the independent replicate, not an internal row loop.

---

# 13. Failure and timeout policy

A model fit or replicate should not hang indefinitely.

For every model family define a reasonable execution timeout based on benchmark evidence.

If exceeded:

```text
record failure
record model/world/seed/condition
continue other independent units
retain failure in final report
```

Do **not** silently replace the failed result or exclude it from denominators without a registered rule.

Repeated timeouts are evidence of an implementation or model-stability problem that must be diagnosed before scaling further.

---

# 14. Scientific efficiency principle

The preferred programme is not:

```text
fit every conceivable model everywhere
```

It is:

```text
simple model
→ falsification gate
→ richer model only if needed
→ synthetic identifiability
→ public-data transport
→ prospective app prediction
```

Likewise, compute should answer:

> Where do the competing explanations become distinguishable, and where does that distinction fail?

rather than exhaustively filling a combinatorial grid.

---

# 15. Codex operating instruction

When asked to implement or run a research milestone, Codex must:

```text
1. identify the requested run profile: smoke / pilot / confirmatory;
2. run preflight before any job expected to exceed 2 minutes;
3. estimate model-fit count and runtime;
4. refuse to launch a blocking confirmatory run expected to exceed 5 minutes;
5. ensure progress reporting, checkpointing and resume are available;
6. use deterministic bounded parallelism;
7. preserve scientific definitions and registered seed schedules;
8. report whether each proposed speed-up is engineering-only or scientifically protocol-changing;
9. run numerical-equivalence tests after optimisation;
10. provide the exact terminal command for the user to launch/resume the formal job.
```

The objective is a research programme that is computationally efficient, interruptible and observable **without becoming scientifically opportunistic**.
