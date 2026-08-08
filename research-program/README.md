# Research Programme

This folder contains the canonical forward research strategy for the Trident Control Architecture Validation project.

## Canonical documents

### `RESEARCH_PROGRAM_V2.md`

Defines the scientific programme:

```text
synthetic identifiability and falsification
→ public cognitive-database model constraint
→ prospective Attention / WM Coach transport
→ personalised intervention testing
```

It defines the working variable architecture (`K, V, C, A, T, PC, R, P, Y, Transfer`) and the revised M2.6–M10 milestone sequence.

### `COMPUTATIONAL_BUDGET_AND_EXECUTION_V1.md`

Defines the computational execution strategy:

```text
smoke
→ pilot
→ confirmatory
```

It requires preflight/runtime estimation, visible progress, checkpoint/resume, bounded parallelism, deterministic sharding, numerical-equivalence testing for optimisation, and gated escalation to richer models.

The compute strategy is intended to reduce unnecessary model fits and avoid long blocking Codex/VS Code runs **without altering registered scientific questions or hiding failures**.

## Operating rule

Read both documents before implementing or launching a new milestone.

Where scientific ambition and computational cost conflict:

1. preserve the registered scientific question;
2. optimise implementation first;
3. stage the experiment into smoke/pilot/confirmatory profiles;
4. use pre-registered sequential precision or adaptive stress-boundary designs for future experiments where valid;
5. escalate to richer models only when simpler alternatives fail the relevant gate.

`AGENTS.md` makes these requirements mandatory for Codex work in this repository.
