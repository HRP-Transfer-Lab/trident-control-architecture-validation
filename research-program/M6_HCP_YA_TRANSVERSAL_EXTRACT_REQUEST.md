# M6 HCP-YA Transversal Extract Request

**Status:** M6.0a source-schema and construct-hardened local data-preparation request

This document defines the participant-level HCP-YA behavioural extract needed
for the transversal K/C_candidate/V support preflight. It does not contain
participant data and does not authorise model fitting.

Participant-level HCP data must not be committed to Git.

## Target Local File

```text
data/processed/hcp_ya_transversal_extract.csv
```

This path is ignored by repository policy.

## Required Identity Columns

```text
Subject
Family_ID
```

`Family_ID` is preferred because HCP-YA contains twins/siblings and the primary
analysis requires family-isolated cross-validation.

If restricted family structure is unavailable, prepare an unrelated-only extract
and add a boolean unrelated indicator column. The analysis config must then be
amended before fitting so the unrelated-only safeguard is explicit.

Ordinary random participant folds are not allowed.

## Required HCP Source Columns

```text
Flanker_Unadj
ProcSpeed_Unadj
CardSort_Unadj
PicSeq_Unadj
ReadEng_Unadj
PicVocab_Unadj
SCPT_SEN
SCPT_SPEC
ListSort_Unadj
WM_Task_2bk_Acc
PMAT24_A_CR
PMAT24_A_RTCR
```

The schema records both `hcp_source_column` and `canonical_column` in:

```text
config/hcp_ya_transversal_extract_schema_v1.yaml
```

If the local HCP export uses a different source name for a registered construct,
update the source mapping in a separate pre-outcome schema commit before
outcome fitting. Do not manually rename after inspecting outcome associations.

Optional, non-blocking relational-processing behavioural column if available:

```text
Relational_Task_Acc
```

## Construct Boundary

The extract supports:

```text
K: broad non-target/general performance candidates
C_candidate: Flanker inhibitory-control/attention candidate
V: Short Penn CPT sustained-attention candidate
```

Primary K is deliberately limited to:

```text
ProcSpeed_Unadj
PicSeq_Unadj
ReadEng_Unadj
PicVocab_Unadj
```

Do not include Flanker, Card Sort, List Sorting, 2-back, PMAT or HCP global
cognition composites such as CogTotalComp in K.

The HCP Flanker variable is a behavioural `C_candidate`, not a confirmed
`C_signal` mechanism.

Held-out outcomes:

```text
attention/control: CardSort_Unadj
working memory: ListSort_Unadj
working memory: WM_Task_2bk_Acc
reasoning: PMAT24_A_CR
reasoning secondary RT: PMAT24_A_RTCR
reasoning/relational optional: Relational_Task_Acc
```

The anti-circularity rule is hard:

```text
an outcome column cannot define K/C/V for the same outcome domain
```

## Template Command

Create an empty local CSV template:

```powershell
$env:PYTHONPATH='src'

python -m trident_validation.mechanistic.hcp_extract_schema `
  --schema config/hcp_ya_transversal_extract_schema_v1.yaml `
  --write-template
```

Then populate the local file from authorised HCP-YA data access.

## Preflight Command

After populating the local extract:

```powershell
$env:PYTHONPATH='src'

python -m trident_validation.mechanistic.hcp_transversal `
  --config config/hcp_ya_transversal_v1.yaml `
  --data-path data/processed/hcp_ya_transversal_extract.csv `
  --preflight-only `
  --output-dir reports/generated/hcp_ya_transversal_v1
```

No HCP outcome model should be fit until this support preflight passes and a
separate analysis-freeze commit enables model fitting.
