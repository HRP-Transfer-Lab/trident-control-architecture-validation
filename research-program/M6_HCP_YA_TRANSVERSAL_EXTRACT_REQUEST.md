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

Optional, non-blocking n-back aggregate column:

```text
WM_Task_2bk_Acc
```

This released aggregate is flagged as `optional_known_issue_pending` because an
HCP Users thread in February 2026 raised an unresolved concern about the
calculation of `WM_Task_Acc`, `WM_Task_0bk_Acc` and `WM_Task_2bk_Acc`. The
primary support gate does not require it, and it should not be used for
scientific inference unless HCP publishes a resolution or the measure is
independently reconstructed from task/run/trial-level files in a later
prospective amendment.

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
working memory optional/future: WM_Task_2bk_Acc
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

## Canonical Build Command

After downloading an authorised local HCP-YA behavioural CSV/TSV/Parquet export,
build the canonical ignored extract without manual renaming:

```powershell
$env:PYTHONPATH='src'

python -m trident_validation.mechanistic.hcp_extract_builder `
  --source path\to\authorised_hcp_export.csv `
  --schema config/hcp_ya_transversal_extract_schema_v1.yaml `
  --output data/processed/hcp_ya_transversal_extract.csv `
  --summary reports/generated/hcp_ya_transversal_v1/extract_build_summary.json `
  --force
```

The builder writes only registered canonical columns, refuses missing required
source columns, fills absent optional columns as empty, and writes a
participant-free build summary. The output CSV remains local/ignored and must
not be committed.

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
