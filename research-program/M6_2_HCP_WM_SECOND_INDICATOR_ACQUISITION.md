# M6.2 HCP WM Second Indicator Acquisition

**Status:** acquisition required before support preflight

**Scientific analysis performed:** false

**Model fitting performed:** false

No ListSort-2Back association, correlation, regression, residual transport
model, bottleneck test or HCP outcome analysis was inspected or run.

## Purpose

M6.2 requires an independently reconstructed HCP 2-back behavioural measure for
the same official HCP 100 Unrelated Subjects cohort used in M6.1.

The unresolved `WM_Task_2bk_Acc` aggregate concern remains active. That field is
not admissible for scientific inference and was not used.

## Local Support Search

Search date: 2026-08-10

Searched local locations:

```text
C:\Users\admin\Downloads
data/
current repository tree
C:\Users\admin\OneDrive\Documents
```

Search terms included HCP, WM, 2bk, 2back, nback, EPrime, tfMRI and Working.

Local HCP files found:

| File | Status for M6.2 |
|---|---|
| `C:\Users\admin\Downloads\HCP_YA_subjects_2026_08_10_13_26_13.csv` | subject-level M6.1 BALSA export only; not trial/run-level WM behaviour |
| `C:\Users\admin\Downloads\HCP_YA_subjects_2026_08_10_13_26_49.csv` | subject-level M6.1 BALSA export only; not trial/run-level WM behaviour |
| `data/processed/hcp_ya_transversal_extract.csv` | ignored local canonical M6.1 extract; contains ineligible empty `WM_Task_2bk_Acc` field only |

No local HCP task/run/trial-level WM behavioural source was found. Therefore
the M6.2 support preflight cannot be run yet.

## Required HCP Acquisition

Download the official HCP-YA working-memory task behavioural source files for
the same official HCP 100 Unrelated Subjects cohort.

The required source must be participant-level linkable but retained only in
ignored local paths. It must contain all HCP working-memory task runs needed to
reconstruct participant-level 2-back accuracy independently from trial or event
records.

Minimum required data elements:

```text
Subject identifier
task/run identifier for each HCP WM run
condition identifier separating 2-back from 0-back trials or blocks
trial/event validity or exclusion flags
stimulus/event records sufficient to define scored trials
participant response or response code
correct response or explicit correctness field
missing/no-response coding
reaction time for correct responses, if available
```

Acceptable source form:

```text
official HCP task-fMRI Working Memory behavioural/E-Prime/tabular trial files
for all WM task runs covering the official HCP 100 Unrelated Subjects cohort
```

Not sufficient:

```text
subject-level HCP behavioural aggregate exports only
WM_Task_2bk_Acc
imaging EV/onset files without response/correctness information
files lacking Subject linkage
files lacking 2-back versus 0-back condition identifiers
```

## Prospective Reconstruction Rule

Primary reconstructed variable:

```text
HCP_2Back_Reconstructed_Acc
```

Formula:

```text
valid scored 2-back trials answered correctly
/
valid scored 2-back trials with admissible response/correctness coding
```

Aggregation:

```text
compute valid 2-back accuracy across all eligible HCP WM runs for each Subject
after run/trial validity checks pass
```

Optional diagnostics only:

```text
HCP_0Back_Reconstructed_Acc
HCP_2Back_Reconstructed_RTCR
```

The optional diagnostics must not be composited with ListSort and must not be
used to define K, C_candidate or V.

## Required Future Support Checks

A later M6.2 support-preflight implementation must fail unless all checks pass:

```text
source files exist and are SHA-256 hashed
exact Subject overlap is verified against the existing official 100 unrelated cohort
no duplicate Subject aggregation is possible
all required WM runs are accounted for or exclusions are explicit
valid trial counts are checked before aggregation
invalid, missing and no-response trials are handled by declared rules
2-back and 0-back records are separated before aggregation
participant-level reconstructed output is written only to ignored local paths
tracked reports contain no Subject IDs and no participant-level values
```

## Current Support Decision

M6.2 second WM indicator support status:

```text
blocked_pending_HCP_WM_task_behavioural_file_acquisition
```

M6.2 scientific model fitting remains forbidden.
