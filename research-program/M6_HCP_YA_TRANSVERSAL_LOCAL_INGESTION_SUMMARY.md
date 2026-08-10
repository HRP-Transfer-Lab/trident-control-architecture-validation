# M6 HCP-YA Transversal Local Ingestion Summary

**Status:** participant-free local ingestion record

No participant-level HCP rows, subject identifiers, scientific associations or
outcome models are recorded here.

## Source Exports

| File | SHA-256 | Rows | Unique subjects | Subject unique |
|---|---|---:|---:|---:|
| `HCP_YA_subjects_2026_08_10_13_26_13.csv` | `sha256:98098863d9e00e667e48b384b7d84d66c2f3454ed2b800c775950a859a1754e7` | 100 | 100 | true |
| `HCP_YA_subjects_2026_08_10_13_26_49.csv` | `sha256:83ae8655eb60d2f341a6021cbc8f78c4430aca9fa766170b306b8d421902d223` | 100 | 100 | true |

## Source Column Coverage

The first source export supplied:

```text
Subject
PicSeq_Unadj
CardSort_Unadj
Flanker_Unadj
PMAT24_A_CR
PMAT24_A_RTCR
ReadEng_Unadj
PicVocab_Unadj
ProcSpeed_Unadj
ListSort_Unadj
```

The second source export supplied:

```text
Subject
SCPT_SEN
SCPT_SPEC
```

Both files also contained non-registered MRI/MEG completion columns, which were
not written to the canonical extract. The duplicate non-registered completion
columns in the second source were dropped during the merge.

## Merge Validation

- Cohort mode: `hcp_100_unrelated`
- Official HCP 100 Unrelated Subjects provenance declared: true
- Merge key: `Subject`
- Source row counts identical: true
- Subject sets identical: true
- Duplicate `Subject` values present: false
- Merge status: `one_to_one_subject_merge_validated`
- Merged rows: 100
- Merged unique subjects: 100

## Canonical Extract

- Local ignored path: `data/processed/hcp_ya_transversal_extract.csv`
- Local extract SHA-256: `sha256:b43456153602ece24b2e8ea283b68e4faca3146f2a9b373588913704ae86069b`
- Canonical rows: 100
- Canonical columns: 16
- Participant-level data in Git allowed: false
- Model fitting allowed: false

Canonical columns:

```text
Subject
Family_ID
Unrelated
ProcSpeed_Unadj
PicSeq_Unadj
ReadEng_Unadj
PicVocab_Unadj
Flanker_Unadj
SCPT_SEN
SCPT_SPEC
CardSort_Unadj
ListSort_Unadj
WM_Task_2bk_Acc
PMAT24_A_CR
PMAT24_A_RTCR
Relational_Task_Acc
```

Absent optional fields filled empty:

```text
Family_ID
Unrelated
WM_Task_2bk_Acc
Relational_Task_Acc
```
