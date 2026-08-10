# M6.1b HCP-YA Transversal Sensitivity Protocol

**Status:** post-M6.1 exploratory sensitivity addendum

This addendum does not alter the frozen M6.1 protocol, config, split, numerical
analysis or report. It was formulated after M6.1 outcome inspection and is
therefore not a confirmatory pre-outcome test.

## Question

Are the M6.1 broad-K and C/V demand-pattern results robust to bounded
K-construction sensitivity checks?

## Boundary

The analysis may test only:

```text
K robustness under bounded indicator subsets
C/V demand-pattern increments using the registered full K reference
```

The analysis does not test or confirm:

```text
g
cognitive control as a latent mechanism
vigilance as a latent mechanism
binding
relational processing
WM updating
strategic A/T/PC policy variables
Trident-G
APC
PACE
bottlenecks
transfer
criticality
```

## Cohort And Validation

The cohort remains the official HCP `100 Unrelated Subjects` group used in
M6.1. The split safeguard remains
`participant_isolated_official_hcp_100_unrelated`.

Frozen validation parameters:

```text
n_folds: 5
split_seed: 20260822
bootstrap_seed: 20260824
bootstrap_iterations: 1000
ridge_alpha: 1.0
```

All indicator scaling and model fitting occur inside training folds only.
Participant-level HCP data, fold assignments, predictions, score rows, Subject
IDs and residuals must not be committed.

## Outcomes

The analysis uses only the M6.1 eligible outcomes:

```text
ListSort_Unadj
PMAT24_A_CR
CardSort_Unadj
```

It does not use:

```text
WM_Task_2bk_Acc
Relational_Task_Acc
PMAT24_A_RTCR
```

## K Sensitivity Variants

Each K variant is constructed inside each training fold by standardising its
indicators using training-fold mean/SD, orienting higher as better and taking an
unweighted arithmetic mean.

Registered sensitivity variants:

```text
registered_full_k: ProcSpeed_Unadj, PicSeq_Unadj, ReadEng_Unadj, PicVocab_Unadj
no_proc_speed: PicSeq_Unadj, ReadEng_Unadj, PicVocab_Unadj
no_picseq: ProcSpeed_Unadj, ReadEng_Unadj, PicVocab_Unadj
no_readeng: ProcSpeed_Unadj, PicSeq_Unadj, PicVocab_Unadj
no_picvocab: ProcSpeed_Unadj, PicSeq_Unadj, ReadEng_Unadj
language_crystallized_pair: ReadEng_Unadj, PicVocab_Unadj
speed_sequence_pair: ProcSpeed_Unadj, PicSeq_Unadj
```

The pair diagnostics are not new confirmed constructs.

## Models And Contrasts

For every K variant and outcome:

```text
S0: Y ~ intercept
S1: Y ~ K_variant
K_variant_increment: S1 - S0
```

For the registered full K reference only:

```text
S2: Y ~ K + C
S3: Y ~ K + V
S4: Y ~ K + C + V
C_increment_beyond_registered_K: S2 - S1
V_increment_beyond_registered_K: S3 - S1
joint_CV_increment_beyond_registered_K: S4 - S1
```

The primary score is participant-isolated held-out predictive log density per
participant. RMSE, MAE, held-out R2 and held-out correlation are secondary
descriptives.

## Interpretation

This is a stress test of whether the broad K signal appears dependent on a
single indicator family, especially processing speed. It cannot decide whether
K is a confirmed g factor.

C/V increments are demand-pattern diagnostics only. A larger C increment for
CardSort than for ListSort or PMAT is compatible with control-demand recruitment
but does not establish a control factor.

The available data do not identify binding, relational processing, WM updating
or strategic policy variables. Those require later task designs that vary
operator demand and/or context deliberately.
