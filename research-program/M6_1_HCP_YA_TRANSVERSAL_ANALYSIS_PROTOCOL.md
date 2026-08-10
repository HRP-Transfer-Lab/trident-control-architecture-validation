# M6.1 HCP-YA Transversal K/C Candidate/V Analysis Protocol

**Status:** pre-outcome frozen analysis protocol

This protocol authorises one exploratory HCP-YA unrelated-subject analysis only
after this protocol, config, implementation and mock-only tests are committed
and pushed. It does not authorise confirmatory architecture claims.

## Boundary

The analysis tests whether the same lower-level K/C_candidate/V coordinates
provide out-of-sample predictive information across working memory and fluid
reasoning, with attention/control as a secondary near-domain comparison.

The analysis does not confirm g, a latent cognitive-control mechanism, a latent
vigilance mechanism, Trident-G, APC, PACE, W-specific capacity, bottlenecks,
criticality or transfer.

The cohort is the official HCP `100 Unrelated Subjects` group. The split
safeguard is `participant_isolated_official_hcp_100_unrelated`. This does not
permit ordinary participant folds for arbitrary or full HCP cohorts.

Participant-level HCP data, predictions, fold assignments and subject IDs must
not be committed.

## Frozen Constructs

K_candidate is the unweighted arithmetic mean of these indicators after
training-fold mean/SD standardisation:

```text
ProcSpeed_Unadj
PicSeq_Unadj
ReadEng_Unadj
PicVocab_Unadj
```

C_candidate is `Flanker_Unadj`, standardised inside each training fold.

V_candidate is the unweighted arithmetic mean of `SCPT_SEN` and `SCPT_SPEC`
after training-fold mean/SD standardisation.

All indicators are oriented so higher is better. No data-fitted indicator
weights are allowed.

These columns are not allowed to construct K/C/V:

```text
CardSort_Unadj
ListSort_Unadj
PMAT24_A_CR
PMAT24_A_RTCR
WM_Task_2bk_Acc
Relational_Task_Acc
HCP global cognition composites
```

## Outcomes

Primary outcomes:

```text
ListSort_Unadj
PMAT24_A_CR
```

Secondary near-domain outcome:

```text
CardSort_Unadj
```

Diagnostic-only field:

```text
PMAT24_A_RTCR
```

`PMAT24_A_RTCR` is not combined with PMAT accuracy. `WM_Task_2bk_Acc` remains
scientifically ineligible.

## Model Set

For each eligible outcome, compare exactly:

```text
M0: Y ~ intercept
M1: Y ~ K
M2: Y ~ K + C
M3: Y ~ K + V
M4: Y ~ K + C + V
M5: Y ~ K + C + V + C_by_V
```

M5 is secondary only and low precision. It is not the default preferred model
because it is more flexible.

The prediction model is a simple ridge-Gaussian linear model. All construct
standardisation, outcome standardisation for the pooled secondary analysis, and
model fitting happen inside training folds only.

## Validation And Metrics

Validation uses 5-fold participant-isolated cross-validation.

Frozen seeds:

```text
split_seed: 20260822
bootstrap_seed: 20260823
bootstrap_iterations: 1000
ridge_alpha: 1.0
```

Primary score:

```text
participant-isolated held-out predictive log density per participant
```

Participant-level paired score deltas are used only in memory to compute
aggregate contrasts and bootstrap intervals. They are not written.

Secondary metrics:

```text
RMSE
MAE
held-out R2
held-out correlation if well-defined
```

## Registered Contrasts

```text
K_increment: M1 - M0
C_increment_beyond_K: M2 - M1
V_increment_beyond_K: M3 - M1
joint_CV_increment_beyond_K: M4 - M1
interaction_increment: M5 - M4
```

The interaction increment is secondary only.

## Secondary Domain-Weight Test

The secondary pooled test stacks `CardSort_Unadj`, `ListSort_Unadj` and
`PMAT24_A_CR`. Each outcome is standardised inside the training fold for its
domain. All three rows for a participant remain in the same fold.

Compare:

```text
common slope: Y_standardised ~ domain + K + C + V
domain-specific slope: Y_standardised ~ domain + K + C + V + domain_by_K + domain_by_C + domain_by_V
```

The purpose is to ask whether allowing relative K/C/V weights to vary by domain
materially improves held-out predictive density. This is secondary because
N=100 limits precision. Domain-specific weights are not interpreted as separate
domain-specific latent constructs.

## Interpretation Categories

Broad transversal K signal is encouraging if M1 improves held-out prediction
over M0 for both List Sorting and PMAT accuracy. This is not a confirmed g
claim.

C and V increments beyond K are reported separately by domain. The same
variable need not matter equally at each representational level.

Different domain weights are allowed by the hypothesis. A larger C increment
for attention/control than reasoning is not by itself a failure of the
transversal hypothesis.

## M6.2 And M6.3 Gates

M6.2 remains:

```text
blocked_pending_independent_second_WM_indicator
```

The current HCP support contains only one scientifically eligible WM outcome,
`ListSort_Unadj`. A W-specific or WM-capacity claim requires at least two
independent eligible WM indicators.

M6.3 remains blocked until M6.2 establishes a defensible independent
layer-specific WM candidate. K_by_W and C_by_W bottleneck tests are not run.

ListSort residual variation may be reported only as descriptive unexplained
outcome variation. It must not be called W, WM capacity, a layer-specific
factor or a bottleneck.
