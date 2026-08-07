# Contrastive Framework

## Purpose

The contrastive layer asks whether scientifically targeted contrasts expose structure that is hidden by dominant average variation.

## Registered Contrasts

- State deviation versus personal background.
- High versus low control demand.
- Transition versus stable occupancy.
- Low versus preserved vigilance.
- Recovery versus non-recovery.

## Methods

- cPCA: identifies directions enriched in target variation relative to background variation.
- CVQ: quantifies contrastive variance and guards against leakage.
- CVAE: gated extension used only if sample-size and synthetic-recovery requirements pass.

## Leakage Controls

Contrasts must not use downstream labels, intervention outcomes or future-session information when building representations intended for prospective prediction.

